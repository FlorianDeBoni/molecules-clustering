#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// =======================================================
// Eigenvector solver
// =======================================================
__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float &x, float &y, float &z)
{
    float a00 = m00 - lambda;
    float a01 = m01;
    float a02 = m02;
    float a10 = m01;
    float a11 = m11 - lambda;
    float a12 = m12;
    float a20 = m02;
    float a21 = m12;
    float a22 = m22 - lambda;

    float b0 = -a02;
    float b1 = -a12;
    float det = a00 * a11 - a01 * a10;

    if (fabsf(det) > 1e-8f) {
        x = (b0 * a11 - b1 * a01) / det;
        y = (a00 * b1 - a10 * b0) / det;
        z = 1.0f;
    } else {
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;

        if (fabsf(det) > 1e-8f) {
            x = (b0 * a22 - b1 * a02) / det;
            y = 1.0f;
            z = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;

            if (fabsf(det) > 1e-8f) {
                x = 1.0f;
                y = (b0 * a12 - b1 * a02) / det;
                z = (a01 * b1 - a11 * b0) / det;
            } else {
                x = 1.0f;
                y = 0.0f;
                z = 0.0f;
            }
        }
    }

    float norm = sqrtf(x*x + y*y + z*z);
    if (norm > 1e-8f) {
        x /= norm; y /= norm; z /= norm;
    }
}

// =======================================================
// Eigenvalues solver (symmetric 3x3)
// =======================================================
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;

    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;

    float p = sm00*sm00 + sm11*sm11 + sm22*sm22
            + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);

    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;

    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;

    float det = b00*(b11*b22 - b12*b12)
              - b01*(b01*b22 - b12*b02)
              + b02*(b01*b12 - b11*b02);
    det *= 0.5f;
    det = fminf(1.0f, fmaxf(-1.0f, det));

    float phi = acosf(det) / 3.0f;

    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + 2.0f * CUDART_PI_F / 3.0f);
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];

    if (lambda[0] < lambda[1]) { float t=lambda[0];lambda[0]=lambda[1];lambda[1]=t; }
    if (lambda[1] < lambda[2]) { float t=lambda[1];lambda[1]=lambda[2];lambda[2]=t; }
    if (lambda[0] < lambda[1]) { float t=lambda[0];lambda[0]=lambda[1];lambda[1]=t; }
}

// =======================================================
// RMSD kernel — shared memory + __half for reference coords
//
// Launch configuration:
//   grid  : (N_references_subset)   — one block per reference frame
//   block : (BLOCK_SIZE)            — e.g. 256 threads, 1-D
//   smem  : rmsd_smem_bytes(N_atoms)
//           = 3*N_atoms*sizeof(__half)   — centered reference coords
//           + 3*sizeof(float)            — reference centroid (float[3])
//
// Each block:
//   1. Thread 0 reads its reference frame from global memory to compute
//      the centroid (scalar loop, one thread only — no redundancy).
//      Centroid is stored in the float[3] tail of smem.
//   2. All threads cooperatively load (raw - centroid) into the __half
//      portion of smem in one coalesced pass — ~27 kB total.
//   3. Each thread handles one or more target frames (grid-stride):
//        Pass A — target centroid (one pass over global mem)
//        Pass B — correlation matrix A + RMSD sum fused (one pass over
//                 global mem for targets; reference reads smem only)
//
// Target coords are read exactly twice per frame (passes A and B).
// Reference coords are read from global memory exactly once (step 1+2).
// All arithmetic is in float32; __half is storage only.
// =======================================================
__global__
void RMSD(const float* __restrict__ references,
          const float* __restrict__ targets,
          size_t N_references_subset,
          size_t N_targets_subset,
          size_t N_atoms,
          float* rmsd_device)
{
    // ── Shared memory layout ──────────────────────────────────────────────
    // [0 .. 3*N_atoms*sizeof(__half))  : centered reference coords (__half)
    // [aligned ..+3*sizeof(float))     : reference centroid (float[3])
    //
    // Total: ~27 kB for N_atoms~4693 — fits in 48 kB smem limit.
    extern __shared__ __half s_ref_cen[];   // 3 * N_atoms halfs (centered)

    // Centroid stored in float[3] at the aligned end of the __half array.
    size_t half_bytes   = 3 * N_atoms * sizeof(__half);
    size_t offset_bytes = (half_bytes + alignof(float) - 1)
                          & ~(alignof(float) - 1);
    float* s_centroid   = reinterpret_cast<float*>(
                              reinterpret_cast<char*>(s_ref_cen) + offset_bytes);

    const int ref_idx = blockIdx.x;
    const int tid     = threadIdx.x;
    const int bdim    = blockDim.x;

    if (ref_idx >= (int)N_references_subset) return;

    // ── Step 0a: Thread 0 computes reference centroid from global memory ──
    // Fix Issue 1: only ONE thread does this work (was 256× redundant before).
    // Reading from global memory here is unavoidable — we need the centroid
    // before we can store centered coords into smem.
    if (tid == 0) {
        float cx = 0.f, cy = 0.f, cz = 0.f;
        for (size_t a = 0; a < N_atoms; a++) {
            cx += references[0 * N_atoms * N_references_subset + a * N_references_subset + ref_idx];
            cy += references[1 * N_atoms * N_references_subset + a * N_references_subset + ref_idx];
            cz += references[2 * N_atoms * N_references_subset + a * N_references_subset + ref_idx];
        }
        s_centroid[0] = cx / (float)N_atoms;
        s_centroid[1] = cy / (float)N_atoms;
        s_centroid[2] = cz / (float)N_atoms;
    }
    __syncthreads();

    // All threads read centroid from smem.
    const float rcx = s_centroid[0];
    const float rcy = s_centroid[1];
    const float rcz = s_centroid[2];

    // ── Step 0b: Cooperatively load centered reference coords into smem ───
    // Fix Issue 3: use size_t throughout to avoid int overflow.
    // Fix Issue 2 (partial): store (raw - centroid) so per-target loops
    // never re-read raw reference coords or repeat the subtraction.
    const size_t total_coords = 3 * N_atoms;
    for (size_t i = tid; i < total_coords; i += bdim) {
        size_t dim  = i / N_atoms;
        size_t atom = i % N_atoms;
        float  cval = (dim == 0) ? rcx : (dim == 1) ? rcy : rcz;
        float  raw  = references[dim * N_atoms * N_references_subset
                                 + atom * N_references_subset
                                 + ref_idx];
        s_ref_cen[i] = __float2half(raw - cval);
    }
    __syncthreads();

    // ── Each thread handles one (or more) target frames ───────────────────
    for (int snap_idx = tid; snap_idx < (int)N_targets_subset; snap_idx += bdim) {

        // ── Pass A: target centroid ───────────────────────────────────────
        // One unavoidable pass over global memory per target frame.
        float scx = 0.f, scy = 0.f, scz = 0.f;
        for (size_t a = 0; a < N_atoms; a++) {
            scx += targets[0 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx];
            scy += targets[1 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx];
            scz += targets[2 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx];
        }
        scx /= (float)N_atoms;
        scy /= (float)N_atoms;
        scz /= (float)N_atoms;

        // ── Pass B (FUSED): correlation matrix A + RMSD sum ───────────────
        // Fix Issue 2: target coords read exactly once per atom per pass.
        // Centered reference coords come from smem (no global read, no subtraction).
        float a00=0,a01=0,a02=0,a10=0,a11=0,a12=0,a20=0,a21=0,a22=0;

        for (size_t a = 0; a < N_atoms; a++) {
            // Reference: already centered, from __half smem
            float rx = __half2float(s_ref_cen[0 * N_atoms + a]);
            float ry = __half2float(s_ref_cen[1 * N_atoms + a]);
            float rz = __half2float(s_ref_cen[2 * N_atoms + a]);

            // Target: centered, read once from global memory
            float sx = targets[0 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scx;
            float sy = targets[1 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scy;
            float sz = targets[2 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scz;

            a00+=rx*sx; a01+=rx*sy; a02+=rx*sz;
            a10+=ry*sx; a11+=ry*sy; a12+=ry*sz;
            a20+=rz*sx; a21+=rz*sy; a22+=rz*sz;
        }

        // Compute M = A^T * A
        float m00=a00*a00+a10*a10+a20*a20;
        float m01=a00*a01+a10*a11+a20*a21;
        float m02=a00*a02+a10*a12+a20*a22;
        float m11=a01*a01+a11*a11+a21*a21;
        float m12=a01*a02+a11*a12+a21*a22;
        float m22=a02*a02+a12*a12+a22*a22;

        // ── Step 3: Eigenvalues ───────────────────────────────────────────
        float lambda[3];
        compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

        // ── Step 4: Eigenvectors ──────────────────────────────────────────
        float3 v0,v1,v2;
        compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[0],v0.x,v0.y,v0.z);
        compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[1],v1.x,v1.y,v1.z);

        // ── Step 5: Orthonormalize V ──────────────────────────────────────
        float d=v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
        v1.x-=d*v0.x; v1.y-=d*v0.y; v1.z-=d*v0.z;

        float n=sqrtf(v1.x*v1.x+v1.y*v1.y+v1.z*v1.z);
        if(n>1e-8f){v1.x/=n;v1.y/=n;v1.z/=n;}

        v2.x=v0.y*v1.z-v0.z*v1.y;
        v2.y=v0.z*v1.x-v0.x*v1.z;
        v2.z=v0.x*v1.y-v0.y*v1.x;

        // ── Step 6: Compute U = A*V ───────────────────────────────────────
        float av0x=a00*v0.x+a01*v0.y+a02*v0.z;
        float av0y=a10*v0.x+a11*v0.y+a12*v0.z;
        float av0z=a20*v0.x+a21*v0.y+a22*v0.z;

        float av1x=a00*v1.x+a01*v1.y+a02*v1.z;
        float av1y=a10*v1.x+a11*v1.y+a12*v1.z;
        float av1z=a20*v1.x+a21*v1.y+a22*v1.z;

        float s0=sqrtf(fmaxf(lambda[0],1e-8f));
        float s1=sqrtf(fmaxf(lambda[1],1e-8f));

        float3 u0={av0x/s0,av0y/s0,av0z/s0};
        float3 u1={av1x/s1,av1y/s1,av1z/s1};
        float3 u2={u0.y*u1.z-u0.z*u1.y,
                   u0.z*u1.x-u0.x*u1.z,
                   u0.x*u1.y-u0.y*u1.x};

        // ── Step 7: Compute R = U*V^T ─────────────────────────────────────
        float R00=u0.x*v0.x+u1.x*v1.x+u2.x*v2.x;
        float R01=u0.x*v0.y+u1.x*v1.y+u2.x*v2.y;
        float R02=u0.x*v0.z+u1.x*v1.z+u2.x*v2.z;

        float R10=u0.y*v0.x+u1.y*v1.x+u2.y*v2.x;
        float R11=u0.y*v0.y+u1.y*v1.y+u2.y*v2.y;
        float R12=u0.y*v0.z+u1.y*v1.z+u2.y*v2.z;

        float R20=u0.z*v0.x+u1.z*v1.x+u2.z*v2.x;
        float R21=u0.z*v0.y+u1.z*v1.y+u2.z*v2.y;
        float R22=u0.z*v0.z+u1.z*v1.z+u2.z*v2.z;

        // ── Step 8: RMSD — second pass over target global mem ────────────
        float sum2 = 0.f;
        for (size_t a = 0; a < N_atoms; a++) {
            float rx = __half2float(s_ref_cen[0 * N_atoms + a]);
            float ry = __half2float(s_ref_cen[1 * N_atoms + a]);
            float rz = __half2float(s_ref_cen[2 * N_atoms + a]);

            float sx = targets[0 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scx;
            float sy = targets[1 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scy;
            float sz = targets[2 * N_atoms * N_targets_subset + a * N_targets_subset + snap_idx] - scz;

            float x = R00*sx + R01*sy + R02*sz;
            float y = R10*sx + R11*sy + R12*sz;
            float z = R20*sx + R21*sy + R22*sz;

            float dx=rx-x, dy=ry-y, dz=rz-z;
            sum2 += dx*dx + dy*dy + dz*dz;
        }

        rmsd_device[(size_t)ref_idx * N_targets_subset + snap_idx] = sqrtf(sum2 / N_atoms);
    }
}
