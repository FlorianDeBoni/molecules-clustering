// gpu.cu - Complete corrected version
#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>

__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float &x, float &y, float &z)
{
    // Solve the homogeneous system:
    // (m00 - lambda)*x1 + m01*x2 + m02*x3 = 0
    // m01*x1 + (m11 - lambda)*x2 + m12*x3 = 0
    // m02*x1 + m12*x2 + (m22 - lambda)*x3 = 0
    
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

    // z = 1 and solve by inversing the resulting system
    if (fabsf(det) > 1e-8f) {
        x = (b0 * a11 - b1 * a01) / det;
        y = (a00 * b1 - a10 * b0) / det;
        z = 1.0f;
    } else {
        
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;
        
        // y = 1 and solve by inversing the resulting system
        if (fabsf(det) > 1e-8f) {
            x = (b0 * a22 - b1 * a02) / det;
            y = 1.0f;
            z = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;
            
            // x = 1 and solve by inversing the resulting system
            if (fabsf(det) > 1e-8f) {
                x = 1.0f;
                y = (b0 * a12 - b1 * a02) / det;
                z = (a01 * b1 - a11 * b0) / det;
            } else {
                // Fallback
                x = 1.0f;
                y = 0.0f;
                z = 0.0f;
            }
        }
    }

    // Normalize
    float norm = sqrtf(x*x + y*y + z*z);
    if (norm > 1e-8f) {
        x /= norm;
        y /= norm;
        z /= norm;
    }
}

// Robust eigenvalue solver using iterative method
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;
    
    // Shift matrix
    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;
    
    float p = sm00*sm00 + sm11*sm11 + sm22*sm22 + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);
    
    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;
    
    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;
    
    float det = b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02);
    det = det / 2.0f;
    det = fminf(1.0f, fmaxf(-1.0f, det));
    
    float phi = acosf(det) / 3.0f;
    
    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + (2.0f * 3.14159265358979323846f / 3.0f));
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];
    
    // Sort
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
    if (lambda[1] < lambda[2]) { float tmp = lambda[1]; lambda[1] = lambda[2]; lambda[2] = tmp; }
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
}

__global__
void RMSD(
    const float* __restrict__ dst,
    int N_snapshots,
    int N_atoms,
    float* out
)
{

    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    int ref_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute upper triangle: ref_idx < snap
    if (snap >= N_snapshots || ref_idx >= N_snapshots || ref_idx >= snap)
        return;

    // ----------------- STEP 0: Centroids -----------------
    float cx=0.f, cy=0.f, cz=0.f;
    float sx=0.f, sy=0.f, sz=0.f;

    for (int a = 0; a < N_atoms; ++a) {
        size_t idx_ref_x = 0 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_y = 1 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_z = 2 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;

        size_t idx_snap_x = 0 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_y = 1 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_z = 2 * N_atoms * N_snapshots + a * N_snapshots + snap;

        cx += dst[idx_ref_x];
        cy += dst[idx_ref_y];
        cz += dst[idx_ref_z];

        sx += dst[idx_snap_x];
        sy += dst[idx_snap_y];
        sz += dst[idx_snap_z];
    }

    cx /= N_atoms; cy /= N_atoms; cz /= N_atoms;
    sx /= N_atoms; sy /= N_atoms; sz /= N_atoms;

    // ----------------- STEP 1: Correlation matrix A -----------------
    float a00=0.f, a01=0.f, a02=0.f;
    float a10=0.f, a11=0.f, a12=0.f;
    float a20=0.f, a21=0.f, a22=0.f;

    for (int a = 0; a < N_atoms; ++a)
    {
        size_t idx_ref_x = 0 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_y = 1 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_z = 2 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;

        size_t idx_snap_x = 0 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_y = 1 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_z = 2 * N_atoms * N_snapshots + a * N_snapshots + snap;

        float rx = dst[idx_ref_x] - cx;
        float ry = dst[idx_ref_y] - cy;
        float rz = dst[idx_ref_z] - cz;

        float sxv = dst[idx_snap_x] - sx;
        float syv = dst[idx_snap_y] - sy;
        float szv = dst[idx_snap_z] - sz;

        a00 += rx*sxv; a01 += rx*syv; a02 += rx*szv;
        a10 += ry*sxv; a11 += ry*syv; a12 += ry*szv;
        a20 += rz*sxv; a21 += rz*syv; a22 += rz*szv;
    }


    // ----------------- STEP 2: M = A^T * A -----------------
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    // ----------------- STEP 3: Eigenvalues -----------------
    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

    // ----------------- STEP 4: Eigenvectors (V) -----------------
    float3 v0, v1, v2;


    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[0], v0.x,v0.y,v0.z);
    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[1], v1.x,v1.y,v1.z);

    // We compute v2 with Gram-Schmidt later
    // compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[2], vec);
    // float v2x=vec[0], v2y=vec[1], v2z=vec[2];

    // ----------------- STEP 5: Orthonormalize V -----------------
    // Gram-Schmidt v1
    float dot = v1.x*v0.x + v1.y*v0.y + v1.z*v0.z;
    v1.x -= dot*v0.x; v1.y -= dot*v0.y; v1.z -= dot*v0.z;
    float mag = sqrtf(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
    if(mag>1e-8f) { v1.x/=mag; v1.y/=mag; v1.z/=mag; }

    // v2 = v0 × v1
    v2.x = v0.y*v1.z - v0.z*v1.y;
    v2.y = v0.z*v1.x - v0.x*v1.z;
    v2.z = v0.x*v1.y - v0.y*v1.x;

    // ----------------- STEP 6: Compute U = A*V -----------------
    float av0x = a00*v0.x + a01*v0.y + a02*v0.z;
    float av0y = a10*v0.x + a11*v0.y + a12*v0.z;
    float av0z = a20*v0.x + a21*v0.y + a22*v0.z;

    float av1x = a00*v1.x + a01*v1.y + a02*v1.z;
    float av1y = a10*v1.x + a11*v1.y + a12*v1.z;
    float av1z = a20*v1.x + a21*v1.y + a22*v1.z;

    float av2x = a00*v2.x + a01*v2.y + a02*v2.z;
    float av2y = a10*v2.x + a11*v2.y + a12*v2.z;
    float av2z = a20*v2.x + a21*v2.y + a22*v2.z;

    float s0 = (lambda[0]>1e-6f)?sqrtf(lambda[0]):1.f;
    float s1 = (lambda[1]>1e-6f)?sqrtf(lambda[1]):1.f;
    float s2 = (lambda[2]>1e-6f)?sqrtf(lambda[2]):1.f;

    float u0x = av0x/s0, u0y = av0y/s0, u0z = av0z/s0;
    float u1x = av1x/s1, u1y = av1y/s1, u1z = av1z/s1;
    float u2x = av2x/s2, u2y = av2y/s2, u2z = av2z/s2;

    // ----------------- STEP 7: Compute R = U*V^T -----------------
    float R00 = u0x*v0.x + u1x*v1.x + u2x*v2.x;
    float R01 = u0x*v0.y + u1x*v1.y + u2x*v2.y;
    float R02 = u0x*v0.z + u1x*v1.z + u2x*v2.z;

    float R10 = u0y*v0.x + u1y*v1.x + u2y*v2.x;
    float R11 = u0y*v0.y + u1y*v1.y + u2y*v2.y;
    float R12 = u0y*v0.z + u1y*v1.z + u2y*v2.z;

    float R20 = u0z*v0.x + u1z*v1.x + u2z*v2.x;
    float R21 = u0z*v0.y + u1z*v1.y + u2z*v2.y;
    float R22 = u0z*v0.z + u1z*v1.z + u2z*v2.z;

    // ----------------- STEP 8: Compute RMSD -----------------
    float sum2 = 0.f;
    for (int a = 0; a < N_atoms; ++a) {
        size_t idx_ref_x = 0 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_y = 1 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;
        size_t idx_ref_z = 2 * N_atoms * N_snapshots + a * N_snapshots + ref_idx;

        size_t idx_snap_x = 0 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_y = 1 * N_atoms * N_snapshots + a * N_snapshots + snap;
        size_t idx_snap_z = 2 * N_atoms * N_snapshots + a * N_snapshots + snap;

        float rx = dst[idx_ref_x] - cx;
        float ry = dst[idx_ref_y] - cy;
        float rz = dst[idx_ref_z] - cz;

        float sxv = dst[idx_snap_x] - sx;
        float syv = dst[idx_snap_y] - sy;
        float szv = dst[idx_snap_z] - sz;

        float RYx = R00*sxv + R01*syv + R02*szv;
        float RYy = R10*sxv + R11*syv + R12*szv;
        float RYz = R20*sxv + R21*syv + R22*szv;

        float dx = rx - RYx;
        float dy = ry - RYy;
        float dz = rz - RYz;

        sum2 += dx*dx + dy*dy + dz*dz;
    }

    float rmsd = sqrtf(sum2 / N_atoms);
    size_t idx = (size_t)ref_idx * N_snapshots 
               - ((size_t)ref_idx * ((size_t)ref_idx + 1)) / 2 
               + (snap - ref_idx - 1);

    out[idx] = rmsd;
}