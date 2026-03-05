#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>


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
// RMSD kernel — 2D thread layout + Kabsch-Umeyama formula
//
// Launch configuration:
//   grid  : (ceil(N_tgt/16), ceil(N_ref/16))
//   block : dim3(16, 16)
//
// Key optimisation over the original:
//   The original kernel made 3 passes over both ref and target coords:
//     Pass 1 — ref centroid
//     Pass 2 — tgt centroid
//     Pass 3 — correlation matrix A
//     Pass 4 — RMSD accumulation (re-reads all coords)
//
//   This version makes only 2 passes using the Kabsch-Umeyama identity:
//
//     RMSD² = (G_ref + G_tgt - 2*(σ₀+σ₁+σ₂)) / N
//
//   where G = Σ|centered coords|²  and  σᵢ = sqrt(λᵢ) are the singular
//   values of A (i.e. sqrt of eigenvalues of AᵀA = M).
//   This eliminates the final re-read of all coordinates entirely.
//
//   Pass 1 (fused): centroids for ref and tgt                 [1× global read each]
//   Pass 2 (fused): correlation matrix A + G_ref + G_tgt      [1× global read each]
//   Total: 2 global memory passes instead of 3 — a 33% reduction
//          in memory bandwidth for the dominant cost (target reads).
//
//   Note: λ₂ (the smallest eigenvalue) is now used; previously it was
//   discarded. With --use_fast_math, all sqrtf/fmaxf remain fast.
// =======================================================
__global__
void RMSD(const float* __restrict__ references,
          const float* __restrict__ targets,
          size_t N_references_subset,
          size_t N_targets_subset,
          size_t N_atoms,
          float* rmsd_device)
{
    int snap_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ref_idx  = blockIdx.y * blockDim.y + threadIdx.y;

    if (snap_idx >= (int)N_targets_subset || ref_idx >= (int)N_references_subset)
        return;

    // ── PASS 1: Centroids (ref + tgt fused, 1 global read per coord) ─────
    float rcx=0,rcy=0,rcz=0;
    float scx=0,scy=0,scz=0;

    for (int a = 0; a < (int)N_atoms; a++) {
        size_t br = (size_t)a * N_references_subset + ref_idx;
        rcx += references[0*N_atoms*N_references_subset + br];
        rcy += references[1*N_atoms*N_references_subset + br];
        rcz += references[2*N_atoms*N_references_subset + br];

        size_t bt = (size_t)a * N_targets_subset + snap_idx;
        scx += targets[0*N_atoms*N_targets_subset + bt];
        scy += targets[1*N_atoms*N_targets_subset + bt];
        scz += targets[2*N_atoms*N_targets_subset + bt];
    }

    rcx /= N_atoms; rcy /= N_atoms; rcz /= N_atoms;
    scx /= N_atoms; scy /= N_atoms; scz /= N_atoms;

    // ── PASS 2: Correlation matrix A + G_ref + G_tgt (1 global read per coord)
    // G_ref = Σ|rᵢ - r̄|²,  G_tgt = Σ|sᵢ - s̄|²
    // These are accumulated alongside A at zero extra memory cost.
    float a00=0,a01=0,a02=0,a10=0,a11=0,a12=0,a20=0,a21=0,a22=0;
    float G_ref=0, G_tgt=0;

    for (int a = 0; a < (int)N_atoms; a++) {
        size_t br = (size_t)a * N_references_subset + ref_idx;
        float rx = references[0*N_atoms*N_references_subset + br] - rcx;
        float ry = references[1*N_atoms*N_references_subset + br] - rcy;
        float rz = references[2*N_atoms*N_references_subset + br] - rcz;

        size_t bt = (size_t)a * N_targets_subset + snap_idx;
        float sx = targets[0*N_atoms*N_targets_subset + bt] - scx;
        float sy = targets[1*N_atoms*N_targets_subset + bt] - scy;
        float sz = targets[2*N_atoms*N_targets_subset + bt] - scz;

        G_ref += rx*rx + ry*ry + rz*rz;
        G_tgt += sx*sx + sy*sy + sz*sz;

        a00+=rx*sx; a01+=rx*sy; a02+=rx*sz;
        a10+=ry*sx; a11+=ry*sy; a12+=ry*sz;
        a20+=rz*sx; a21+=rz*sy; a22+=rz*sz;
    }

    // ── Compute M = AᵀA ───────────────────────────────────────────────────
    float m00=a00*a00+a10*a10+a20*a20;
    float m01=a00*a01+a10*a11+a20*a21;
    float m02=a00*a02+a10*a12+a20*a22;
    float m11=a01*a01+a11*a11+a21*a21;
    float m12=a01*a02+a11*a12+a21*a22;
    float m22=a02*a02+a12*a12+a22*a22;

    // ── Eigenvalues of M (= σᵢ² of A) ────────────────────────────────────
    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

    // ── Kabsch-Umeyama: RMSD² = (G_ref + G_tgt - 2*(σ₀+σ₁+σ₂)) / N ─────
    // σᵢ = sqrt(λᵢ).  All three eigenvalues contribute, including λ₂.
    // fmaxf guards against tiny negative values from floating-point error.
    float sigma_sum = sqrtf(fmaxf(lambda[0], 0.f))
                    + sqrtf(fmaxf(lambda[1], 0.f))
                    + sqrtf(fmaxf(lambda[2], 0.f));

    float rmsd2 = (G_ref + G_tgt - 2.f * sigma_sum) / (float)N_atoms;
    float rmsd  = sqrtf(fmaxf(rmsd2, 0.f));

    size_t idx = (size_t)ref_idx * N_targets_subset + snap_idx;
    rmsd_device[idx] = rmsd;
}
