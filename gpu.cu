#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

// =============================================================================
// Eigenvalues of a symmetric 3×3 matrix (analytical, Cardano's method).
// Returns lambda[0] >= lambda[1] >= lambda[2].
// =============================================================================
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean  = trace / 3.0f;

    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;

    float p = sm00*sm00 + sm11*sm11 + sm22*sm22
            + 2.0f * (m01*m01 + m02*m02 + m12*m12);
    p = __fsqrt_rn(p / 6.0f);

    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;

    float b00 = sm00 * invp,  b01 = m01 * invp,  b02 = m02 * invp;
    float b11 = sm11 * invp,  b12 = m12 * invp,  b22 = sm22 * invp;

    float det = b00 * (b11*b22 - b12*b12)
              - b01 * (b01*b22 - b12*b02)
              + b02 * (b01*b12 - b11*b02);
    det *= 0.5f;
    det  = fminf(1.0f, fmaxf(-1.0f, det));

    float phi = acosf(det) / 3.0f;

    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + 2.0f * CUDART_PI_F / 3.0f);
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];

    // Sort descending
    if (lambda[0] < lambda[1]) { float t = lambda[0]; lambda[0] = lambda[1]; lambda[1] = t; }
    if (lambda[1] < lambda[2]) { float t = lambda[1]; lambda[1] = lambda[2]; lambda[2] = t; }
    if (lambda[0] < lambda[1]) { float t = lambda[0]; lambda[0] = lambda[1]; lambda[1] = t; }
}

// =============================================================================
// computeCentroidsG
//   One thread per frame (idx).
//   Data layout: coords[dim * N_atoms * N_frames + atom * N_frames + frame]
// =============================================================================
__global__
void computeCentroidsG(const float* __restrict__ coords,
                       size_t N_atoms,
                       size_t N_frames,
                       float* __restrict__ cx,
                       float* __restrict__ cy,
                       float* __restrict__ cz,
                       float* __restrict__ G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_frames) return;

    float sx = 0.f, sy = 0.f, sz = 0.f;
    for (int a = 0; a < N_atoms; ++a) {
        size_t off = a * N_frames + idx;
        sx += coords[0 * N_atoms * N_frames + off];
        sy += coords[1 * N_atoms * N_frames + off];
        sz += coords[2 * N_atoms * N_frames + off];
    }
    sx /= N_atoms;  sy /= N_atoms;  sz /= N_atoms;

    cx[idx] = sx;  cy[idx] = sy;  cz[idx] = sz;

    float g = 0.f;
    for (int a = 0; a < N_atoms; ++a) {
        size_t off = a * N_frames + idx;
        float rx = coords[0 * N_atoms * N_frames + off] - sx;
        float ry = coords[1 * N_atoms * N_frames + off] - sy;
        float rz = coords[2 * N_atoms * N_frames + off] - sz;
        g += rx*rx + ry*ry + rz*rz;
    }
    G[idx] = g;
}

// =============================================================================
// RMSD kernel
//
// Grid:  blocks = ( ceil(N_tgt/32), ceil(N_ref/8) )
// Block: threads = (32, 8)  → tx indexes target frames, ty indexes ref frames
//
// Shared memory layout (allocated by caller as 3*TILE*blockDim.y floats):
//   TILE = blockDim.x = 32  (atoms tiled along tx dimension)
//   s_ref_x[atom_in_tile * blockDim.y + ty]  — ref coords after centroid sub
//   s_ref_y[...]
//   s_ref_z[...]
//
// Each thread (tx, ty) accumulates the 3×3 cross-covariance matrix A between
// ref frame r=blockIdx.y*8+ty and tgt frame t=blockIdx.x*32+tx, then applies
// the QCP formula:  RMSD² = (G_ref + G_tgt - 2*Σ√λᵢ) / N_atoms
// =============================================================================
__global__
void RMSD(const float* __restrict__ refs,
          const float* __restrict__ tgts,
          size_t N_atoms,
          size_t N_ref,
          size_t N_tgt,
          const float* __restrict__ cx_ref,
          const float* __restrict__ cy_ref,
          const float* __restrict__ cz_ref,
          const float* __restrict__ G_ref,
          const float* __restrict__ cx_tgt,
          const float* __restrict__ cy_tgt,
          const float* __restrict__ cz_tgt,
          const float* __restrict__ G_tgt,
          float* __restrict__ rmsd)
{
    extern __shared__ float smem[];

    // TILE must equal blockDim.x so that each tx-lane loads exactly one atom
    // per tile into shared memory.  The caller allocates smem accordingly:
    //   smem_bytes = 3 * blockDim.x * blockDim.y * sizeof(float)
    const int TILE = blockDim.x;   // = 32

    float* s_ref_x = smem;
    float* s_ref_y = s_ref_x + TILE * blockDim.y;
    float* s_ref_z = s_ref_y + TILE * blockDim.y;

    int r = blockIdx.y * blockDim.y + threadIdx.y;   // ref frame index
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // tgt frame index

    if (r >= N_ref || t >= N_tgt) return;

    float rcx = cx_ref[r],  rcy = cy_ref[r],  rcz = cz_ref[r];
    float scx = cx_tgt[t],  scy = cy_tgt[t],  scz = cz_tgt[t];

    float a00=0,a01=0,a02=0;
    float a10=0,a11=0,a12=0;
    float a20=0,a21=0,a22=0;

    for (int start = 0; start < N_atoms; start += TILE)
    {
        int atom_idx = start + threadIdx.x;

        // Thread (tx, ty) loads ref atom (start+tx) for ref frame r(ty)
        // into shared memory slot [tx * blockDim.y + ty].
        if (atom_idx < N_atoms)
        {
            s_ref_x[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[0 * N_atoms * N_ref + atom_idx * N_ref + r] - rcx;
            s_ref_y[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[1 * N_atoms * N_ref + atom_idx * N_ref + r] - rcy;
            s_ref_z[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[2 * N_atoms * N_ref + atom_idx * N_ref + r] - rcz;
        }

        __syncthreads();

        int tile_end = min(TILE, (int)(N_atoms - start));
        for (int k = 0; k < tile_end; ++k)
        {
            // Ref atom (start+k) for this thread's ref frame: from shared mem.
            float rx = s_ref_x[k * blockDim.y + threadIdx.y];
            float ry = s_ref_y[k * blockDim.y + threadIdx.y];
            float rz = s_ref_z[k * blockDim.y + threadIdx.y];

            // Tgt atom (start+k) for this thread's tgt frame: global read.
            // All blockDim.y threads with the same tx share the same address
            // and hit L1 cache.
            int atom_k = start + k;
            float sx = tgts[0 * N_atoms * N_tgt + atom_k * N_tgt + t] - scx;
            float sy = tgts[1 * N_atoms * N_tgt + atom_k * N_tgt + t] - scy;
            float sz = tgts[2 * N_atoms * N_tgt + atom_k * N_tgt + t] - scz;

            a00 += rx*sx;  a01 += rx*sy;  a02 += rx*sz;
            a10 += ry*sx;  a11 += ry*sy;  a12 += ry*sz;
            a20 += rz*sx;  a21 += rz*sy;  a22 += rz*sz;
        }

        __syncthreads();
    }

    // Symmetric matrix M = Aᵀ·A  (3×3, upper triangle sufficient)
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00, m01, m02, m11, m12, m22, lambda);

    float sigma_sum = __fsqrt_rn(fmaxf(lambda[0], 0.f))
                    + __fsqrt_rn(fmaxf(lambda[1], 0.f))
                    + __fsqrt_rn(fmaxf(lambda[2], 0.f));

    float rmsd2 = (G_ref[r] + G_tgt[t] - 2.f * sigma_sum) / N_atoms;
    rmsd[r * N_tgt + t] = __fsqrt_rn(fmaxf(rmsd2, 0.f));
}

__global__
void RMSD_diagonal(const float* __restrict__ refs,
          size_t N_atoms,
          size_t N_ref,
          const float* __restrict__ cx_ref,
          const float* __restrict__ cy_ref,
          const float* __restrict__ cz_ref,
          const float* __restrict__ G_ref,
          float* __restrict__ rmsd)
{
    extern __shared__ float smem[];

    // TILE must equal blockDim.x so that each tx-lane loads exactly one atom
    // per tile into shared memory.  The caller allocates smem accordingly:
    //   smem_bytes = 3 * blockDim.x * blockDim.y * sizeof(float)
    const int TILE = blockDim.x;   // = 32
    const int TILE_y = blockDim.y;

    float* s_ref_x = smem;
    float* s_ref_y = s_ref_x + TILE * blockDim.y;
    float* s_ref_z = s_ref_y + TILE * blockDim.y;

    int r = blockIdx.y * blockDim.y + threadIdx.y;   // ref frame index
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // tgt frame index

    if (r >= N_ref || t >= N_ref) return;
    if (TILE * (blockIdx.x + 1) < TILE_y * blockIdx.y) return;


    // if (t_max < r_min) return;

    // bool compute = (t >= r);
    // compute = compute && (!(r >= N_ref || t >= N_ref));

    float rcx = cx_ref[r],  rcy = cy_ref[r],  rcz = cz_ref[r];
    float scx = cx_ref[t],  scy = cy_ref[t],  scz = cz_ref[t];

    float a00=0,a01=0,a02=0;
    float a10=0,a11=0,a12=0;
    float a20=0,a21=0,a22=0;

    for (int start = 0; start < N_atoms; start += TILE)
    {
        int atom_idx = start + threadIdx.x;

        // Thread (tx, ty) loads ref atom (start+tx) for ref frame r(ty)
        // into shared memory slot [tx * blockDim.y + ty].
        if (atom_idx < N_atoms)
        {
            s_ref_x[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[0 * N_atoms * N_ref + atom_idx * N_ref + r] - rcx;
            s_ref_y[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[1 * N_atoms * N_ref + atom_idx * N_ref + r] - rcy;
            s_ref_z[threadIdx.x * blockDim.y + threadIdx.y] =
                refs[2 * N_atoms * N_ref + atom_idx * N_ref + r] - rcz;
        }

        __syncthreads();

        int tile_end = min(TILE, (int)(N_atoms - start));
        for (int k = 0; k < tile_end; ++k)
        {
            // Ref atom (start+k) for this thread's ref frame: from shared mem.
            float rx = s_ref_x[k * blockDim.y + threadIdx.y];
            float ry = s_ref_y[k * blockDim.y + threadIdx.y];
            float rz = s_ref_z[k * blockDim.y + threadIdx.y];

            // Tgt atom (start+k) for this thread's tgt frame: global read.
            // All blockDim.y threads with the same tx share the same address
            // and hit L1 cache.
            int atom_k = start + k;
            float sx = refs[0 * N_atoms * N_ref + atom_k * N_ref + t] - scx;
            float sy = refs[1 * N_atoms * N_ref + atom_k * N_ref + t] - scy;
            float sz = refs[2 * N_atoms * N_ref + atom_k * N_ref + t] - scz;

            a00 += rx*sx;  a01 += rx*sy;  a02 += rx*sz;
            a10 += ry*sx;  a11 += ry*sy;  a12 += ry*sz;
            a20 += rz*sx;  a21 += rz*sy;  a22 += rz*sz;
        }

        __syncthreads();
    }

    // Symmetric matrix M = Aᵀ·A  (3×3, upper triangle sufficient)
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;
    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00, m01, m02, m11, m12, m22, lambda);
    float sigma_sum = sqrtf(fmaxf(lambda[0], 0.f))
                    + sqrtf(fmaxf(lambda[1], 0.f))
                    + sqrtf(fmaxf(lambda[2], 0.f));
    float rmsd2 = (G_ref[r] + G_ref[t] - 2.f * sigma_sum) / N_atoms;
    rmsd2 = sqrtf(fmaxf(rmsd2, 0.f));
    // rmsd[r * N_ref + t] = sqrtf(fmaxf(rmsd2, 0.f));
    rmsd[r * N_ref + t] = rmsd2;
}

__global__
void AssignClusters(
    int N_frames,
    int K,
    const float* __restrict__ rmsd,
    int* centroidsGPU,
    int* clustersGPU,
    float* frameCosts
)
{
    // Assigning each frame to a cluster
    int frame_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (frame_id >= N_frames) {
        return;
    }

    int best_cluster = 0;
    float best_d = HUGE_VALF;
    for (int k = 0; k<K; k++) {
        // size_t distance_idx = (size_t) N_frames * centroidsGPU[k] + frame_id;
        // float curr_distance = rmsd[distance_idx];
        float curr_distance = getRMSD_GPU(centroidsGPU[k], frame_id, rmsd, N_frames);
        if (curr_distance < best_d) {
            best_cluster = k;
            best_d = curr_distance;
        }
    }
    clustersGPU[frame_id] = best_cluster;
}

__global__
void ComputeMedoidCosts(
    int N_frames,
    const float* __restrict__ rmsd,
    int* clustersGPU,
    float* frameCosts
)
{
    int frame_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (frame_id >= N_frames) {
        return;
    }

    int assigned_cluster = clustersGPU[frame_id];

    float cost = 0;
    for (int j = 0; j<N_frames; j++) {
        if (assigned_cluster == clustersGPU[j]) {
            // size_t idx = (size_t) N_frames * j + frame_id;
            // cost += rmsd[idx];
            // losing some coalescence since using triangular matrix
            cost += getRMSD_GPU(j, frame_id, rmsd, N_frames);
        }
    }
    frameCosts[frame_id] = cost;
}


__global__
void UpdateMedoids(
    int N_frames,
    int* centroidsGPU,
    int* clustersGPU,
    float* frameCosts
)
{
    int k = blockIdx.x;
    int tid = threadIdx.x;

    // Shared mem : [costs | indices]
    extern __shared__ float smem[];
    int* s_indices = (int*)&smem[blockDim.x];


    // Calcul des minima locaux pour chaque thread
    float local_min_cost = HUGE_VALF;
    int local_min_idx = -1;

    for (int i = tid; i < N_frames; i += blockDim.x) {
        if (clustersGPU[i] == k) {
            // parcours divergent coalescence pas sûre
            float cost = frameCosts[i];
            if (cost < local_min_cost) {
                local_min_cost = cost;
                local_min_idx = i;
            }
        }
    }

    // Store to shared memory
    smem[tid] = local_min_cost;
    s_indices[tid] = local_min_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smem[tid + stride] < smem[tid]) {
                smem[tid] = smem[tid + stride];
                s_indices[tid] = s_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 updates the centroid
    if (tid == 0) {
        centroidsGPU[k] = s_indices[0];
    }
}