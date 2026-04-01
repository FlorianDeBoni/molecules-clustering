#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Unified CUDA error-checking macro (used by both host code and gpu.cu).
// Prints file, line, and the CUDA error string, then exits.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                       \
    } while (0)


// ---------------------------------------------------------------------------
// GPU Helper function to compute equivalent index for an upper triangular matrix
// ---------------------------------------------------------------------------
inline __device__ float getRMSD_GPU(int i, int j, const float* rmsdPacked, int N_snapshots) {
    if (i == j) return 0.0f;
    if (i > j) {
        int tmp = i;
        i = j;
        j = tmp;
    }
    size_t idx = (size_t)i * N_snapshots
           - ((size_t)i * ((size_t)i + 1)) / 2
           + (j - i - 1);
    return rmsdPacked[idx];
}

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Computes per-frame centroid (cx, cy, cz) and the inner-product G = Σ|r-c|²
// for N_frames frames, each with N_atoms atoms.
// Data layout: coords[dim * N_atoms * N_frames + atom * N_frames + frame]
__global__
void computeCentroidsG(const float* __restrict__ coords,
                       size_t N_atoms,
                       size_t N_frames,
                       float* __restrict__ centroids_x,
                       float* __restrict__ centroids_y,
                       float* __restrict__ centroids_z,
                       float* __restrict__ G);

// Computes the N_ref × N_tgt RMSD matrix using the Kabsch/QCP inner-product
// formula.  Shared memory size must be 3 * blockDim.x * blockDim.y * sizeof(float).
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
          float* __restrict__ rmsd);


__global__
void AssignClusters(
    int N_frames,
    int K,
    const float* __restrict__ centroid_rows,  // [K x N_frames], dense
    int* clustersGPU
);

__global__
void ComputeMedoidCosts_Chunk(
    int N_frames,
    int chunk_start,
    int chunk_size,
    const float* __restrict__ rmsd_chunk,  // [chunk_size x N_frames], dense
    const int*   clustersGPU,
    float*       frameCosts,
    bool         reset
);

__global__
void UpdateMedoids(
    int N_frames,
    int* centroidsGPU,
    int* clustersGPU,
    float* frameCosts
);

#endif // GPU_CUH
