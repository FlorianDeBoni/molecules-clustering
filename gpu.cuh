#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Unified CUDA error-checking macro
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
// GPU helper — packed upper-triangle index lookup
// ---------------------------------------------------------------------------
inline __device__ float getRMSD_GPU(int i, int j,
                                    const float* rmsdPacked,
                                    int N_snapshots)
{
    if (i == j) return 0.0f;
    if (i > j) { int tmp = i; i = j; j = tmp; }
    size_t idx = (size_t)i * N_snapshots
               - ((size_t)i * ((size_t)i + 1)) / 2
               + (j - i - 1);
    return rmsdPacked[idx];
}

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

__global__
void computeCentroidsG(const float* __restrict__ coords,
                       size_t N_atoms,
                       size_t N_frames,
                       float* __restrict__ centroids_x,
                       float* __restrict__ centroids_y,
                       float* __restrict__ centroids_z,
                       float* __restrict__ G);

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
void RMSD_diagonal(const float* __restrict__ refs,
                   size_t N_atoms,
                   size_t N_ref,
                   const float* __restrict__ cx_ref,
                   const float* __restrict__ cy_ref,
                   const float* __restrict__ cz_ref,
                   const float* __restrict__ G_ref,
                   float* __restrict__ rmsd);

__global__
void AssignClusters(int N_frames,
                    int K,
                    const float* __restrict__ rmsd,
                    int* centroidsGPU,
                    int* clustersGPU,
                    float* frameCosts);

__global__
void ComputeMedoidCosts(int N_frames,
                        const float* __restrict__ rmsd,
                        int* clustersGPU,
                        float* frameCosts);

__global__
void UpdateMedoids(int N_frames,
                   int* centroidsGPU,
                   int* clustersGPU,
                   float* frameCosts);

#endif // GPU_CUH