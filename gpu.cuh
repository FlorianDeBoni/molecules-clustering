#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <iostream>

// Simple macro for CUDA errors
#define CHECK_SUCCESS(exp, msg) { \
    if ((exp) != cudaSuccess) { \
        std::cout << "Failed: " << msg << " (" << cudaGetErrorString(exp) << ")\n"; \
        exit(1); \
    } \
}

// GPU memory helpers
inline void allocateOnGPU(float** GPU_dptr, size_t mem_bytes) {
    CHECK_SUCCESS(cudaMalloc(GPU_dptr, mem_bytes), "cudaMalloc");
}

inline void freeOnGPU(float* ptr) {
    if(ptr) cudaFree(ptr);
}

__global__
void computeCentroidsG(const float* __restrict__ coords, 
                       size_t N_atoms,
                       size_t N_frames,
                       float* __restrict__ centroids_x,
                       float* __restrict__ centroids_y,
                       float* __restrict__ centroids_z,
                       float* __restrict__ G);

__global__
void RMSD_precomputed(const float* __restrict__ refs,
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

// 2D kernel: one thread per (ref, target) pair.
// Uses the Kabsch-Umeyama identity to eliminate the final coord re-read:
//   RMSD^2 = (G_ref + G_tgt - 2*(s0+s1+s2)) / N
// reducing global memory passes from 3 to 2 versus the naive implementation.
__global__
void RMSD(
    const float* __restrict__ references,
    const float* __restrict__ targets,
    size_t N_references_subset,
    size_t N_targets_subset,
    size_t N_atoms,
    float* rmsd_device
);

#endif // GPU_CUH
