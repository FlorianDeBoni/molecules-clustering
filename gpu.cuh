#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// Returns the dynamic shared memory size in bytes required per block.
// Layout:
//   [0 .. 3*N_atoms*sizeof(__half))  centered reference coords (__half)
//   [aligned ..+3*sizeof(float))     reference centroid (float[3])
//
// Total: ~27 kB for N_atoms~4693 — fits within the 48 kB smem limit.
inline size_t rmsd_smem_bytes(size_t N_atoms) {
    size_t half_bytes = 3 * N_atoms * sizeof(__half);
    size_t aligned    = (half_bytes + alignof(float) - 1)
                        & ~(alignof(float) - 1);
    return aligned + 3 * sizeof(float);
}

// 1D kernel: one block per reference frame.
// Smem layout: raw __half coords + centered __half coords + float[3] centroid.
// Thread 0 computes the reference centroid (no redundant work across 256 threads).
// Per-target loop reads target coords exactly twice (centroid pass + RMSD pass).
// Dynamic shared memory must be set at launch: rmsd_smem_bytes(N_atoms).
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
