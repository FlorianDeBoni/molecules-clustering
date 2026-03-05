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

// Returns the shared memory size in bytes required per block for a given N_atoms.
// Each block loads one reference frame as __half: 3 * N_atoms * sizeof(__half)
inline size_t rmsd_smem_bytes(size_t N_atoms) {
    return 3 * N_atoms * sizeof(__half);
}

// 1D kernel: one block per reference frame.
// Each block cooperatively loads its reference into shared memory as __half,
// then every thread handles one or more target frames in a grid-stride loop.
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
