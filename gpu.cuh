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
void RMSD(
    const float* __restrict__ dst,   // reordered coordinates: X,Y,Z blocks
    int N_frames,
    int N_atoms,
    float*out
);

__global__ 
void runKMedoidsGPU(
    int N_frames,
    int K,
    const float* __restrict__ rmsd,
    int MAX_ITER,
    int* centroidsGPU, 
    int* clustersGPU,
    float* frameCosts
);

__global__ 
void updateCentroidsGPU(
    int N_frames,
    int K,
    int* centroidsGPU, 
    int* clustersGPU,
    float* frameCosts
);



#endif // GPU_CUH
