#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>

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

__global__
void RMSD(
    const float* __restrict__ references,
    const float* __restrict__ targets,
    size_t N_references_subset,
    size_t N_targets_subset,
    size_t N_atoms,
    float* rmsd_device
);

__global__ 
void runKMedoidsGPU(
    int N_frames,
    int K,
    const float* __restrict__ rmsd,
    int* centroidsGPU, 
    int* clustersGPU,
    float* frameCosts
);

__global__
void computeMedoidCosts(
    int N_frames,
    const float* __restrict__ rmsd,
    const int*   __restrict__ clustersGPU,      // original mapping, read-only
    const int*   __restrict__ sorted_frames,    // frame IDs sorted by cluster
    const int*   __restrict__ cluster_offsets,  // [K+1]
    float* frameCosts
);

__global__ 
void updateCentroidsGPU(
    int N_frames,
    int* centroidsGPU, 
    int* clustersGPU,
    float* frameCosts
);



#endif // GPU_CUH
