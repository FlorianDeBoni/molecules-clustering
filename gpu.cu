#include "gpu.h"
#include <cuda_runtime.h>


void allocateOnGPU(float** GPU_dptr, size_t mem_bytes) {
    CHECK_SUCCESS(cudaMalloc(GPU_dptr, mem_bytes), "cudaMalloc");
}

void freeOnGPU(float* ptr) {
    CHECK_SUCCESS(cudaFree(ptr), "cudaFree");
}