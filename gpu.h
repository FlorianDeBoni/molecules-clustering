#include <cuda_runtime.h>
#include <iostream>

#define CHECK_SUCCESS(exp, msg) {if ((exp) != cudaSuccess) { \
                                    std::cout << "Failed : " << msg << '\n'; \
                                    exit(1); }\
                                }\

void allocateOnGPU(float** GPU_dptr, size_t mem_bytes);
void freeOnGPU(float* ptr);