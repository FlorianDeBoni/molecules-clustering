/*
    Compile with:
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 \
    reading_coordinates.cu FileUtils.cpp gpu.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o reading_coordinates
*/

#include "FileUtils.h"
#include <iostream>
#include "gpu.cuh"

int main() {
    FileUtils file; 

    std::cout << file << std::endl;

    size_t N_frames = file.getN_frames();
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);

    // Copy reordered CPU → GPU
    float* frameGPU;
    cudaMalloc(&frameGPU, total_size);
    cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice);

    // Allocate output 3x3 matrices: one per frame
    float* d_outA;
    cudaMalloc(&d_outA, N_frames * 9 * sizeof(float));

    // Launch computeA
    int threads = 256;
    int blocks = (N_frames + threads - 1) / threads;

    computeA<<<blocks, threads>>>(
        frameGPU,   // reordered coordinates
        d_outA,     // output A matrices
        N_frames,
        N_atoms,
        0           // ref snapshot index
    );
    cudaDeviceSynchronize();

    std::cout << "A Matrix kernel done.\n";

    // Cleanup
    delete[] frame;
    cudaFree(frameGPU);
    cudaFree(d_outA);

    return 0;
}
