/*
    Compile with:
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 \
    reading_coordinates.cu FileUtils.cpp gpu.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o reading_coordinates


    Run:
    ./reading_coordinates
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

    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    std::cout << frame[0] << ", " << frame[1] << ", " << frame[2] << " ... " << frame[N_atoms * N_dims * N_frames -1] << std::endl;

    // Upload reordered data to GPU
    float* frameGPU;
    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);
    cudaMalloc(&frameGPU, total_size);
    cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice);

    // Allocate output snapshot buffer
    float* snapshotGPU;
    size_t snapshot_size = total_size;   // 1 snapshot per frame
    cudaMalloc(&snapshotGPU, snapshot_size);

    // Launch kernel
    int threads = 256;
    int blocks = (N_frames + threads - 1) / threads;

    loadSnapshotKernel<<<blocks, threads>>>(
        frameGPU,
        snapshotGPU,
        N_frames,
        N_atoms,
        N_dims
    );
    cudaDeviceSynchronize();
    std::cout << "Snapshot kernel done.\n";

    // Cleanup
    cudaFree(frameGPU);
    cudaFree(snapshotGPU);
    delete[] frame;
    return 0;
}
