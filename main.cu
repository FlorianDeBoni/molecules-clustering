/*
    Compile with:
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 \
    main.cu FileUtils.cpp gpu.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o main
*/

#include "FileUtils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include "gpu.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iomanip>
#include <utils.h>


int main(int argc, char** args) {

    // Time measurements
    chrono_type global_start = chrono_time::now();


    int K = 10;
    int MAX_ITER = 50;

    std::string file_name;
    if (argc >= 2) {
        file_name = args[1];
    } else {
        std::cerr<< "Argument for dataset binary file missing, check the Makefile" << std::endl;
        throw std::invalid_argument("Requested frames exceed available frames");
    }
    FileUtils file(file_name); 

    // std::cout << file << std::endl;

    size_t N_frames = 15000;
    // size_t N_frames = file.getN_frames();
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);    

    measure_seconds(global_start, "Loading source data");

    // Copy reordered CPU → GPU
    chrono_type mem_transfer_start = chrono_time::now();
    float* frameGPU;
    CHECK_SUCCESS(cudaMalloc(&frameGPU, total_size), "Allocating frameGPU");
    CHECK_SUCCESS(cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice), "Memcpy frame -> frameGPU");

    // Allocate RMSD matrix
    float* rmsd;
    size_t size_rmsd = N_frames * N_frames * sizeof(float);
    CHECK_SUCCESS(cudaMalloc(&rmsd, size_rmsd), "Allocating rmsd vector on GPU");

    cudaDeviceSynchronize();
    measure_seconds(mem_transfer_start, "CPU to GPU memory transfer");

    dim3 threads(8,64);
    dim3 blocks((N_frames + threads.x - 1) / threads.x, 
                (N_frames + threads.y - 1) / threads.y);

    chrono_type rmsd_kernel_start = chrono_time::now();
    RMSD<<<blocks, threads>>>(
        frameGPU,
        N_frames,
        N_atoms,
        rmsd
    );
    CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD Kernel");
    measure_seconds(rmsd_kernel_start, "RMSD Kernel");

    float* rmsdHost = new float[N_frames*N_frames];
    CHECK_SUCCESS(cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Memcpy rmsd -> rmsdHost");

    // ---------------- CLUSTERING GPU -----------------
    chrono_type clustering_loop_start = chrono_time::now();

    int* clusters = new int[N_frames];
    int* clustersGPU;
    CHECK_SUCCESS(cudaMalloc(&clustersGPU, N_frames * sizeof(int)), "Allocating clustersGPU");

    // Pick first K unique indices
    int* centroids = new int[K];
    int* centroidsGPU;
    CHECK_SUCCESS(cudaMalloc(&centroidsGPU, K * sizeof(int)), "Allocating centroidsGPU");
    pickRandomCentroids(N_frames, K, centroids);
    CHECK_SUCCESS(cudaMemcpy(centroidsGPU, centroids, K*sizeof(int), cudaMemcpyHostToDevice), "Memcpy centroids -> centroidsGPU");

    // costs for each centroid candidate
    float* frameCostsGPU;
    CHECK_SUCCESS(cudaMalloc(&frameCostsGPU, N_frames * sizeof(float)), "Allocating frameCostsGPU");

    measure_seconds(clustering_loop_start, "==> Clustering memory setup");

    // Assignment step params
    dim3 clusteringThreads(1024);
    dim3 clusteringBlocks(1 + ((N_frames - 1) / (threads.x)));

    // Centroids update step params
    dim3 threadsPerClusterBlock(512);
    dim3 reducingBlocks(K);
    size_t sharedMemSize = threadsPerClusterBlock.x * (sizeof(float) + sizeof(int));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        runKMedoidsGPU<<<clusteringBlocks, clusteringThreads>>>(
            N_frames,
            K,
            rmsd,
            MAX_ITER,
            centroidsGPU,
            clustersGPU,
            frameCostsGPU
        );

        updateCentroidsGPU<<<K, threadsPerClusterBlock, sharedMemSize>>>(
            N_frames,
            K,
            centroidsGPU,
            clustersGPU,
            frameCostsGPU
        );
    }
    measure_seconds(clustering_loop_start, "==> Clustering Total time");

    CHECK_SUCCESS(cudaMemcpy(centroids, centroidsGPU, K * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy centroidsGPU -> centroids");
    CHECK_SUCCESS(cudaMemcpy(clusters, clustersGPU, N_frames * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy clustersGPU -> clusters");

    float db_index = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdHost);
    // float db_index = runKMedoids(N_frames, K, rmsdHost, MAX_ITER, centroids, clusters);
    // float db_index = k_analysis(rmsdHost, N_frames, MAX_ITER);

    std::cout << "Davies–Bouldin index: " << db_index << std::endl;

    measure_seconds(global_start, "Entire program");

    // Print db for random clustering
    float random_db_index = runRandomClustering(N_frames, K, rmsdHost);
    std::cout << "Random Davies–Bouldin index: " << random_db_index << std::endl;

    saveClusters(clusters, N_frames, centroids, K);

    // Cleanup
    delete[] frame;
    delete[] centroids;
    delete[] rmsdHost;
    delete[] clusters;
    cudaFree(frameGPU);
    cudaFree(rmsd);

    return 0;
}
