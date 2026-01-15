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
#include <algorithm>
#include <random>
#include "gpu.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iomanip>

// alias for the clock type
using chrono_time = std::chrono::_V2::high_resolution_clock;
using chrono_type = std::chrono::_V2::high_resolution_clock::time_point;


// Takes an std timestamp as start and a measurment title 
// prints out [measurement] : now() - start to std::cout in seconds
void measure_seconds(const chrono_type& start, const std::string& measurement) {
    std::chrono::duration<float> elapsed = chrono_time::now()- start;
    std::cout << std::left  << std::setw(30) << measurement << ": " 
              << std::right << std::setw(10) << elapsed.count() << " s\n";
}

void pickRandomCentroids(int N_frames, int K, int* centroids) {
    // Create a vector with all frame indices
    int* indices = new int[N_frames];
    for (int i = 0; i < N_frames; i++) indices[i] = i;

    // Randomly shuffle
    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t i = N_frames - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(g);
        std::swap(indices[i], indices[j]);
    }

    for (int k = 0; k<K; k++) {
        centroids[k] = indices[k];
    }

    delete[] indices;
}

void createClusters(
    int N_frames,
    int K,
    const float* rmsd,
    const int* centroids,
    int* clusters
) {
    for (int i = 0; i < N_frames; i++) {
        float best = 1e30f;
        int best_k = -1;

        for (int k = 0; k < K; k++) {
            float d = rmsd[centroids[k] * N_frames + i];
            if (d < best) {
                best = d;
                best_k = k;
            }
        }
        clusters[i] = best_k;
    }
}



void updateCentroids(
    int N_frames,
    int K,
    const int* clusters,
    const float* rmsdHost,
    int* centroids
) {
    for (int k = 0; k < K; k++) {
        float best_cost = 1e30f;
        int best_idx = -1;

        for (int i = 0; i < N_frames; i++) {
            if (clusters[i] != k) continue;

            float cost = 0.0f;
            for (int j = 0; j < N_frames; j++) {
                if (clusters[j] != k) continue;
                cost += rmsdHost[i * N_frames + j];
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
        }

        if (best_idx != -1)
            centroids[k] = best_idx;
    }
}

float daviesBouldinIndex(
    int N_frames,
    int K,
    const int* clusters,
    const int* centroids,
    const float* rmsd
) {
    std::vector<float> S(K, 0.0f);
    std::vector<int> counts(K, 0);

    // --- Compute S_i (intra-cluster scatter)
    for (int i = 0; i < N_frames; i++) {
        int k = clusters[i];
        S[k] += rmsd[centroids[k] * N_frames + i];
        counts[k]++;
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0)
            S[k] /= counts[k];
    }

    // --- Compute DB
    float db = 0.0f;

    for (int i = 0; i < K; i++) {
        float maxR = 0.0f;

        for (int j = 0; j < K; j++) {
            if (i == j) continue;

            float Mij = rmsd[centroids[i] * N_frames + centroids[j]];
            if (Mij > 0.0f) {
                float Rij = (S[i] + S[j]) / Mij;
                maxR = std::max(maxR, Rij);
            }
        }

        db += maxR;
    }

    return db / K;
}

int main() {

    // Time measurements
    chrono_type global_start = chrono_time::now();


    int K = 10;
    int MAX_ITER = 50;

    FileUtils file; 

    // std::cout << file << std::endl;

    size_t N_frames = file.getN_frames();
    // size_t N_frames = 10000;
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

    // Load RMSD tab
    float* rmsd;
    size_t size_rmsd = N_frames * N_frames * sizeof(float);
    CHECK_SUCCESS(cudaMalloc(&rmsd, size_rmsd), "Allocating rmsd vector on GPU");

    cudaDeviceSynchronize();
    measure_seconds(mem_transfer_start, "CPU to GPU memory transfer");

    dim3 threads(256,1);
    dim3 blocks((N_frames + threads.x - 1) / threads.x, (N_frames + threads.y - 1) / threads.y);

    chrono_type rmsd_kernel_start = chrono_time::now();
    RMSD<<<blocks, threads>>>(
        frameGPU,
        N_frames,
        N_atoms,
        rmsd
    );
    cudaDeviceSynchronize();
    measure_seconds(rmsd_kernel_start, "RMSD Kernel");

    float* rmsdHost = new float[N_frames*N_frames];
    CHECK_SUCCESS(cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Memcpy rmsd -> rmsdHost");

    chrono_type clustering_loop_start = chrono_time::now();
    // Pick first K unique indices
    int* centroids = new int[K];
    pickRandomCentroids(N_frames, K, centroids);
    int* clusters = new int[N_frames];

    // LOOP STARTS HERE
    for (int i=0; i<MAX_ITER; i++) {
        // std::cout << "Iteration " << i+1 << std::endl;
        // for (int k=0; k<K; k++) {
        //     for(int i=0; i<3; i++) {
        //         std::cout << "RMSD between " << centroids[k] << " and " << i << " is " << rmsdHost[centroids[k]*N_frames + i] << std::endl;
        //     }
        // }

        // Affecting molecules to the different centroids based on rmsd
        createClusters(N_frames, K, rmsdHost, centroids, clusters);

        // Define new centroids
        updateCentroids(N_frames, K, clusters, rmsdHost, centroids);
    }
    measure_seconds(clustering_loop_start, "Clustering loop");
    measure_seconds(global_start, "Entire program");

    std::cout << "Final centroids: " << std::endl;
    for (int i=0; i<K; i++) {
        std::cout << centroids[i] << std::endl;
    }

    float db = daviesBouldinIndex(
        N_frames,
        K,
        clusters,
        centroids,
        rmsdHost
    );

    std::cout << "Davies–Bouldin index: " << db << std::endl;

    // Print db for random clustering
    pickRandomCentroids(N_frames, K, centroids);
    for (int i = 0; i < N_frames; i++) {
        clusters[i] = rand() % K;
    }
    createClusters(N_frames, K, rmsdHost, centroids, clusters);
    db = daviesBouldinIndex(
        N_frames,
        K,
        clusters,
        centroids,
        rmsdHost
    );

    std::cout << "Random Davies–Bouldin index: " << db << std::endl;

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
