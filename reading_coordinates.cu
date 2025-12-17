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
#include <algorithm>
#include <random>
#include "gpu.cuh"

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

void createClusters(int N_frames, int K, float* rmsdHost, std::vector<int> cluster)
{
    for (size_t i = 0; i < N_frames; i++) {
        float min_rmsd = 1e30f;
        int min_k = -1;
        for (int k = 0; k < K; k++) {
            float d = rmsdHost[k * N_frames + i];
            if (d < min_rmsd) {
                min_rmsd = d;
                min_k = k;
            }
        }
        cluster[i] = min_k;
    }
}

void updateCentroids(int N_frames, int K, const std::vector<int>& cluster, float* rmsdHost, int* centroids)
{
    for (int k = 0; k < K; k++) {
        float best_sum = 1e30f;
        int best_idx = -1;

        // Consider all frames in this cluster
        for (size_t i = 0; i < N_frames; i++) {
            if (cluster[i] != k) continue;

            // Sum RMSD of frame i to all other frames in the cluster
            float sum_rmsd = 0.0f;
            for (size_t j = 0; j < N_frames; j++) {
                if ((cluster[j] != k) || (j == centroids[k])) continue;

                // RMSD of frame i to frame j
                // Approximate using RMSD of j to centroid i
                sum_rmsd += rmsdHost[i + k * N_frames]; 
            }

            if (sum_rmsd < best_sum) {
                best_sum = sum_rmsd;
                best_idx = i;
            }
        }

        // Update centroid for cluster k
        if (best_idx != -1) {
            centroids[k] = best_idx;
        }
    }
}

int main() {

    int K = 5;
    int MAX_ITER = 20;

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

    // Load RMSD tab
    float* rmsd;
    size_t size_rmsd = N_frames * sizeof(float) * K;
    cudaMalloc(&rmsd, size_rmsd);

    // Pick first K unique indices
    int* centroids = new int[K];
    pickRandomCentroids(N_frames, K, centroids);
    

    int threads = 256;
    int blocks = (N_frames + threads - 1) / threads;

    float* rmsdHost = new float[N_frames * K];

    // LOOP STARTS HERE
    for (int i=0; i<MAX_ITER; i++) {
        for (int k = 0; k<K; k++) {
            RMSD<<<blocks, threads>>>(
                frameGPU,   // reordered coordinates
                N_frames,
                N_atoms,
                centroids[k],
                k,
                rmsd
            );
        }
        cudaDeviceSynchronize();

        cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost);
        std::cout << "Iteration " << i+1 << std::endl;
        for (int k=0; k<K; k++) {
            for(int i=0; i<3; i++) {
                std::cout << "RMSD between " << centroids[k] << " and " << i << " is " << rmsdHost[k*N_frames + i] << std::endl;
            }
        }

        // Affecting molecules to the different centroids based on rmsd
        std::vector<int> cluster(N_frames, -1);
        createClusters(N_frames, K, rmsdHost, cluster);

        // Define new centroids
        updateCentroids(N_frames, K, cluster, rmsdHost, centroids);
    }

    // Cleanup
    delete[] frame;
    delete[] centroids;
    delete[] rmsdHost;
    cudaFree(frameGPU);
    cudaFree(rmsd);

    return 0;
}
