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

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void pickRandomCentroids(int N_frames, int K, int* centroids) {
    int* indices = new int[N_frames];
    for (int i = 0; i < N_frames; i++) indices[i] = i;

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

    for (int i = 0; i < N_frames; i++) {
        int k = clusters[i];
        S[k] += rmsd[centroids[k] * N_frames + i];
        counts[k]++;
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0)
            S[k] /= counts[k];
    }

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

// Run K-medoids clustering and return DB index
float runKMedoids(int N_frames, int K, const float* rmsdHost, 
                  int MAX_ITER, int* final_centroids = nullptr, 
                  int* final_clusters = nullptr) {
    int* centroids = new int[K];
    int* clusters = new int[N_frames];
    
    pickRandomCentroids(N_frames, K, centroids);
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        createClusters(N_frames, K, rmsdHost, centroids, clusters);
        
        int* old_centroids = new int[K];
        memcpy(old_centroids, centroids, K * sizeof(int));
        
        updateCentroids(N_frames, K, clusters, rmsdHost, centroids);
        
        bool converged = true;
        for (int k = 0; k < K; k++) {
            if (centroids[k] != old_centroids[k]) {
                converged = false;
                break;
            }
        }
        
        delete[] old_centroids;
        
        if (converged) break;
    }
    
    float db = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdHost);
    
    // Copy results if requested
    if (final_centroids != nullptr) {
        memcpy(final_centroids, centroids, K * sizeof(int));
    }
    if (final_clusters != nullptr) {
        memcpy(final_clusters, clusters, N_frames * sizeof(int));
    }
    
    delete[] centroids;
    delete[] clusters;
    
    return db;
}

// Run random clustering and return DB index
float runRandomClustering(int N_frames, int K, const float* rmsdHost) {
    int* centroids = new int[K];
    int* clusters = new int[N_frames];
    
    pickRandomCentroids(N_frames, K, centroids);
    for (int i = 0; i < N_frames; i++) {
        clusters[i] = rand() % K;
    }
    createClusters(N_frames, K, rmsdHost, centroids, clusters);
    
    float db = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdHost);
    
    delete[] centroids;
    delete[] clusters;
    
    return db;
}

int main() {

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    int MAX_ITER = 50;
    int K_MIN = 2;
    int K_MAX = 50;

    FileUtils file; 

    std::cout << file << std::endl;

    // size_t N_frames = 20000;
    size_t N_frames = file.getN_frames();
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "Processing " << N_frames << " frames with " 
              << N_atoms << " atoms each" << std::endl;

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);    

    // Copy reordered CPU → GPU
    float* frameGPU;
    CUDA_CHECK(cudaMalloc(&frameGPU, total_size));
    CUDA_CHECK(cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice));

    std::cout << "Copied " << (total_size / (1024.0*1024.0)) 
              << " MB to GPU" << std::endl;

    // Allocate RMSD matrix
    float* rmsd;
    size_t size_rmsd = N_frames * N_frames * sizeof(float);
    CUDA_CHECK(cudaMalloc(&rmsd, size_rmsd));
    
    std::cout << "Allocated " << (size_rmsd / (1024.0*1024.0)) 
              << " MB for RMSD matrix" << std::endl;

    dim3 threads(16, 16);
    dim3 blocks((N_frames + threads.x - 1) / threads.x, 
                (N_frames + threads.y - 1) / threads.y);

    std::cout << "\nKernel Start" << std::endl;
    RMSD<<<blocks, threads>>>(frameGPU, N_frames, N_atoms, rmsd);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel Finished" << std::endl;

    // Copy RMSD matrix back to host
    float* rmsdHost = new float[N_frames*N_frames];
    CUDA_CHECK(cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost));

    // Verify symmetry
    std::cout << "\nVerifying RMSD matrix symmetry..." << std::endl;
    bool symmetric = true;
    for (int i = 0; i < std::min(1000, (int)N_frames); i++) {
        for (int j = i+1; j < std::min(1000, (int)N_frames); j++) {
            if (fabsf(rmsdHost[i * N_frames + j] - rmsdHost[j * N_frames + i]) > 1e-5f) {
                symmetric = false;
                break;
            }
        }
        if (!symmetric) break;
    }
    std::cout << (symmetric ? "✓" : "✗") << " RMSD matrix is " 
              << (symmetric ? "" : "NOT ") << "symmetric" << std::endl;

    // ==============================================================
    // MAIN ANALYSIS: Scan K from K_MIN to K_MAX
    // ==============================================================
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "DAVIES-BOULDIN INDEX ANALYSIS (K = " << K_MIN 
              << " to " << K_MAX << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Storage for results
    std::vector<int> K_values;
    std::vector<float> db_kmedoids;
    std::vector<float> db_random;
    
    std::cout << "\nK\tDB_KMedoids\tDB_Random\tDifference\tConvergence" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (int K = K_MIN; K <= K_MAX; K++) {
        std::cout << K << "\t" << std::flush;
        
        // Run K-medoids clustering
        float db_km = runKMedoids(N_frames, K, rmsdHost, MAX_ITER);
        std::cout << db_km << "\t" << std::flush;
        
        // Run random clustering baseline
        float db_rand = runRandomClustering(N_frames, K, rmsdHost);
        std::cout << db_rand << "\t" << std::flush;
        
        float diff = db_rand - db_km;
        std::cout << diff << "\t" << std::flush;
        std::cout << (diff > 0 ? "✓ Better" : "✗ Worse") << std::endl;
        
        // Store results
        K_values.push_back(K);
        db_kmedoids.push_back(db_km);
        db_random.push_back(db_rand);
    }
    
    std::cout << std::string(70, '=') << std::endl;
    
    // ==============================================================
    // Save results to CSV for plotting
    // ==============================================================
    
    std::ofstream csv_out("output/db_index_vs_K.csv");
    csv_out << "K,DB_KMedoids,DB_Random,Difference\n";
    for (size_t i = 0; i < K_values.size(); i++) {
        csv_out << K_values[i] << "," 
                << db_kmedoids[i] << "," 
                << db_random[i] << ","
                << (db_random[i] - db_kmedoids[i]) << "\n";
    }
    csv_out.close();
    
    std::cout << "\n✓ Results saved to output/db_index_vs_K.csv" << std::endl;
    
    // ==============================================================
    // Find optimal K
    // ==============================================================
    
    auto min_it = std::min_element(db_kmedoids.begin(), db_kmedoids.end());
    int optimal_K_db = K_values[std::distance(db_kmedoids.begin(), min_it)];
    float optimal_db = *min_it;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "OPTIMAL K ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Optimal K (lowest DB index): " << optimal_K_db << std::endl;
    std::cout << "DB Index at optimal K: " << optimal_db << std::endl;
    
    // Check if K-medoids consistently beats random
    int better_count = 0;
    for (size_t i = 0; i < K_values.size(); i++) {
        if (db_kmedoids[i] < db_random[i]) better_count++;
    }
    
    float percent_better = 100.0f * better_count / K_values.size();
    std::cout << "\nK-medoids beats random: " << better_count << "/" 
              << K_values.size() << " times (" << percent_better << "%)" << std::endl;
    
    if (percent_better < 50) {
        std::cout << "\n⚠️  WARNING: K-medoids rarely beats random clustering!" << std::endl;
        std::cout << "This suggests your data may not have well-separated clusters." << std::endl;
        std::cout << "Consider:" << std::endl;
        std::cout << "  - Using fewer atoms (backbone only)" << std::endl;
        std::cout << "  - Different distance metric" << std::endl;
        std::cout << "  - PCA or other dimensionality reduction first" << std::endl;
    }
    
    // ==============================================================
    // Run detailed analysis at optimal K
    // ==============================================================
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "DETAILED ANALYSIS AT OPTIMAL K = " << optimal_K_db << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    int* final_centroids = new int[optimal_K_db];
    int* final_clusters = new int[N_frames];
    
    float final_db = runKMedoids(N_frames, optimal_K_db, rmsdHost, MAX_ITER, 
                                  final_centroids, final_clusters);
    
    std::cout << "\nFinal centroids (frame indices):" << std::endl;
    for (int k = 0; k < optimal_K_db; k++) {
        std::cout << "  Cluster " << k << ": frame " << final_centroids[k] << std::endl;
    }
    
    // Compute cluster sizes
    std::vector<int> cluster_sizes(optimal_K_db, 0);
    for (int i = 0; i < N_frames; i++) {
        cluster_sizes[final_clusters[i]]++;
    }
    
    std::cout << "\nCluster sizes:" << std::endl;
    for (int k = 0; k < optimal_K_db; k++) {
        float percent = 100.0f * cluster_sizes[k] / N_frames;
        std::cout << "  Cluster " << k << ": " << cluster_sizes[k] 
                  << " frames (" << percent << "%)" << std::endl;
    }
    
    // Save centroids to file
    std::ofstream cent_out("output/optimal_centroids.txt");
    cent_out << "# Optimal K = " << optimal_K_db << "\n";
    cent_out << "# Davies-Bouldin Index = " << final_db << "\n";
    cent_out << "# Cluster\tFrame_Index\tSize\n";
    for (int k = 0; k < optimal_K_db; k++) {
        cent_out << k << "\t" << final_centroids[k] << "\t" 
                 << cluster_sizes[k] << "\n";
    }
    cent_out.close();
    
    std::cout << "\n✓ Optimal clustering results saved to output/optimal_centroids.txt" 
              << std::endl;

    // Cleanup
    delete[] frame;
    delete[] rmsdHost;
    delete[] final_centroids;
    delete[] final_clusters;
    CUDA_CHECK(cudaFree(frameGPU));
    CUDA_CHECK(cudaFree(rmsd));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PERFORMANCE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Total execution time: " << elapsed_ms/1000.0f << " s" << std::endl;
    std::cout << "RMSD kernel time: ~15 s (from previous run)" << std::endl;
    std::cout << "Clustering analysis: ~" << (elapsed_ms/1000.0f - 15) << " s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
