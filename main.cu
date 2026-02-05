/*
    Compile with:
    
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 --use_fast_math -Xcompiler -fopenmp \
    main.cu FileUtils.cpp gpu.cu utils.cu \
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
#include <iomanip>
#include "utils.cuh"
#include <omp.h>


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

int main() {
   
    cudaEvent_t evStart, evStop, evTotalStart, evTotalStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventCreate(&evTotalStart));
    CUDA_CHECK(cudaEventCreate(&evTotalStop));

    // TODO: parsing arguments
    const int MAX_ITER = 50;
    const int K_MIN = 2;
    const int K_MAX = 50;
    const int NB_TRIAL = 5;

    FileUtils file; 

    std::cout << file << std::endl;

    size_t N_snapshots = 50000;
    // size_t N_snapshots = file.getN_snapshots();
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "Processing " << N_snapshots << " snapshots with " 
              << N_atoms << " atoms each" << std::endl;

    // Load and reorder into X,Y,Z blocks
    // TODO: This part is slow and might be done as preprocessing
    float* reordered_file_host = file.loadData(N_snapshots);
    file.reorderByLine(reordered_file_host, N_snapshots);

    size_t total_size = N_snapshots * N_atoms * N_dims * sizeof(float);    

    // Copy reordered CPU → GPU
    float* reordered_file_device;
    CUDA_CHECK(cudaMalloc(&reordered_file_device, total_size));
    CUDA_CHECK(cudaEventRecord(evTotalStart));

    float t_transfer_host_to_device = 0.f;
    CUDA_CHECK(cudaEventRecord(evStart));
    CUDA_CHECK(cudaMemcpy(reordered_file_device, reordered_file_host, total_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&t_transfer_host_to_device, evStart, evStop));


    std::cout << "Copied " << (total_size / (1024.0*1024.0)) 
              << " MB to GPU" << std::endl;

    // Allocate RMSD matrix
    size_t rmsd_elems = ((size_t)N_snapshots * ((size_t)N_snapshots - 1)) / 2;
    size_t size_rmsd  = rmsd_elems * sizeof(float);
    float* rmsd_device;
    CUDA_CHECK(cudaMalloc(&rmsd_device, size_rmsd));
    CUDA_CHECK(cudaMemset(rmsd_device, 0, size_rmsd));
    
    std::cout << "Allocated " << (size_rmsd / (1024.0*1024.0)) 
              << " MB for RMSD matrix" << std::endl;

    dim3 threads(16, 16);
    dim3 blocks((N_snapshots + threads.x - 1) / threads.x, 
                (N_snapshots + threads.y - 1) / threads.y);

    std::cout << "\nKernel Start" << std::endl;
    float t_kernel = 0.f;
    CUDA_CHECK(cudaEventRecord(evStart));
    RMSD<<<blocks, threads>>>(reordered_file_device, N_snapshots, N_atoms, rmsd_device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(reordered_file_device));

    CUDA_CHECK(cudaEventRecord(evStop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel, evStart, evStop));

    std::cout << "Kernel Finished" << std::endl;

    // Copy RMSD matrix back to host
    float t_transfer_device_to_host = 0.f;

    float* rmsd_host = new float[rmsd_elems];
    CUDA_CHECK(cudaEventRecord(evStart));
    CUDA_CHECK(cudaMemcpy(rmsd_host, rmsd_device, size_rmsd, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&t_transfer_device_to_host, evStart, evStop));

    // ==============================================================
    // MAIN ANALYSIS: Scan K from K_MIN to K_MAX
    // ==============================================================

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "DAVIES-BOULDIN INDEX ANALYSIS (K = " << K_MIN 
            << " to " << K_MAX << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Storage for results - DECLARE ONLY ONCE
    std::vector<int> K_values;
    std::vector<float> db_kmedoids;
    std::vector<float> db_random;

    std::cout << "\n"
            << std::setw(4)  << "K"
            << std::setw(16) << "DB_KMedoids"
            << std::setw(16) << "DB_Random"
            << std::setw(16) << "Difference"
            << std::setw(14) << "Result"
            << '\n';
    std::cout << std::string(4+16+16+16+14, '-') << '\n';

    std::cout << std::fixed << std::setprecision(6);

    // Pre-allocate for thread safety
    int num_K = K_MAX - K_MIN + 1;
    K_values.resize(num_K);
    db_kmedoids.resize(num_K);
    db_random.resize(num_K);

    double clustering_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int k_idx = 0; k_idx < num_K; ++k_idx) {
        int K = K_MIN + k_idx;
        
        float best_db_km = 1e30f;
        float best_rd_db_km = 1e30f;

        int* best_centroids = new int[K];
        int* best_clusters  = new int[N_snapshots];

        for (int trial = 0; trial < NB_TRIAL; trial++) {

            int* init_centroids = new int[K];
            int* km_centroids   = new int[K];
            int* km_clusters    = new int[N_snapshots];

            pickKMedoidsPlusPlus(N_snapshots, K, rmsd_host, init_centroids);

            float db_km_trial = runKMedoidsInit(
                N_snapshots, K, rmsd_host, MAX_ITER,
                init_centroids,
                km_centroids,
                km_clusters
            );

            if (db_km_trial < best_db_km) {
                best_db_km = db_km_trial;
                memcpy(best_centroids, km_centroids, K * sizeof(int));
                memcpy(best_clusters, km_clusters, N_snapshots * sizeof(int));
            }

            float db_rand = runRandomClusterAssignment(N_snapshots, K, rmsd_host);

            if (db_rand < best_rd_db_km) {
                best_rd_db_km = db_rand;
            }

            delete[] init_centroids;
            delete[] km_centroids;
            delete[] km_clusters;
        }

        float diff = best_rd_db_km - best_db_km;

        // Store results in pre-allocated vectors (thread-safe)
        K_values[k_idx] = K;
        db_kmedoids[k_idx] = best_db_km;
        db_random[k_idx] = best_rd_db_km;

        // Print with critical section to avoid interleaved output
        #pragma omp critical
        {
            std::cout
                << std::setw(4)  << K
                << std::setw(16) << best_db_km
                << std::setw(16) << best_rd_db_km
                << std::setw(16) << diff
                << std::setw(14) << (diff > 0 ? "Better" : "Worse")
                << '\n';
        }

        delete[] best_centroids;
        delete[] best_clusters;
    }

    double clustering_end = omp_get_wtime();
    double clustering_time = clustering_end - clustering_start;

    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Clustering analysis time: " << clustering_time << " s" << std::endl;
    
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
    int* final_clusters = new int[N_snapshots];
    
    pickKMedoidsPlusPlus(N_snapshots, optimal_K_db, rmsd_host, final_centroids);

    float final_db = runKMedoidsInit(
        N_snapshots, optimal_K_db, rmsd_host, MAX_ITER,
        final_centroids,
        final_centroids,
        final_clusters
    );

    std::cout << "\nFinal centroids (snapshot indices):" << std::endl;
    for (int k = 0; k < optimal_K_db; k++) {
        std::cout << "  Cluster " << k << ": snapshot " << final_centroids[k] << std::endl;
    }
    
    // Compute cluster sizes
    std::vector<int> cluster_sizes(optimal_K_db, 0);
    for (int i = 0; i < N_snapshots; i++) {
        cluster_sizes[final_clusters[i]]++;
    }
    
    std::cout << "\nCluster sizes:" << std::endl;
    for (int k = 0; k < optimal_K_db; k++) {
        float percent = 100.0f * cluster_sizes[k] / N_snapshots;
        std::cout << "  Cluster " << k << ": " << cluster_sizes[k] 
                  << " snapshots (" << percent << "%)" << std::endl;
    }
    
    // Save centroids to file
    std::ofstream cent_out("output/optimal_centroids.txt");
    cent_out << "# Optimal K = " << optimal_K_db << "\n";
    cent_out << "# Davies-Bouldin Index = " << final_db << "\n";
    cent_out << "# Cluster\tSnapshot_Index\tSize\n";
    for (int k = 0; k < optimal_K_db; k++) {
        cent_out << k << "\t" << final_centroids[k] << "\t" 
                 << cluster_sizes[k] << "\n";
    }
    cent_out.close();
    
    std::cout << "\n✓ Optimal clustering results saved to output/optimal_centroids.txt" 
              << std::endl;

    // Cleanup
    delete[] reordered_file_host;
    delete[] rmsd_host;
    delete[] final_centroids;
    delete[] final_clusters;
    CUDA_CHECK(cudaFree(rmsd_device));

    CUDA_CHECK(cudaEventRecord(evTotalStop));
    CUDA_CHECK(cudaEventSynchronize(evTotalStop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, evTotalStart, evTotalStop));

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PERFORMANCE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "Host to device copy time   : " << t_transfer_host_to_device   / 1000.0f << " s" << std::endl;
    std::cout << "Kernel time     : " << t_kernel / 1000.0f << " s" << std::endl;
    std::cout << "Device to Host copy time   : " << t_transfer_device_to_host   / 1000.0f << " s" << std::endl;
    std::cout << "Clustering time : " << clustering_time << " s" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Total execution: " << total_ms / 1000.0f << " s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaEventDestroy(evTotalStart));
    CUDA_CHECK(cudaEventDestroy(evTotalStop));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
