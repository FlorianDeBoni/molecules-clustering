/*
    Compile with:
    
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 --use_fast_math -Xcompiler -fopenmp \
    main.cu FileUtils.cpp gpu.cu utils.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o main

    ./main output/snapshots_coords_all.bin
*/

#include "FileUtils.hpp"
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
#include "utils.cuh"

int main(int argc, char** args) {

    // Time measurements
    chrono_type global_start = chrono_time::now();

    std::string file_name;
    if (argc >= 2) {
        file_name = args[1];
    } else {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>" << std::endl;
        return 1;
    }

    FileUtils file(file_name); 

    size_t N_frames = 20001;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    // Load entire dataset into memory once
    std::vector<float> all_data(N_frames * N_atoms * 3);
    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);
    std::cout << "Loaded " << N_frames * N_atoms * N_dims * sizeof(float) / (1024*1024) << " MiB into CPU RAM." << std::endl;

    const size_t MAX_DATA_CHUNK_SIZE = 3000; // In MB
    const size_t NB_FRAMES_PER_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS = (size_t) std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);
    const size_t RMSD_LOOPS_NEEDED = NB_ROW_ITERATIONS * (NB_ROW_ITERATIONS + 1) / 2;

    std::cout << "Max frames per chunk: " << NB_FRAMES_PER_CHUNK << "\n";
    std::cout << "Number of RMSD iterations: " << RMSD_LOOPS_NEEDED << std::endl;

    // Allocate PACKED upper triangle matrix
    size_t rmsd_all_size = N_frames * N_frames;
    float* rmsdHostAll = new float[rmsd_all_size];

    size_t rmsd_chunk_size = NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK;
    float* rmsdHostChunk = new float[rmsd_chunk_size];

    size_t iter = 0;

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // ALLOCATING TO GPU
    float* d_references = nullptr;
    float* d_targets = nullptr;
    float* d_rmsd = nullptr;

    // You can allocate references/targets once as maximum possible chunk
    CHECK_SUCCESS(cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)), "Allocating memory for references");
    CHECK_SUCCESS(cudaMalloc(&d_targets, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)), "Allocating memory for targets");
    CHECK_SUCCESS(cudaMalloc(&d_rmsd, NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)), "Allocating rmsd vector on GPU");

    dim3 threads(16,16);
    size_t size_rmsd = NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float);

    std::cout << "\n" << std::string(70, '=') << std::endl; std::cout << "RMSD COMPUTATION START" << std::endl; std::cout << std::string(70, '=') << std::endl;
    for (size_t row = 0; row < NB_ROW_ITERATIONS; ++row) {
        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        CHECK_SUCCESS(cudaMemcpy(d_references, references_coordinates.data(), references_coordinates.size() * sizeof(float), cudaMemcpyHostToDevice), "Copying References on GPU");

        const size_t nb_frames_subset_references = stop_row-start_row;

        for (size_t col = row; col < NB_ROW_ITERATIONS; ++col) {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            
            std::cout << "Iteration " << ++iter << "/" << RMSD_LOOPS_NEEDED
                      << ", Row [" << start_row << "," << stop_row << ")"
                      << ", Col [" << start_col << "," << stop_col << ")\n";

            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
            CHECK_SUCCESS(cudaMemcpy(d_targets, targets_coordinates.data(), targets_coordinates.size() * sizeof(float), cudaMemcpyHostToDevice), "Copying Targets on GPU");

            const size_t nb_frames_subset_targets = stop_col-start_col;

            dim3 blocks((nb_frames_subset_references + threads.x - 1) / threads.x, 
                        (nb_frames_subset_targets + threads.y - 1) / threads.y);

            CHECK_SUCCESS(cudaDeviceSynchronize(), "Ready to launch RMSD Kernel");

            RMSD<<<blocks, threads>>>(
                d_references,
                d_targets,
                nb_frames_subset_references,
                nb_frames_subset_targets,
                N_atoms,
                d_rmsd
            );
            CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD Kernel");
            CHECK_SUCCESS(cudaMemcpy(rmsdHostChunk, d_rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Copying RMSD chunk to CPU");
            
            // Copy chunk to all 
            for(size_t i = 0; i < nb_frames_subset_references; ++i) {
                for(size_t j = 0; j < nb_frames_subset_targets; ++j) {
                    size_t global_row = start_row + i;
                    size_t global_col = start_col + j;
                    size_t chunk_idx = i * nb_frames_subset_targets + j;  // Match kernel layout
                    size_t global_idx = global_row * N_frames + global_col;
                    
                    rmsdHostAll[global_idx] = rmsdHostChunk[chunk_idx];
                }
            }
        }
    }

    CHECK_SUCCESS(cudaFree(d_references), "Freeing References on GPU");
    CHECK_SUCCESS(cudaFree(d_rmsd), "Freeing RMSD vector on GPU");
    CHECK_SUCCESS(cudaFree(d_targets), "Freeing Targets on GPU");

    // Calculate size for upper triangle without diagonal
    size_t upper_triangle_size = (N_frames * (N_frames - 1)) / 2;
    float* rmsdUpperTriangle = new float[upper_triangle_size];

    // Copy upper triangle without diagonal
    size_t idx = 0;
    for (size_t i = 0; i < N_frames; ++i) {
        for (size_t j = i + 1; j < N_frames; ++j) {
            size_t global_idx = i * N_frames + j;
            rmsdUpperTriangle[idx++] = rmsdHostAll[global_idx];
        }
    }

    delete[] rmsdHostAll;

    int K = 10;
    int MAX_ITER = 50;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "RMSD COMPUTATION COMPLETE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    measure_seconds(global_start, "Total RMSD computation time");
    std::cout << std::endl;

    // =========================================================================
    // CLUSTERING
    // =========================================================================
    
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "K-MEDOIDS CLUSTERING (K=" << K << ", MAX_ITER=" << MAX_ITER << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    chrono_type clustering_loop_start = chrono_time::now();
    
    int* centroids = new int[K];
    int* clusters = new int[N_frames];

    float db_index = runKMedoids(N_frames, K, rmsdUpperTriangle, MAX_ITER, centroids, clusters);
    
    measure_seconds(clustering_loop_start, "K-medoids clustering time");
    std::cout << std::endl;

    // =========================================================================
    // RESULTS
    // =========================================================================
    
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "CLUSTERING RESULTS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "K-medoids Davies-Bouldin Index: " << db_index << std::endl;
    
    // Compute and display cluster sizes
    std::vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < N_frames; i++) {
        cluster_sizes[clusters[i]]++;
    }
    
    std::cout << "\nCluster centroids and sizes:" << std::endl;
    for (int k = 0; k < K; k++) {
        float percent = 100.0f * cluster_sizes[k] / N_frames;
        std::cout << "  Cluster " << std::setw(2) << k 
                  << " | Centroid: frame " << std::setw(6) << centroids[k] 
                  << " | Size: " << std::setw(6) << cluster_sizes[k] 
                  << " (" << std::setw(5) << std::setprecision(2) << percent << "%)" << std::endl;
    }
    
    // Random clustering baseline
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "BASELINE COMPARISON" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    float random_db_index = runRandomClustering(N_frames, K, rmsdUpperTriangle);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Random clustering Davies-Bouldin Index: " << random_db_index << std::endl;
    
    float improvement = ((random_db_index - db_index) / random_db_index) * 100.0f;
    std::cout << "\nK-medoids improvement over random: " 
              << std::setprecision(2) << improvement << "%" 
              << (improvement > 0 ? " ✓ BETTER" : " ✗ WORSE") << std::endl;
    
    std::cout << std::string(70, '=') << std::endl << std::endl;

    saveClusters(clusters, N_frames, centroids, K);

    measure_seconds(global_start, "Total program execution time");

    // Cleanup
    delete[] centroids;
    delete[] rmsdUpperTriangle;
    delete[] clusters;

    return 0;
}
