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
#include <chrono>
#include <iomanip>
#include "utils.cuh"

int main(int argc, char** args) {

    // Time measurements
    chrono_type global_start = chrono_time::now();

    int K = 10;
    int MAX_ITER = 50;

    std::string file_name;
    if (argc >= 2) {
        file_name = args[1];
    } else {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>" << std::endl;
        std::cerr << "Example: " << args[0] << " data/trajectory.bin" << std::endl;
        return 1;
    }
    FileUtils file(file_name); 

    size_t N_frames = 10000;
    size_t N_atoms = file.getN_atoms();
    size_t N_dims = file.getN_dims();

    size_t MAX_DATA_CHUNK_SIZE = 650; // In MB

    int NB_FRAMES_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims, N_frames);
    int SQ_SUBMATRIX_SIZE = NB_FRAMES_CHUNK / 2;
    int NB_ROW_ITERATIONS = (int) std::floor( ( N_frames - 1 ) / SQ_SUBMATRIX_SIZE ) + 1;
    int RMSD_LOOPS_NEEDED = (int) NB_ROW_ITERATIONS * (NB_ROW_ITERATIONS + 1) / 2;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "RMSD COMPUTATION CONFIGURATION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Max chunk size           : " << MAX_DATA_CHUNK_SIZE << " MB\n";
    std::cout << "Max frames per chunk     : " << NB_FRAMES_CHUNK << "\n";
    std::cout << "Submatrix tile size      : " << SQ_SUBMATRIX_SIZE << "\n";
    std::cout << "Number of RMSD iterations: " << RMSD_LOOPS_NEEDED << "\n";
    std::cout << "Total frames to process  : " << N_frames << "\n";
    std::cout << "Number of atoms          : " << N_atoms << "\n";
    std::cout << std::string(70, '=') << std::endl << std::endl;

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);

    // Allocate PACKED upper triangle matrix
    size_t rmsd_size = (size_t)N_frames * (N_frames - 1) / 2;
    float* rmsdHost = new float[rmsd_size];

    int row_begin = 0;

    for(int iter=0; iter < RMSD_LOOPS_NEEDED; ++iter) {
        int col_begin = col_index_parcours(iter, NB_ROW_ITERATIONS - 1) * SQ_SUBMATRIX_SIZE;
        int col_end = std::min(col_begin + SQ_SUBMATRIX_SIZE, (int)N_frames);
        int row_end = std::min(row_begin + SQ_SUBMATRIX_SIZE, (int)N_frames);

        int size_row = row_end - row_begin;
        int size_col = col_end - col_begin;

        int nb_frames_subset;

        if(col_begin == row_begin) {
            nb_frames_subset = size_col;
        }
        else {
            nb_frames_subset = size_col + size_row;
        }

        std::cout << "Iteration " << iter << "/" << RMSD_LOOPS_NEEDED-1 
                  << " | Tile: [" << row_begin << ":" << row_end << ", " 
                  << col_begin << ":" << col_end << "] | Frames: " << nb_frames_subset << std::endl;

        float* frame_subset = file.getFrameSubset(frame, row_begin, row_end, col_begin, col_end, N_frames);
        file.reorderByLine(frame_subset, nb_frames_subset);

        size_t total_size = nb_frames_subset * N_atoms * N_dims * sizeof(float);

        // Copy reordered CPU → GPU
        chrono_type mem_transfer_start = chrono_time::now();
        float* frameGPU;
        CHECK_SUCCESS(cudaMalloc(&frameGPU, total_size), "Allocating frameGPU");
        CHECK_SUCCESS(cudaMemcpy(frameGPU, frame_subset, total_size, cudaMemcpyHostToDevice), "Memcpy frame -> frameGPU");

        // Allocate RMSD matrix
        float* rmsd;
        size_t size_rmsd = (size_t)nb_frames_subset * (nb_frames_subset - 1) / 2 * sizeof(float);

        CHECK_SUCCESS(cudaMalloc(&rmsd, size_rmsd), "Allocating rmsd vector on GPU");

        cudaDeviceSynchronize();

        dim3 threads(16,16);
        dim3 blocks((nb_frames_subset + threads.x - 1) / threads.x, 
                    (nb_frames_subset + threads.y - 1) / threads.y);

        chrono_type rmsd_kernel_start = chrono_time::now();
        RMSD<<<blocks, threads>>>(
            frameGPU,
            nb_frames_subset,
            N_atoms,
            rmsd
        );
        CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD Kernel");

        size_t rmsdCount = (size_t)nb_frames_subset * (nb_frames_subset - 1) / 2;
        float* rmsdSubsetHost = new float[rmsdCount];
        CHECK_SUCCESS(cudaMemcpy(rmsdSubsetHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Memcpy rmsd -> rmsdSubsetHost");

        // Reconstruct into packed upper triangle format
        for(int i = row_begin; i < row_end; ++i) {
            for(int j = col_begin; j < col_end; ++j) {
                
                // Only fill upper triangle (j > i)
                if (j <= i) continue;

                int ii, jj;
                
                if (col_begin == row_begin) {
                    // Diagonal tile: both indices from same range
                    ii = i - row_begin;
                    jj = j - col_begin;
                } else {
                    // Off-diagonal tile: i from row range, j from col range
                    ii = i - row_begin;              // Index in first part (0 to size_row-1)
                    jj = size_row + (j - col_begin); // Index in second part
                }

                // Ensure ii < jj for upper triangle lookup in subset
                int a = std::min(ii, jj);
                int b = std::max(ii, jj);

                size_t subset_idx = (size_t)a * nb_frames_subset
                                  - ((size_t)a * ((size_t)a + 1)) / 2
                                  + (b - a - 1);

                if (subset_idx >= rmsdCount) {
                    std::cerr << "ERROR: Index overflow " << subset_idx << " >= " << rmsdCount << std::endl;
                    continue;
                }

                float v = rmsdSubsetHost[subset_idx];

                // Store in packed upper triangle of full matrix
                // Global indices: i < j (already checked above)
                size_t global_idx = (size_t)i * N_frames
                                  - ((size_t)i * ((size_t)i + 1)) / 2
                                  + (j - i - 1);

                rmsdHost[global_idx] = v;
            }
        }

        if(col_end == (int)N_frames) {
            row_begin += SQ_SUBMATRIX_SIZE;
        }

        delete[] rmsdSubsetHost;
        delete[] frame_subset;
        cudaFree(frameGPU);
        cudaFree(rmsd);
    }

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

    float db_index = runKMedoids(N_frames, K, rmsdHost, MAX_ITER, centroids, clusters);
    
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
    
    float random_db_index = runRandomClustering(N_frames, K, rmsdHost);
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
    delete[] frame;
    delete[] centroids;
    delete[] rmsdHost;
    delete[] clusters;

    return 0;
}