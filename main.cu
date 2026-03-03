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
    size_t N_atoms = file.getN_atoms();
    size_t N_dims = file.getN_dims();

    size_t MAX_DATA_CHUNK_SIZE = 1500; // In MB

    int NB_FRAMES_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims, N_frames);
    int SQ_SUBMATRIX_SIZE = NB_FRAMES_CHUNK / 2;
    // int SQ_SUBMATRIX_CARD = SQ_SUBMATRIX_SIZE * SQ_SUBMATRIX_SIZE;
    int NB_ROW_ITERATIONS = (int) std::floor( ( N_frames - 1 ) / SQ_SUBMATRIX_SIZE ) + 1;
    int RMSD_LOOPS_NEEDED = (int) NB_ROW_ITERATIONS * (NB_ROW_ITERATIONS + 1) / 2;
    
    std::cout << "\n";
    std::cout << "Taille maximale d'un chunk : " << MAX_DATA_CHUNK_SIZE << "MB\n";
    // std::cout << "Nombre de frames max dans un chunk : " << NB_FRAMES_CHUNK << "\n";
    std::cout << "Taille d'une sous-matrice de calcul : " << SQ_SUBMATRIX_SIZE << "\n";
    std::cout << "Nombre de tours : " << RMSD_LOOPS_NEEDED << "\n";

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);

    float* rmsdHost = new float[N_frames*N_frames];

    int row_begin = 0;

    // Pour monitor l'utilisation en mémoire du GPU toutes les secondes :
    // nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 1 

    for(int i=0; i < RMSD_LOOPS_NEEDED; ++i) {
        int col_begin = col_index_parcours(i,NB_ROW_ITERATIONS - 1) * SQ_SUBMATRIX_SIZE;
        int col_end = std::min(col_begin + SQ_SUBMATRIX_SIZE,(int) N_frames);
        int row_end = std::min(row_begin + SQ_SUBMATRIX_SIZE,(int) N_frames);

        int size_row = row_end - row_begin;
        int size_col = col_end - col_begin;

        int nb_frames_subset;

        if(col_begin == row_begin) {
            nb_frames_subset = size_col;
        }
        else {
            nb_frames_subset = size_col + size_row;
        }

        std::cout << "========================== " << "Iteration : " << i << " ==========================" << "\n";

        std::cout << "col_begin : " << col_begin << "\n";
        std::cout << "col_end : " << col_end << "\n";
        std::cout << "row_begin : " << row_begin << "\n";
        std::cout << "row_end : " << row_end << "\n";

        std::cout << "nb_frames_subset : " << nb_frames_subset << "\n";

        float* frame_subset = file.getFrameSubset(frame, row_begin, row_end, col_begin, col_end, N_frames);

        file.reorderByLine(frame_subset, nb_frames_subset);

        size_t total_size = nb_frames_subset * N_atoms * N_dims * sizeof(float);

        measure_seconds(global_start, "Loading source data");

        // Copy reordered CPU → GPU
        chrono_type mem_transfer_start = chrono_time::now();
        float* frameGPU;
        CHECK_SUCCESS(cudaMalloc(&frameGPU, total_size), "Allocating frameGPU");
        CHECK_SUCCESS(cudaMemcpy(frameGPU, frame_subset, total_size, cudaMemcpyHostToDevice), "Memcpy frame -> frameGPU");

        // Allocate RMSD matrix
        float* rmsd;
        size_t size_rmsd = nb_frames_subset * nb_frames_subset * sizeof(float);
        CHECK_SUCCESS(cudaMalloc(&rmsd, size_rmsd), "Allocating rmsd vector on GPU");

        cudaDeviceSynchronize();
        measure_seconds(mem_transfer_start, "CPU to GPU memory transfer");

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
        measure_seconds(rmsd_kernel_start, "RMSD Kernel");

        float* rmsdSubsetHost = new float[nb_frames_subset*nb_frames_subset];
        CHECK_SUCCESS(cudaMemcpy(rmsdSubsetHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Memcpy rmsd -> rmsdSubsetHost");

        for(int i=row_begin; i < row_end; ++i) {
            for(int j=col_begin; j < col_end; ++j) {

                int r = (i - row_begin);
                int c = (col_begin == row_begin)
                ? (j - col_begin)
                : (size_row + (j - col_begin));

                float v = rmsdSubsetHost[r * nb_frames_subset + c];

                rmsdHost[i * (int)N_frames + j] = v;
                rmsdHost[j * (int)N_frames + i] = v;

                // rmsdHost[i*static_cast<int>(N_frames) + j] = rmsdSubsetHost[(i-row_begin)*nb_frames_subset + j - col_begin];
                // rmsdHost[j*static_cast<int>(N_frames) + i] = rmsdSubsetHost[(i-row_begin)*nb_frames_subset + j - col_begin];
            }
        }

        if(col_end == (int) N_frames) {
            row_begin += SQ_SUBMATRIX_SIZE;
        };

        cudaFree(frameGPU);
        cudaFree(rmsd);

    }

    chrono_type clustering_loop_start = chrono_time::now();
    // Pick first K unique indices
    int* centroids = new int[K];
    int* clusters = new int[N_frames];

    float db_index = runKMedoids(N_frames, K, rmsdHost, MAX_ITER, centroids, clusters);
    // float db_index = k_analysis(rmsdHost, N_frames, MAX_ITER);

    std::cout << "Davies–Bouldin index: " << db_index << std::endl;

    measure_seconds(clustering_loop_start, "Clustering loop");
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
    // cudaFree(frameGPU);
    // cudaFree(rmsd);

    return 0;
}
