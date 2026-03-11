#include "FileUtils.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <iomanip>
#include <chrono>

#include "gpu.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

static inline double elapsed_s(const chrono_type& start) {
    return std::chrono::duration<double>(chrono_time::now() - start).count();
}

int main(int argc, char** args) {

    chrono_type global_start = chrono_time::now();

    if (argc < 2) {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    std::string file_name = args[1];
    FileUtils file(file_name);

    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "\nDataset info\n";
    std::cout << "Frames : " << N_frames << "\n";
    std::cout << "Atoms  : " << N_atoms  << "\n\n";

    // ------------------------------------------------------------
    // Load dataset
    // ------------------------------------------------------------

    std::vector<float> all_data(N_frames * N_atoms * 3);

    chrono_type t_read = chrono_time::now();

    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);

    double read_time = elapsed_s(t_read);

    std::cout << "Read dataset : "
              << (double)N_frames / read_time
              << " molecules/s (" << read_time << " s)\n";

    // ------------------------------------------------------------
    // Chunk sizing
    // ------------------------------------------------------------

    const size_t MAX_DATA_CHUNK_SIZE = 12000;

    const size_t NB_FRAMES_PER_CHUNK =
        get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);

    const size_t NB_ROW_ITERATIONS =
        (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

    std::cout << "Frames per chunk : " << NB_FRAMES_PER_CHUNK << "\n";
    std::cout << "Chunk iterations : " << NB_ROW_ITERATIONS << "\n";

    // ------------------------------------------------------------
    // RMSD matrix allocation
    // ------------------------------------------------------------

    size_t rmsd_all_size = N_frames * N_frames;

    float* rmsdHostAll = new float[rmsd_all_size];

    size_t rmsd_chunk_size =
        NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK;

    float* rmsdHostChunk = new float[rmsd_chunk_size];

    // ------------------------------------------------------------
    // Host buffers
    // ------------------------------------------------------------

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // ------------------------------------------------------------
    // GPU buffers
    // ------------------------------------------------------------

    float *d_references = nullptr;
    float *d_targets    = nullptr;
    float *d_rmsd       = nullptr;

    cudaMalloc(&d_references,
        NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float));

    cudaMalloc(&d_targets,
        NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float));

    cudaMalloc(&d_rmsd,
        NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float));

    // centroid + G

    float *d_cx_ref,*d_cy_ref,*d_cz_ref,*d_G_ref;
    float *d_cx_tgt,*d_cy_tgt,*d_cz_tgt,*d_G_tgt;

    cudaMalloc(&d_cx_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_ref , NB_FRAMES_PER_CHUNK*sizeof(float));

    cudaMalloc(&d_cx_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_tgt , NB_FRAMES_PER_CHUNK*sizeof(float));

    dim3 threads(32,8);

    // ------------------------------------------------------------
    // CUDA timers
    // ------------------------------------------------------------

    cudaEvent_t start_evt, stop_evt;

    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    float centroid_time_ms = 0.0f;
    float rmsd_time_ms     = 0.0f;

    size_t total_rmsd_pairs = 0;
    size_t total_molecules_processed = 0;

    chrono_type pipeline_start = chrono_time::now();

    // ============================================================
    // RMSD computation
    // ============================================================

    for (size_t row=0; row<NB_ROW_ITERATIONS; row++) {

        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);

        size_t nb_ref = stop_row - start_row;

        total_molecules_processed += nb_ref;

        file.extractSnapshotsFastInPlace(
            start_row, stop_row, all_data, references_coordinates);

        cudaMemcpy(d_references,
                   references_coordinates.data(),
                   references_coordinates.size()*sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads1D = 128;
        int blocks_ref = (nb_ref + threads1D - 1) / threads1D;

        cudaEventRecord(start_evt);

        computeCentroidsG<<<blocks_ref,threads1D>>>(
            d_references,N_atoms,nb_ref,
            d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref);

        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);

        float ms;
        cudaEventElapsedTime(&ms,start_evt,stop_evt);
        centroid_time_ms += ms;

        for(size_t col = row; col < NB_ROW_ITERATIONS; col++) {

            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);

            size_t nb_tgt = stop_col - start_col;

            total_molecules_processed += nb_tgt;

            // Extract target chunk
            file.extractSnapshotsFastInPlace(
                start_col, stop_col, all_data, targets_coordinates);

            cudaMemcpy(d_targets,
                    targets_coordinates.data(),
                    targets_coordinates.size()*sizeof(float),
                    cudaMemcpyHostToDevice);

            // Centroids for targets
            int threads1D = 128;
            int blocks_tgt = (nb_tgt + threads1D - 1) / threads1D;

            cudaEventRecord(start_evt);
            computeCentroidsG<<<blocks_tgt, threads1D>>>(
                d_targets, N_atoms, nb_tgt,
                d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt);
            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);

            float ms;
            cudaEventElapsedTime(&ms, start_evt, stop_evt);
            centroid_time_ms += ms;

            // ---------------------------
            // Shared memory RMSD launch
            // ---------------------------

            dim3 threads(32, 8); // can tune
            dim3 blocks(
                (nb_tgt + threads.x - 1) / threads.x,
                (nb_ref + threads.y - 1) / threads.y
            );

            int TILE = std::min(128, (int)N_atoms); // tile size for atoms
            size_t threads_per_block = threads.x * threads.y;
            size_t smem_bytes = 2 * 3 * TILE * threads_per_block * sizeof(float);

            cudaEventRecord(start_evt);
            RMSD<<<blocks, threads, smem_bytes>>>(
                d_references, d_targets,
                N_atoms, nb_ref, nb_tgt,
                d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref,
                d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt,
                d_rmsd
            );
            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);

            cudaEventElapsedTime(&ms, start_evt, stop_evt);
            rmsd_time_ms += ms;

            total_rmsd_pairs += nb_ref * nb_tgt;

            // Copy back to host
            cudaMemcpy(rmsdHostChunk,
                    d_rmsd,
                    nb_ref*nb_tgt*sizeof(float),
                    cudaMemcpyDeviceToHost);

            // Scatter chunk into full matrix
            for(size_t i=0;i<nb_ref;i++)
                for(size_t j=0;j<nb_tgt;j++)
                {
                    size_t global_row = start_row + i;
                    size_t global_col = start_col + j;

                    rmsdHostAll[global_row*N_frames + global_col] =
                        rmsdHostChunk[i*nb_tgt + j];
                }
        }
    }

    double pipeline_time = elapsed_s(pipeline_start);

    // ------------------------------------------------------------
    // Throughput
    // ------------------------------------------------------------

    double centroid_time_s = centroid_time_ms/1000.0;
    double rmsd_time_s     = rmsd_time_ms/1000.0;

    std::cout << "\n===== PERFORMANCE =====\n";

    std::cout << "Centroid compute : "
              << total_molecules_processed / centroid_time_s
              << " molecules/s ("
              << centroid_time_s
              << "s)\n";

    std::cout << "RMSD kernel      : "
              << total_rmsd_pairs / rmsd_time_s
              << " RMSD/s ("
              << rmsd_time_s
              << "s)\n";

    std::cout << "Full pipeline    : "
              << total_rmsd_pairs / pipeline_time
              << " RMSD/s ("
              << pipeline_time
              << "s)\n";

    // ============================================================
    // Pack upper triangle
    // ============================================================

    size_t upper_triangle_size =
        (N_frames*(N_frames-1))/2;

    float* rmsdUpperTriangle = new float[upper_triangle_size];

    size_t idx = 0;

    for(size_t i=0;i<N_frames;i++)
        for(size_t j=i+1;j<N_frames;j++)
            rmsdUpperTriangle[idx++] =
                rmsdHostAll[i*N_frames+j];

    size_t window = 5;

    std::cout << "\n===== RMSD TILE JUNCTION DEBUG =====\n";
    std::cout << "Inspecting frames "
            << NB_FRAMES_PER_CHUNK-window << " .. " << NB_FRAMES_PER_CHUNK+window-1
            << " around chunk boundary at frame "
            << NB_FRAMES_PER_CHUNK << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    // column header
    std::cout << std::setw(10) << "";
    for(size_t j=NB_FRAMES_PER_CHUNK-window;j<NB_FRAMES_PER_CHUNK+window;j++)
    {
        std::string label = (j==NB_FRAMES_PER_CHUNK ? ">" : "") + std::to_string(j);
        std::cout << std::setw(10) << label;
    }
    std::cout << "\n";

    // rows
    for(size_t i=NB_FRAMES_PER_CHUNK-window;i<NB_FRAMES_PER_CHUNK+window;i++)
    {
        std::string label = (i==NB_FRAMES_PER_CHUNK ? ">" : "") + std::to_string(i);
        std::cout << std::setw(10) << label;

        for(size_t j=NB_FRAMES_PER_CHUNK-window;j<NB_FRAMES_PER_CHUNK+window;j++)
        {
            std::cout << std::setw(10)
                    << rmsdHostAll[i*N_frames + j];
        }

        std::cout << "\n";
    }

    std::cout << "\n( '>' marks the first frame of the next chunk )\n\n";

    delete[] rmsdHostAll;

    // ============================================================
    // K-MEDOIDS CLUSTERING
    // ============================================================

    int K = 10;
    int MAX_ITER = 50;

    std::cout << "\n===== K-MEDOIDS =====\n";

    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    chrono_type t_clust = chrono_time::now();

    float db_index =
        runKMedoids(N_frames,K,rmsdUpperTriangle,
                    MAX_ITER,centroids,clusters);

    double clust_time = elapsed_s(t_clust);

    std::cout << "Clustering speed : "
              << N_frames / clust_time
              << " molecules/s ("
              << clust_time
              << "s)\n";

    std::cout << "Davies-Bouldin index : "
              << db_index << "\n";

    // ------------------------------------------------------------
    // Random baseline
    // ------------------------------------------------------------

    float random_db =
        runRandomClustering(N_frames,K,rmsdUpperTriangle);

    std::cout << "Random DB index : "
              << random_db << "\n";

    std::cout << "Improvement : "
              << ((random_db-db_index)/random_db)*100.0
              << "%\n";

    saveClusters(clusters,N_frames,centroids,K);

    // ------------------------------------------------------------

    cudaFree(d_references);
    cudaFree(d_targets);
    cudaFree(d_rmsd);

    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdHostChunk;
    delete[] rmsdUpperTriangle;
    delete[] centroids;
    delete[] clusters;

    measure_seconds(global_start,"Total program time");

    return 0;
}