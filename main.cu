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

// ─── throughput helper ───────────────────────────────────────────────────────
static inline double elapsed_s(const chrono_type& start) {
    return std::chrono::duration<double>(chrono_time::now() - start).count();
}

static void print_throughput(const std::string& label,
                             double seconds,
                             size_t frames,
                             int label_width = 35)
{
    double fps = (seconds > 0.0) ? frames / seconds : 0.0;
    std::cout << std::left  << std::setw(label_width) << ("  [" + label + "]")
              << std::right << std::fixed
              << std::setw(12) << std::setprecision(0) << fps << " frames/s"
              << "   (" << std::setprecision(3) << seconds << " s)\n";
}
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** args) {

    chrono_type global_start = chrono_time::now();

    std::string file_name;
    if (argc >= 2) {
        file_name = args[1];
    } else {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>" << std::endl;
        return 1;
    }

    FileUtils file(file_name);

    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    // ── Read .bin file ────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "LOADING BINARY FILE\n";
    std::cout << std::string(70, '=') << "\n";

    std::vector<float> all_data(N_frames * N_atoms * 3);

    chrono_type t_read = chrono_time::now();
    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);
    double read_s = elapsed_s(t_read);

    std::cout << "Loaded " << N_frames * N_atoms * N_dims * sizeof(float) / (1024*1024)
              << " MiB into CPU RAM.\n";
    print_throughput("Read .bin", read_s, N_frames);

    // ── Chunk sizing ──────────────────────────────────────────────────────────
    const size_t MAX_DATA_CHUNK_SIZE  = 12000; // MB
    const size_t NB_FRAMES_PER_CHUNK  = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS    = (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);
    const size_t RMSD_LOOPS_NEEDED    = NB_ROW_ITERATIONS * (NB_ROW_ITERATIONS + 1) / 2;

    std::cout << "Max frames per chunk: " << NB_FRAMES_PER_CHUNK << "\n";
    std::cout << "Number of RMSD iterations: " << RMSD_LOOPS_NEEDED << "\n";

    // ── Allocations ───────────────────────────────────────────────────────────
    size_t rmsd_all_size   = N_frames * N_frames;
    float* rmsdHostAll     = new float[rmsd_all_size];

    size_t rmsd_chunk_size = NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK;
    float* rmsdHostChunk   = new float[rmsd_chunk_size];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    float* d_references = nullptr;
    float* d_targets    = nullptr;
    float* d_rmsd       = nullptr;

    CHECK_SUCCESS(cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)), "Allocating memory for references");
    CHECK_SUCCESS(cudaMalloc(&d_targets,    NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)), "Allocating memory for targets");
    CHECK_SUCCESS(cudaMalloc(&d_rmsd,       NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)), "Allocating rmsd vector on GPU");

    dim3 threads(16, 16);
    size_t size_rmsd = NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float);

    // ── Accumulators for aggregate throughput ─────────────────────────────────
    double total_extract_s  = 0.0;
    double total_kernel_s   = 0.0;
    size_t total_rmsd_pairs = 0;

    size_t iter = 0;

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "RMSD COMPUTATION START\n";
    std::cout << std::string(70, '=') << "\n";

    for (size_t row = 0; row < NB_ROW_ITERATIONS; ++row) {
        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        const size_t nb_ref = stop_row - start_row;

        // ── Extract references chunk ──────────────────────────────────────────
        chrono_type t_extract_ref = chrono_time::now();
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        total_extract_s += elapsed_s(t_extract_ref);

        CHECK_SUCCESS(cudaMemcpy(d_references, references_coordinates.data(),
                                 references_coordinates.size() * sizeof(float),
                                 cudaMemcpyHostToDevice), "Copying References on GPU");

        for (size_t col = row; col < NB_ROW_ITERATIONS; ++col) {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            const size_t nb_tgt = stop_col - start_col;

            std::cout << "\nIteration " << ++iter << "/" << RMSD_LOOPS_NEEDED
                      << "  Row [" << start_row << "," << stop_row << ")"
                      << "  Col [" << start_col << "," << stop_col << ")\n";

            // ── Extract targets chunk ─────────────────────────────────────────
            chrono_type t_extract_tgt = chrono_time::now();
            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
            double ext_s = elapsed_s(t_extract_tgt);
            total_extract_s += ext_s;
            print_throughput("Extract targets chunk", ext_s, nb_tgt);

            CHECK_SUCCESS(cudaMemcpy(d_targets, targets_coordinates.data(),
                                     targets_coordinates.size() * sizeof(float),
                                     cudaMemcpyHostToDevice), "Copying Targets on GPU");

            // ── RMSD kernel ───────────────────────────────────────────────────
            dim3 blocks((nb_tgt + threads.x - 1) / threads.x,
                        (nb_ref + threads.y - 1) / threads.y);

            CHECK_SUCCESS(cudaDeviceSynchronize(), "Ready to launch RMSD Kernel");

            chrono_type t_kernel = chrono_time::now();
            RMSD<<<blocks, threads>>>(
                d_references, d_targets,
                nb_ref, nb_tgt, N_atoms,
                d_rmsd
            );
            CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD Kernel");
            double kern_s = elapsed_s(t_kernel);
            total_kernel_s  += kern_s;
            total_rmsd_pairs += nb_ref * nb_tgt;

            print_throughput("RMSD kernel (pairs/s)", kern_s, nb_ref * nb_tgt);

            CHECK_SUCCESS(cudaMemcpy(rmsdHostChunk, d_rmsd, size_rmsd,
                                     cudaMemcpyDeviceToHost), "Copying RMSD chunk to CPU");

            // ── Scatter chunk into full matrix ────────────────────────────────
            for (size_t i = 0; i < nb_ref; ++i) {
                for (size_t j = 0; j < nb_tgt; ++j) {
                    size_t global_row = start_row + i;
                    size_t global_col = start_col + j;
                    size_t chunk_idx  = i * nb_tgt + j;
                    size_t global_idx = global_row * N_frames + global_col;
                    rmsdHostAll[global_idx] = rmsdHostChunk[chunk_idx];
                }
            }

            // ── Print bottom-right 5x5 corner of first tile ──────────────────
            if (row == 0 && col == 0) {
                size_t preview = 5;
                size_t corner  = nb_ref - preview;
                std::cout << "\n  Bottom-right 5x5 corner of tile (0,0)"
                          << " [frames " << corner << ".." << (corner + preview - 1) << "]:\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << std::setw(10) << "";
                for (size_t j = 0; j < preview; ++j)
                    std::cout << std::setw(10) << (corner + j);
                std::cout << "\n";
                for (size_t i = 0; i < preview; ++i) {
                    std::cout << std::setw(10) << (corner + i);
                    for (size_t j = 0; j < preview; ++j) {
                        size_t chunk_idx = (corner + i) * nb_tgt + (corner + j);
                        std::cout << std::setw(10) << rmsdHostChunk[chunk_idx];
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
        }
    }

    // ── Aggregate RMSD throughput ─────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "RMSD PHASE SUMMARY\n";
    print_throughput("Extract (all chunks, avg)", total_extract_s, N_frames);
    print_throughput("Kernel  (all chunks, pairs/s)", total_kernel_s, total_rmsd_pairs);

    CHECK_SUCCESS(cudaFree(d_references), "Freeing References on GPU");
    CHECK_SUCCESS(cudaFree(d_rmsd),       "Freeing RMSD vector on GPU");
    CHECK_SUCCESS(cudaFree(d_targets),    "Freeing Targets on GPU");

    // ── Pack upper triangle ───────────────────────────────────────────────────
    size_t upper_triangle_size = (N_frames * (N_frames - 1)) / 2;
    float* rmsdUpperTriangle   = new float[upper_triangle_size];

    size_t idx = 0;
    for (size_t i = 0; i < N_frames; ++i)
        for (size_t j = i + 1; j < N_frames; ++j)
            rmsdUpperTriangle[idx++] = rmsdHostAll[i * N_frames + j];

    // ── Debug: 6x6 window centered on bottom-right corner of tile (0,0) ──────

    size_t p = NB_FRAMES_PER_CHUNK;
    std::cout << "\n  6x6 window around tile (0,0) corner (boundary at frame " << p << "):\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(10) << "";
    for (size_t j = p - 3; j < p + 3; ++j) {
        std::string lbl = (j == p ? ">" : "") + std::to_string(j);
        std::cout << std::setw(10) << lbl;
    }
    std::cout << "\n";
    for (size_t i = p - 3; i < p + 3; ++i) {
        std::string lbl = (i == p ? ">" : "") + std::to_string(i);
        std::cout << std::setw(10) << lbl;
        for (size_t j = p - 3; j < p + 3; ++j)
            std::cout << std::setw(10) << rmsdHostAll[i * N_frames + j];
        std::cout << "\n";
    }
    std::cout << "  (> marks first frame of next tile)\n\n";

    delete[] rmsdHostAll;

    int K        = 10;
    int MAX_ITER = 50;

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "RMSD COMPUTATION COMPLETE\n";
    std::cout << std::string(70, '=') << "\n";
    measure_seconds(global_start, "Total RMSD computation time");

    // ── K-medoids ─────────────────────────────────────────────────────────────
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "K-MEDOIDS CLUSTERING (K=" << K << ", MAX_ITER=" << MAX_ITER << ")\n";
    std::cout << std::string(70, '=') << "\n";

    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    chrono_type t_clust = chrono_time::now();
    float db_index = runKMedoids(N_frames, K, rmsdUpperTriangle, MAX_ITER, centroids, clusters);
    double clust_s = elapsed_s(t_clust);

    print_throughput("Cluster assignments (frames/s)", clust_s, N_frames);
    std::cout << "\n";

    // ── Results ───────────────────────────────────────────────────────────────
    std::cout << std::string(70, '=') << "\n";
    std::cout << "CLUSTERING RESULTS\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "K-medoids Davies-Bouldin Index: " << db_index << "\n";

    std::vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < (int)N_frames; i++)
        cluster_sizes[clusters[i]]++;

    std::cout << "\nCluster centroids and sizes:\n";
    for (int k = 0; k < K; k++) {
        float percent = 100.0f * cluster_sizes[k] / N_frames;
        std::cout << "  Cluster " << std::setw(2) << k
                  << " | Centroid: frame " << std::setw(6) << centroids[k]
                  << " | Size: "           << std::setw(6) << cluster_sizes[k]
                  << " (" << std::setw(5) << std::setprecision(2) << percent << "%)\n";
    }

    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "BASELINE COMPARISON\n";
    std::cout << std::string(70, '-') << "\n";

    float random_db_index = runRandomClustering(N_frames, K, rmsdUpperTriangle);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Random clustering Davies-Bouldin Index: " << random_db_index << "\n";

    float improvement = ((random_db_index - db_index) / random_db_index) * 100.0f;
    std::cout << "\nK-medoids improvement over random: "
              << std::setprecision(2) << improvement << "%"
              << (improvement > 0 ? " ✓ BETTER" : " ✗ WORSE") << "\n";

    std::cout << std::string(70, '=') << "\n\n";

    saveClusters(clusters, N_frames, centroids, K);

    measure_seconds(global_start, "Total program execution time");

    delete[] centroids;
    delete[] rmsdUpperTriangle;
    delete[] clusters;

    return 0;
}
