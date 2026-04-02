#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include "CudaTimer.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                       \
    } while (0)

int main(int argc, char** args)
{
    CudaTimer timer;

    if(argc < 2){
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    timer.start("1. Loading .bin");
    FileUtils file(args[1]);
    size_t N_frames = 10000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "\n===== DATASET INFO =====\n";
    std::cout << "Frames : " << N_frames << "\n";
    std::cout << "Atoms  : " << N_atoms  << "\n\n";

    std::vector<float> all_data(N_frames * N_atoms * 3);
    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);
    timer.stop("1. Loading .bin");

    // -----------------------------------------------------------------------
    // Chunk sizing
    // -----------------------------------------------------------------------
    const size_t MAX_DATA_CHUNK_SIZE   = 500;
    const size_t NB_FRAMES_PER_CHUNK   = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS     = (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

    std::cout << "Chunk size     : " << NB_FRAMES_PER_CHUNK << " frames\n";
    std::cout << "Row iterations : " << NB_ROW_ITERATIONS   << "\n\n";

    // -----------------------------------------------------------------------
    // Host buffers
    // -----------------------------------------------------------------------
    size_t upper_triangle_size = (N_frames * (N_frames - 1)) / 2;
    float* rmsdUpperTriangle = new float[upper_triangle_size]();

    float* rmsdHostChunk = new float[NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // -----------------------------------------------------------------------
    // GPU buffers
    // -----------------------------------------------------------------------
    float *d_references, *d_targets, *d_rmsd;
    CUDA_CHECK(cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets,    NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rmsd, NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));

    float *d_cx_cache, *d_cy_cache, *d_cz_cache, *d_G_cache;
    size_t cache_slots = NB_ROW_ITERATIONS * NB_FRAMES_PER_CHUNK;
    CUDA_CHECK(cudaMalloc(&d_cx_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_cache,  cache_slots * sizeof(float)));

    dim3 threads(64, 8);
    size_t total_centroid_frames = 0;
    size_t total_rmsd_pairs      = 0;

    // -----------------------------------------------------------------------
    // Precompute centroids
    // -----------------------------------------------------------------------
    timer.start("2. Computing RMSD");
    timer.start("2.1. Computation centroids");
    std::vector<float> chunk_coords;
    for (size_t c = 0; c < NB_ROW_ITERATIONS; c++)
    {
        size_t start_c = c * NB_FRAMES_PER_CHUNK;
        size_t stop_c  = std::min(start_c + NB_FRAMES_PER_CHUNK, N_frames);
        size_t nb_c    = stop_c - start_c;

        file.extractSnapshotsFastInPlace(start_c, stop_c, all_data, chunk_coords);
        CUDA_CHECK(cudaMemcpy(d_references, chunk_coords.data(),
                                chunk_coords.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

        computeCentroidsG<<<(nb_c + 127) / 128, 128>>>(
            d_references, N_atoms, nb_c,
            d_cx_cache + c * NB_FRAMES_PER_CHUNK,
            d_cy_cache + c * NB_FRAMES_PER_CHUNK,
            d_cz_cache + c * NB_FRAMES_PER_CHUNK,
            d_G_cache  + c * NB_FRAMES_PER_CHUNK);
        CUDA_CHECK(cudaDeviceSynchronize());
        total_centroid_frames += nb_c;
    }
    timer.stop("2.1. Computation centroids");

    // -----------------------------------------------------------------------
    // Debug capture buffer
    // -----------------------------------------------------------------------
    const size_t window    = 5;
    const size_t dbg_start = NB_FRAMES_PER_CHUNK - window;
    const size_t dbg_end   = NB_FRAMES_PER_CHUNK + window;
    const size_t dbg_size  = 2 * window;
    std::vector<float> dbg(dbg_size * dbg_size, -1.0f);

    // -----------------------------------------------------------------------
    // RMSD computation — transfer/write accumulators
    // -----------------------------------------------------------------------
    float acc_h2d_ms = 0.f, acc_d2h_ms = 0.f, acc_tri_ms = 0.f;

    timer.start("2.2. Computation Global RMSD");
    for(size_t row = 0; row < NB_ROW_ITERATIONS; row++)
    {
        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        size_t nb_ref    = stop_row - start_row;

        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);

        timer.start("_h2d");
        CUDA_CHECK(cudaMemcpy(d_references, references_coordinates.data(),
                              references_coordinates.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        timer.stopAccum("_h2d", acc_h2d_ms);

        for(size_t col = row; col < NB_ROW_ITERATIONS; col++)
        {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt    = stop_col - start_col;

            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);

            timer.start("_h2d");
            CUDA_CHECK(cudaMemcpy(d_targets, targets_coordinates.data(),
                                  targets_coordinates.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            timer.stopAccum("_h2d", acc_h2d_ms);

            dim3 blocks((nb_tgt + threads.x - 1) / threads.x,
                        (nb_ref + threads.y - 1) / threads.y);

            const int TILE    = threads.x;
            size_t smem_bytes = 3 * TILE * threads.y * sizeof(float);

            RMSD<<<blocks, threads, smem_bytes>>>(
                d_references, d_targets, N_atoms, nb_ref, nb_tgt,
                d_cx_cache + row * NB_FRAMES_PER_CHUNK,
                d_cy_cache + row * NB_FRAMES_PER_CHUNK,
                d_cz_cache + row * NB_FRAMES_PER_CHUNK,
                d_G_cache  + row * NB_FRAMES_PER_CHUNK,
                d_cx_cache + col * NB_FRAMES_PER_CHUNK,
                d_cy_cache + col * NB_FRAMES_PER_CHUNK,
                d_cz_cache + col * NB_FRAMES_PER_CHUNK,
                d_G_cache  + col * NB_FRAMES_PER_CHUNK,
                d_rmsd);
            CUDA_CHECK(cudaDeviceSynchronize());
            total_rmsd_pairs += nb_ref * nb_tgt;

            timer.start("_d2h");
            CUDA_CHECK(cudaMemcpy(rmsdHostChunk, d_rmsd,
                                  nb_ref * nb_tgt * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            timer.stopAccum("_d2h", acc_d2h_ms);

            // Capture debug window
            for (size_t i = 0; i < nb_ref; i++) {
                size_t gi = start_row + i;
                if (gi < dbg_start || gi >= dbg_end) continue;
                for (size_t j = 0; j < nb_tgt; j++) {
                    size_t gj = start_col + j;
                    if (gj < dbg_start || gj >= dbg_end) continue;
                    dbg[(gi - dbg_start) * dbg_size + (gj - dbg_start)] =
                        rmsdHostChunk[i * nb_tgt + j];
                }
            }

            // Pack into upper-triangle storage
            timer.start("_tri");
            for(size_t i = 0; i < nb_ref; i++)
            {
                size_t global_i = start_row + i;
                for(size_t j = 0; j < nb_tgt; j++)
                {
                    size_t global_j = start_col + j;
                    if(global_i >= global_j) continue;

                    size_t idx = global_i * N_frames
                                 - (global_i * (global_i + 1)) / 2
                                 + (global_j - global_i - 1);

                    rmsdUpperTriangle[idx] = rmsdHostChunk[i * nb_tgt + j];
                }
            }
            timer.stopAccum("_tri", acc_tri_ms);
        }
    }
    timer.stop("2.2. Computation Global RMSD");
    timer.stop("2. Computing RMSD");

    // -----------------------------------------------------------------------
    // Transfer / write breakdown
    // -----------------------------------------------------------------------
    printf("\n%-30s %10s\n", "Transfer/Write breakdown", "Time (s)");
    printf("%s\n", std::string(42, '-').c_str());
    printf("%-30s %10.3f s\n", "  H->D (coords)",    acc_h2d_ms / 1000.f);
    printf("%-30s %10.3f s\n", "  D->H (chunks)",    acc_d2h_ms / 1000.f);
    printf("%-30s %10.3f s\n", "  Triangle writes",  acc_tri_ms / 1000.f);
    printf("%-30s %10.3f s\n", "  Total",
           (acc_h2d_ms + acc_d2h_ms + acc_tri_ms) / 1000.f);

    // -----------------------------------------------------------------------
    // DEBUG: inspect RMSD around chunk boundary
    // -----------------------------------------------------------------------
    std::cout << "\n===== RMSD TILE JUNCTION DEBUG =====\n";
    std::cout << "Inspecting frames "
              << dbg_start << " .. " << dbg_end - 1
              << " around chunk boundary at frame "
              << NB_FRAMES_PER_CHUNK << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << std::setw(10) << "";
    for (size_t j = dbg_start; j < dbg_end; j++) {
        std::string label = (j == NB_FRAMES_PER_CHUNK ? ">" : "") + std::to_string(j);
        std::cout << std::setw(10) << label;
    }
    std::cout << "\n";

    for (size_t i = dbg_start; i < dbg_end; i++) {
        std::string label = (i == NB_FRAMES_PER_CHUNK ? ">" : "") + std::to_string(i);
        std::cout << std::setw(10) << label;
        for (size_t j = dbg_start; j < dbg_end; j++) {
            float val = dbg[(i - dbg_start) * dbg_size + (j - dbg_start)];
            std::cout << std::setw(10) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\n( '>' marks the first frame of the next chunk )\n\n";

    // -----------------------------------------------------------------------
    // K-Medoids clustering
    // -----------------------------------------------------------------------
    int K        = 10;
    int MAX_ITER = 50;
    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    timer.start("3. Computing Clusters");
    float db_index = runKMedoids(N_frames, K, rmsdUpperTriangle, MAX_ITER, centroids, clusters);
    timer.stop("3. Computing Clusters");

    std::cout << "\n===== K-MEDOIDS =====\n";
    std::cout << "Davies-Bouldin index : " << db_index << "\n";

    float random_db = runRandomClustering(N_frames, K, rmsdUpperTriangle);
    std::cout << "Random DB index : " << random_db << "\n";
    std::cout << "Improvement : " << ((random_db - db_index) / random_db) * 100.0 << "%\n";

    saveClusters(clusters, N_frames, centroids, K);

    std::ofstream out("output/rmsd_row0.txt");
    for (size_t j = 1; j < 1000; j++) {
        size_t idx = j - 1;
        out << j << " " << rmsdUpperTriangle[idx] << "\n";
    }
    out.close();

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_references));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_rmsd));
    CUDA_CHECK(cudaFree(d_cx_cache)); CUDA_CHECK(cudaFree(d_cy_cache));
    CUDA_CHECK(cudaFree(d_cz_cache)); CUDA_CHECK(cudaFree(d_G_cache));

    timer.print();

    delete[] rmsdUpperTriangle;
    delete[] rmsdHostChunk;
    delete[] centroids;
    delete[] clusters;

    return 0;
}