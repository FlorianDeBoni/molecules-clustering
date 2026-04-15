#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include "CudaTimer.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// using chrono_type = std::chrono::high_resolution_clock::time_point;
// using chrono_time = std::chrono::high_resolution_clock;

// static inline double elapsed_s(const chrono_type& start) {
//     return std::chrono::duration<double>(chrono_time::now() - start).count();
// }

// void exportClusteringToJSON(
//     const char* filename,
//     const float* frame,
//     const int* clusters,
//     const int* centroids,
//     int N_frames,
//     int N_atoms,
//     int N_dims,
//     int K
// );

// void exportClusteringToJSON(
//     const char* filename,
//     const float* frame,
//     const int*   clusters,
//     const int*   centroids,
//     int N_frames,
//     int N_atoms,
//     int N_dims,
//     int K
// );

void exportClusteringToJSON(
    const char* filename,
    const float* frame,
    const int*   clusters,
    const int*   centroids,
    int N_frames,
    int N_atoms,
    int N_dims,
    int K
) {
    std::ofstream file(filename);
 
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for writing" << std::endl;
        return;
    }
 
    file << std::setprecision(6) << std::fixed;
 
    // ── metadata ────────────────────────────────────────────────────────────
    file << "{\n";
    file << "  \"metadata\": {\n";
    file << "    \"n_frames\": "     << N_frames << ",\n";
    file << "    \"n_atoms\": "      << N_atoms  << ",\n";
    file << "    \"n_dimensions\": " << N_dims   << ",\n";
    file << "    \"n_clusters\": "   << K        << "\n";
    file << "  },\n";
 
    // ── centroids list ───────────────────────────────────────────────────────
    file << "  \"centroids\": [";
    for (int k = 0; k < K; k++) {
        file << centroids[k];
        if (k < K - 1) file << ", ";
    }
    file << "],\n";
 
    // ── snapshots ────────────────────────────────────────────────────────────
    file << "  \"snapshots\": [\n";
 
    // stride between two consecutive snapshots in the AoS flat array
    const int snapshot_stride = N_atoms * N_dims;   // == N_atoms * 3
 
    for (int f = 0; f < N_frames; f++) {
 
        // Check whether this snapshot is a medoid
        bool is_centroid = false;
        for (int k = 0; k < K; k++) {
            if (centroids[k] == f) { is_centroid = true; break; }
        }
 
        file << "    {\n";
        file << "      \"id\": "         << f             << ",\n";
        file << "      \"cluster\": "    << clusters[f]   << ",\n";
        file << "      \"is_centroid\": " << (is_centroid ? "true" : "false") << ",\n";
        file << "      \"atoms\": [\n";
 
        // Base offset for snapshot f in the AoS array
        const int base = f * snapshot_stride;
 
        for (int a = 0; a < N_atoms; a++) {
            // AoS layout: [f][a][0=x, 1=y, 2=z]
            float x = frame[base + a * 3 + 0];
            float y = frame[base + a * 3 + 1];
            float z = frame[base + a * 3 + 2];
 
            file << "        {\"x\": " << x
                 << ", \"y\": "        << y
                 << ", \"z\": "        << z << "}";
 
            if (a < N_atoms - 1) file << ",";
            file << "\n";
        }
 
        file << "      ]\n";
        file << "    }";
        if (f < N_frames - 1) file << ",";
        file << "\n";
    }
 
    file << "  ]\n";
    file << "}\n";
 
    file.close();
    std::cout << "Clustering results exported to " << filename << std::endl;
}



void exportRMSDMatrix(
    const char* filename,
    const float* packed,
    size_t N
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float val = getRMSD(i, j, packed, N);
            file << val;
            if (j < N - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "RMSD matrix exported to " << filename << std::endl;
}

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
    chrono_type global_start = chrono_time::now();

    CudaTimer timer;

    if(argc < 2){
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    timer.start("1. Loading .bin");
    FileUtils file(args[1]);
    size_t N_frames = 1000;
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
    // Store only the upper triangle instead of the full N²  matrix.
    // Size = N*(N-1)/2.  Element (i,j) with i<j lives at getRMSD() index.
    size_t upper_triangle_size = (N_frames * (N_frames - 1)) / 2;
    float* rmsdUpperTriangle = new float[upper_triangle_size]();   // zero-init

    // Chunk result buffer: worst-case tile is NB_FRAMES_PER_CHUNK × NB_FRAMES_PER_CHUNK
    // but we only allocate once at max size and reuse it every tile.
    float* rmsdHostChunk = new float[NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // -----------------------------------------------------------------------
    // GPU buffers
    // -----------------------------------------------------------------------
    float *d_references, *d_targets, *d_rmsd;
    CUDA_CHECK(cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets,    NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
    // d_rmsd: worst-case tile is chunk × chunk.
    // NOTE: this is the allocation that silently failed before (no error check +
    // chunk² could exceed GPU memory).  We now fail loudly if it's too large.
    CUDA_CHECK(cudaMalloc(&d_rmsd, NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));

    float *d_cx_ref, *d_cy_ref, *d_cz_ref, *d_G_ref;
    float *d_cx_tgt, *d_cy_tgt, *d_cz_tgt, *d_G_tgt;
    CUDA_CHECK(cudaMalloc(&d_cx_ref, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy_ref, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz_ref, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_ref,  NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cx_tgt, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy_tgt, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz_tgt, NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_tgt,  NB_FRAMES_PER_CHUNK * sizeof(float)));

    // [Optimization A] Centroid cache — one slot per chunk index.
    // All centroids are precomputed once before the RMSD double loop.
    // Each chunk c occupies a contiguous slice of NB_FRAMES_PER_CHUNK floats
    // starting at c * NB_FRAMES_PER_CHUNK.
    float *d_cx_cache, *d_cy_cache, *d_cz_cache, *d_G_cache;
    size_t cache_slots = NB_ROW_ITERATIONS * NB_FRAMES_PER_CHUNK;
    CUDA_CHECK(cudaMalloc(&d_cx_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz_cache, cache_slots * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_cache,  cache_slots * sizeof(float)));

    dim3 threads(64, 16);
    size_t total_centroid_frames = 0;
    size_t total_rmsd_pairs      = 0;

    // -----------------------------------------------------------------------
    // Precompute centroids for every chunk once, before the RMSD double loop.
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
    // Debug capture buffer: raw 2D window around the first chunk boundary.
    // Initialised to -1 so unwritten cells are visible in the printout.
    // -----------------------------------------------------------------------
    const size_t window = 5;
    const size_t dbg_start = NB_FRAMES_PER_CHUNK - window;
    const size_t dbg_end   = NB_FRAMES_PER_CHUNK + window;  // exclusive
    const size_t dbg_size  = 2 * window;
    std::vector<float> dbg(dbg_size * dbg_size, -1.0f);

    // -----------------------------------------------------------------------
    // RMSD computation
    // -----------------------------------------------------------------------
    timer.start("2.2. Computation Global RMSD");
    for(size_t row = 0; row < NB_ROW_ITERATIONS; row++)
    {
        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        size_t nb_ref    = stop_row - start_row;

        std::cout << "Processing row chunk " << row + 1 << "/" << NB_ROW_ITERATIONS
                  << " (" << nb_ref << " frames)\n";

        // extractSnapshotsFastInPlace uses exclusive [start, end) convention.
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        CUDA_CHECK(cudaMemcpy(d_references, references_coordinates.data(),
                              references_coordinates.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        // auto c0 = chrono_time::now();
        // computeCentroidsG<<<(nb_ref + 127) / 128, 128>>>(
        //     d_references, N_atoms, nb_ref,
        //     d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // centroid_time += elapsed_s(c0);
        // total_centroid_frames += nb_ref;

        for(size_t col = row; col < NB_ROW_ITERATIONS; col++)
        {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt    = stop_col - start_col;

            dim3 blocks((nb_tgt + threads.x - 1) / threads.x,
                        (nb_ref + threads.y - 1) / threads.y);

            // TILE matches blockDim.x (= threads.x = 32) as used inside the kernel.
            const int TILE      = threads.x;
            size_t smem_bytes   = 3 * TILE * threads.y * sizeof(float);

            if (row == col) {
                RMSD_diagonal<<<blocks, threads, smem_bytes>>>(
                    d_references, N_atoms, nb_ref,
                    d_cx_cache + row * NB_FRAMES_PER_CHUNK,
                    d_cy_cache + row * NB_FRAMES_PER_CHUNK,
                    d_cz_cache + row * NB_FRAMES_PER_CHUNK,
                    d_G_cache  + row * NB_FRAMES_PER_CHUNK,
                    d_rmsd);
                CUDA_CHECK(cudaDeviceSynchronize());
                total_rmsd_pairs += nb_ref * nb_tgt;
            }
            else {
                file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
                CUDA_CHECK(cudaMemcpy(d_targets, targets_coordinates.data(),
                                    targets_coordinates.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

                // computeCentroidsG<<<(nb_tgt + 127) / 128, 128>>>(
                //     d_targets, N_atoms, nb_tgt,
                //     d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt);
                // CUDA_CHECK(cudaDeviceSynchronize());
                // centroid_time += elapsed_s(c2);
                // total_centroid_frames += nb_tgt;
                
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
            }

            // Copy only the nb_ref × nb_tgt result actually written by the kernel.
            CUDA_CHECK(cudaMemcpy(rmsdHostChunk, d_rmsd,
                                  nb_ref * nb_tgt * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Capture raw computed values into the debug window (no symmetry tricks).
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

            // Pack into upper-triangle storage (row == col diagonal tile
            // contains RMSD(i,i)=0 on the diagonal; off-diagonal pairs are
            // stored with i < j).
            for(size_t i = 0; i < nb_ref; i++)
            {
                size_t global_i = start_row + i;
                for(size_t j = 0; j < nb_tgt; j++)
                {
                    size_t global_j = start_col + j;
                    if(global_i >= global_j) continue;   // skip diagonal & lower triangle

                    // Upper-triangle packed index for (global_i, global_j), i < j:
                    //   idx = i*N - i*(i+1)/2 + (j - i - 1)
                    size_t idx = global_i * N_frames
                                 - (global_i * (global_i + 1)) / 2
                                 + (global_j - global_i - 1);

                    rmsdUpperTriangle[idx] = rmsdHostChunk[i * nb_tgt + j];
                }
            }
        }
    }

    timer.stop("2.2. Computation Global RMSD");
    timer.stop("2. Computing RMSD");

    // // -----------------------------------------------------------------------
    // // Throughput
    // // -----------------------------------------------------------------------
    // std::cout << "\n===== PERFORMANCE =====\n";
    // std::cout << "Centroid compute : " << total_centroid_frames / centroid_time
    //           << " molecules/s (" << centroid_time << " s)\n";
    // std::cout << "RMSD kernel      : " << total_rmsd_pairs / rmsd_time
    //           << " RMSD/s (" << rmsd_time << " s)\n";
    // std::cout << "Full pipeline    : " << total_rmsd_pairs / pipeline_time
    //           << " RMSD/s (" << pipeline_time << " s)\n";
    
    // saveArrayToFile("output/RMSD_centroid.txt", rmsdUpperTriangle, upper_triangle_size);

    // -----------------------------------------------------------------------
    // DEBUG: inspect RMSD around chunk boundary (raw, no symmetry)
    // -----------------------------------------------------------------------
    std::cout << "\n===== RMSD TILE JUNCTION DEBUG =====\n";
    std::cout << "Inspecting frames "
              << dbg_start << " .. " << dbg_end - 1
              << " around chunk boundary at frame "
              << NB_FRAMES_PER_CHUNK << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    // column header
    std::cout << std::setw(10) << "";
    for (size_t j = dbg_start; j < dbg_end; j++) {
        std::string label = (j == NB_FRAMES_PER_CHUNK ? ">" : "") + std::to_string(j);
        std::cout << std::setw(10) << label;
    }
    std::cout << "\n";

    // rows
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
    // K-Medoids clustering  (rmsdUpperTriangle already in packed format)
    // -----------------------------------------------------------------------
    timer.start("3. Computing Clusters");
    timer.start("3.1. Clustering setup");
    int K        = 5;
    int MAX_ITER = 50;


    // Allocating GPU arrays
    int* centroids = new int[K];
    int* d_centroids;
    CUDA_CHECK(cudaMalloc(&d_centroids, K * sizeof(int)));

    int* clusters  = new int[N_frames];
    int* d_clusters;
    CUDA_CHECK(cudaMalloc(&d_clusters, N_frames * sizeof(int)));

    float* d_rmsdUpperTriangle = nullptr;
    size_t tri_bytes = upper_triangle_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_rmsdUpperTriangle, tri_bytes));
    CUDA_CHECK(cudaMemcpy(d_rmsdUpperTriangle, rmsdUpperTriangle, tri_bytes,
                cudaMemcpyHostToDevice));
    
    float* d_frame_costs;
    CUDA_CHECK(cudaMalloc(&d_frame_costs, N_frames * sizeof(float)));


    // Pick first K unique indices
    pickKMedoidsPlusPlus(N_frames, K, rmsdUpperTriangle, centroids);
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids, K*sizeof(int), cudaMemcpyHostToDevice));

    timer.stop("3.1. Clustering setup");


    // Assignment step params
    dim3 clusteringThreads(1024);
    dim3 clusteringBlocks((N_frames + clusteringThreads.x - 1) / clusteringThreads.x);
    // Centroids update step params
    dim3 threadsPerClusterBlock(1024);
    dim3 reducingBlocks(K);
    size_t sharedMemSize = threadsPerClusterBlock.x * (sizeof(float) + sizeof(int));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        AssignClusters<<<clusteringBlocks, clusteringThreads>>>(
            N_frames,
            K,
            d_rmsdUpperTriangle,
            d_centroids,
            d_clusters,
            d_frame_costs
        );
        // Making sure all assignments are set across mutiliple blocks
        CUDA_CHECK(cudaDeviceSynchronize());

        ComputeMedoidCosts<<<clusteringBlocks, clusteringThreads>>>(
            N_frames,
            d_rmsdUpperTriangle,
            d_clusters,
            d_frame_costs
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        UpdateMedoids<<<K, threadsPerClusterBlock, sharedMemSize>>>(
            N_frames,
            d_centroids,
            d_clusters,
            d_frame_costs
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copying results back
    CUDA_CHECK(cudaMemcpy(centroids, d_centroids, K * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clusters, d_clusters, N_frames * sizeof(int), cudaMemcpyDeviceToHost));

    timer.stop("3. Computing Clusters");


    // -----------------------------------------------------------------------
    // Final Clustering Results:
    // -----------------------------------------------------------------------
    float db_index = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdUpperTriangle);


    std::cout << "\n===== K-MEDOIDS =====\n";
    std::cout << "Davies-Bouldin index : " << db_index << "\n";

    float random_db = runRandomClustering(N_frames, K, rmsdUpperTriangle);
    std::cout << "Random DB index : " << random_db << "\n";
    std::cout << "Improvement : " << ((random_db - db_index) / random_db) * 100.0 << "%\n";

    saveClusters(clusters, N_frames, centroids, K);

    std::cout << "Exporting data to output/clustering_results.json" << std::endl;

    exportClusteringToJSON(
        "output/clustering_results.json",
        all_data.data(),
        clusters,
        centroids,
        N_frames,
        N_atoms,
        N_dims,
        K
    );

    exportRMSDMatrix("output/rmsd_matrix.csv", all_data.data(), N_frames);


    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_rmsdUpperTriangle));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_clusters));
    CUDA_CHECK(cudaFree(d_frame_costs));
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
