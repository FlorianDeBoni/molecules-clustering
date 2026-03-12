#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using chrono_type = std::chrono::high_resolution_clock::time_point;
using chrono_time = std::chrono::high_resolution_clock;

static inline double elapsed_s(const chrono_type& start) {
    return std::chrono::duration<double>(chrono_time::now() - start).count();
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

// ---------------------------------------------------------------------------
// Number of concurrent CUDA streams.
// Tune based on GPU memory: each stream allocates its own target + rmsd buffers.
// Set to 1 to benchmark baseline with no stream overlap.
// ---------------------------------------------------------------------------
static const int N_STREAMS = 4;

int main(int argc, char** args)
{
    chrono_type global_start = chrono_time::now();

    if(argc < 2){
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    FileUtils file(args[1]);
    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "\n===== DATASET INFO =====\n";
    std::cout << "Frames  : " << N_frames << "\n";
    std::cout << "Atoms   : " << N_atoms  << "\n";
    std::cout << "Streams : " << N_STREAMS << "\n\n";

    std::vector<float> all_data(N_frames * N_atoms * 3);
    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);

    // -----------------------------------------------------------------------
    // Chunk sizing
    // -----------------------------------------------------------------------
    const size_t MAX_DATA_CHUNK_SIZE = 2500;
    const size_t NB_FRAMES_PER_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS   = (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

    std::cout << "Chunk size     : " << NB_FRAMES_PER_CHUNK << " frames\n";
    std::cout << "Row iterations : " << NB_ROW_ITERATIONS   << "\n\n";

    // -----------------------------------------------------------------------
    // Host buffers
    // -----------------------------------------------------------------------
    size_t upper_triangle_size = (N_frames * (N_frames - 1)) / 2;
    float* rmsdUpperTriangle   = new float[upper_triangle_size]();   // zero-init

    // -----------------------------------------------------------------------
    // GPU: shared reference buffers (read-only during inner loop, safe across streams)
    // -----------------------------------------------------------------------
    float *d_references;
    float *d_cx_ref, *d_cy_ref, *d_cz_ref, *d_G_ref;
    CUDA_CHECK(cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cx_ref,     NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy_ref,     NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz_ref,     NB_FRAMES_PER_CHUNK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_ref,      NB_FRAMES_PER_CHUNK * sizeof(float)));

    // -----------------------------------------------------------------------
    // GPU: per-stream target + result buffers
    // -----------------------------------------------------------------------
    cudaStream_t streams[N_STREAMS];
    float *d_targets[N_STREAMS], *d_rmsd[N_STREAMS];
    float *d_cx_tgt[N_STREAMS],  *d_cy_tgt[N_STREAMS];
    float *d_cz_tgt[N_STREAMS],  *d_G_tgt[N_STREAMS];
    // Pinned host memory: mandatory for cudaMemcpyAsync to overlap with kernel execution.
    float *rmsdHostChunk[N_STREAMS];

    // Per-stream CUDA events for accurate GPU-side timing.
    cudaEvent_t ev_rmsd_start[N_STREAMS], ev_rmsd_stop[N_STREAMS];
    cudaEvent_t ev_cent_start[N_STREAMS], ev_cent_stop[N_STREAMS];

    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_targets[s], NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rmsd[s],    NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cx_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cy_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cz_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_G_tgt[s],   NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&rmsdHostChunk[s],
                                  NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaEventCreate(&ev_rmsd_start[s]));
        CUDA_CHECK(cudaEventCreate(&ev_rmsd_stop[s]));
        CUDA_CHECK(cudaEventCreate(&ev_cent_start[s]));
        CUDA_CHECK(cudaEventCreate(&ev_cent_stop[s]));
    }

    // CUDA events for reference centroid timing (runs on default stream).
    cudaEvent_t ev_ref_cent_start, ev_ref_cent_stop;
    CUDA_CHECK(cudaEventCreate(&ev_ref_cent_start));
    CUDA_CHECK(cudaEventCreate(&ev_ref_cent_stop));

    dim3 threads(32, 8);

    // Accumulated GPU time in milliseconds (measured via CUDA events).
    double centroid_time_ms = 0.0;
    double rmsd_time_ms     = 0.0;
    size_t total_centroid_frames = 0;
    size_t total_rmsd_pairs      = 0;

    // -----------------------------------------------------------------------
    // Debug capture buffer: raw 2D window around the first chunk boundary.
    // -----------------------------------------------------------------------
    const size_t window    = 5;
    const size_t dbg_start = NB_FRAMES_PER_CHUNK - window;
    const size_t dbg_end   = NB_FRAMES_PER_CHUNK + window;
    const size_t dbg_size  = 2 * window;
    std::vector<float> dbg(dbg_size * dbg_size, -1.0f);

    std::vector<float> references_coordinates;

    // -----------------------------------------------------------------------
    // Tile metadata
    // -----------------------------------------------------------------------
    struct TileMeta {
        size_t start_col, stop_col, nb_tgt;
        int    stream_id;
    };

    // -----------------------------------------------------------------------
    // RMSD computation  (multi-stream inner loop)
    // -----------------------------------------------------------------------
    for (size_t row = 0; row < NB_ROW_ITERATIONS; row++)
    {
        size_t start_row = row * NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        size_t nb_ref    = stop_row - start_row;

        std::cout << "Processing row chunk " << row + 1 << "/" << NB_ROW_ITERATIONS
                  << " (" << nb_ref << " frames)\n";

        // Upload reference tile (blocking; happens once per row).
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        CUDA_CHECK(cudaMemcpy(d_references, references_coordinates.data(),
                              references_coordinates.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Centroid for references — default stream, must finish before any RMSD kernel.
        CUDA_CHECK(cudaEventRecord(ev_ref_cent_start, 0));
        computeCentroidsG<<<(nb_ref + 127) / 128, 128>>>(
            d_references, N_atoms, nb_ref,
            d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref);
        CUDA_CHECK(cudaEventRecord(ev_ref_cent_stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        float ref_cent_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ref_cent_ms, ev_ref_cent_start, ev_ref_cent_stop));
        centroid_time_ms += ref_cent_ms;
        total_centroid_frames += nb_ref;

        // ----------------------------------------------------------------
        // Build tile list for this row.
        // ----------------------------------------------------------------
        std::vector<TileMeta> tiles;
        tiles.reserve(NB_ROW_ITERATIONS - row);

        for (size_t col = row; col < NB_ROW_ITERATIONS; col++) {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt    = stop_col - start_col;
            int    s         = (int)((col - row) % N_STREAMS);
            tiles.push_back({start_col, stop_col, nb_tgt, s});
        }

        // ----------------------------------------------------------------
        // FIX: Pre-extract ALL tile coordinates on the CPU before touching
        // any GPU stream.  This eliminates the CPU stall that previously
        // serialised stream launches inside the dispatch loop.
        // ----------------------------------------------------------------
        std::vector<std::vector<float>> all_tile_coords(tiles.size());
        for (size_t ti = 0; ti < tiles.size(); ti++) {
            file.extractSnapshotsFastInPlace(tiles[ti].start_col, tiles[ti].stop_col,
                                             all_data, all_tile_coords[ti]);
        }

        // ----------------------------------------------------------------
        // Dispatch in batches of N_STREAMS.
        // All stream launches within a batch are back-to-back with no CPU
        // work between them, so the GPU can genuinely overlap them.
        // ----------------------------------------------------------------
        for (size_t batch_start = 0; batch_start < tiles.size(); batch_start += N_STREAMS)
        {
            size_t batch_end = std::min(batch_start + (size_t)N_STREAMS, tiles.size());

            // --- Launch phase: queue everything async, no syncs here ---
            for (size_t ti = batch_start; ti < batch_end; ti++)
            {
                const TileMeta& tm = tiles[ti];
                int s              = tm.stream_id;

                // H2D: targets (async, overlaps with kernels on other streams)
                CUDA_CHECK(cudaMemcpyAsync(d_targets[s], all_tile_coords[ti].data(),
                                          tm.nb_tgt * N_atoms * 3 * sizeof(float),
                                          cudaMemcpyHostToDevice, streams[s]));

                // Centroid kernel — record GPU event for real timing
                CUDA_CHECK(cudaEventRecord(ev_cent_start[s], streams[s]));
                computeCentroidsG<<<(tm.nb_tgt + 127) / 128, 128, 0, streams[s]>>>(
                    d_targets[s], N_atoms, tm.nb_tgt,
                    d_cx_tgt[s], d_cy_tgt[s], d_cz_tgt[s], d_G_tgt[s]);
                CUDA_CHECK(cudaEventRecord(ev_cent_stop[s], streams[s]));

                // RMSD kernel
                dim3 blocks((tm.nb_tgt + threads.x - 1) / threads.x,
                            (nb_ref    + threads.y - 1) / threads.y);
                size_t smem_bytes = 3 * threads.x * threads.y * sizeof(float);

                CUDA_CHECK(cudaEventRecord(ev_rmsd_start[s], streams[s]));
                RMSD<<<blocks, threads, smem_bytes, streams[s]>>>(
                    d_references, d_targets[s], N_atoms, nb_ref, tm.nb_tgt,
                    d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref,
                    d_cx_tgt[s], d_cy_tgt[s], d_cz_tgt[s], d_G_tgt[s],
                    d_rmsd[s]);
                CUDA_CHECK(cudaEventRecord(ev_rmsd_stop[s], streams[s]));

                // D2H: results (async, pinned memory)
                CUDA_CHECK(cudaMemcpyAsync(rmsdHostChunk[s], d_rmsd[s],
                                          nb_ref * tm.nb_tgt * sizeof(float),
                                          cudaMemcpyDeviceToHost, streams[s]));

                total_centroid_frames += tm.nb_tgt;
                total_rmsd_pairs      += nb_ref * tm.nb_tgt;
            }

            // --- Sync phase: wait for all streams in this batch ---
            for (int s = 0; s < N_STREAMS; s++)
                CUDA_CHECK(cudaStreamSynchronize(streams[s]));

            // --- Accumulate GPU timing via events (now safe to query) ---
            for (size_t ti = batch_start; ti < batch_end; ti++) {
                int s = tiles[ti].stream_id;
                float cent_ms = 0.0f, rmsd_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&cent_ms, ev_cent_start[s], ev_cent_stop[s]));
                CUDA_CHECK(cudaEventElapsedTime(&rmsd_ms, ev_rmsd_start[s], ev_rmsd_stop[s]));
                centroid_time_ms += cent_ms;
                rmsd_time_ms     += rmsd_ms;
            }

            // --- Pack results into upper-triangle + debug buffer ---
            for (size_t ti = batch_start; ti < batch_end; ti++)
            {
                const TileMeta& tm = tiles[ti];
                int s              = tm.stream_id;

                // Debug window capture
                for (size_t i = 0; i < nb_ref; i++) {
                    size_t gi = start_row + i;
                    if (gi < dbg_start || gi >= dbg_end) continue;
                    for (size_t j = 0; j < tm.nb_tgt; j++) {
                        size_t gj = tm.start_col + j;
                        if (gj < dbg_start || gj >= dbg_end) continue;
                        dbg[(gi - dbg_start) * dbg_size + (gj - dbg_start)] =
                            rmsdHostChunk[s][i * tm.nb_tgt + j];
                    }
                }

                // Pack into upper-triangle (global_i < global_j only)
                for (size_t i = 0; i < nb_ref; i++) {
                    size_t global_i = start_row + i;
                    for (size_t j = 0; j < tm.nb_tgt; j++) {
                        size_t global_j = tm.start_col + j;
                        if (global_i >= global_j) continue;

                        size_t idx = global_i * N_frames
                                     - (global_i * (global_i + 1)) / 2
                                     + (global_j - global_i - 1);

                        rmsdUpperTriangle[idx] = rmsdHostChunk[s][i * tm.nb_tgt + j];
                    }
                }
            }
        }
    }

    double pipeline_time = elapsed_s(global_start);

    // Convert GPU event times from ms to seconds for consistent reporting.
    double centroid_time_s = centroid_time_ms / 1000.0;
    double rmsd_time_s     = rmsd_time_ms     / 1000.0;

    // -----------------------------------------------------------------------
    // Throughput
    // -----------------------------------------------------------------------
    std::cout << "\n===== PERFORMANCE =====\n";
    std::cout << "Centroid compute : " << (double)total_centroid_frames / centroid_time_s
              << " molecules/s  (GPU time: " << centroid_time_s << " s)\n";
    std::cout << "RMSD kernel      : " << (double)total_rmsd_pairs / rmsd_time_s
              << " RMSD/s  (GPU time: " << rmsd_time_s << " s)\n";
    std::cout << "Full pipeline    : " << (double)total_rmsd_pairs / pipeline_time
              << " RMSD/s  (wall time: " << pipeline_time << " s)\n";

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

    auto t_clust = chrono_time::now();
    float db_index = runKMedoids(N_frames, K, rmsdUpperTriangle, MAX_ITER, centroids, clusters);
    double clust_time = elapsed_s(t_clust);

    std::cout << "\n===== K-MEDOIDS =====\n";
    std::cout << "Clustering speed     : " << N_frames / clust_time
              << " molecules/s (" << clust_time << " s)\n";
    std::cout << "Davies-Bouldin index : " << db_index << "\n";

    float random_db = runRandomClustering(N_frames, K, rmsdUpperTriangle);
    std::cout << "Random DB index : " << random_db << "\n";
    std::cout << "Improvement     : " << ((random_db - db_index) / random_db) * 100.0 << "%\n";

    saveClusters(clusters, N_frames, centroids, K);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(d_targets[s]));
        CUDA_CHECK(cudaFree(d_rmsd[s]));
        CUDA_CHECK(cudaFree(d_cx_tgt[s]));
        CUDA_CHECK(cudaFree(d_cy_tgt[s]));
        CUDA_CHECK(cudaFree(d_cz_tgt[s]));
        CUDA_CHECK(cudaFree(d_G_tgt[s]));
        CUDA_CHECK(cudaFreeHost(rmsdHostChunk[s]));
        CUDA_CHECK(cudaEventDestroy(ev_rmsd_start[s]));
        CUDA_CHECK(cudaEventDestroy(ev_rmsd_stop[s]));
        CUDA_CHECK(cudaEventDestroy(ev_cent_start[s]));
        CUDA_CHECK(cudaEventDestroy(ev_cent_stop[s]));
    }

    CUDA_CHECK(cudaEventDestroy(ev_ref_cent_start));
    CUDA_CHECK(cudaEventDestroy(ev_ref_cent_stop));
    CUDA_CHECK(cudaFree(d_references));
    CUDA_CHECK(cudaFree(d_cx_ref)); CUDA_CHECK(cudaFree(d_cy_ref));
    CUDA_CHECK(cudaFree(d_cz_ref)); CUDA_CHECK(cudaFree(d_G_ref));

    delete[] rmsdUpperTriangle;
    delete[] centroids;
    delete[] clusters;

    measure_seconds(global_start, "Total program time");

    return 0;
}