#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
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
    // GPU: shared reference buffers (one copy, read by all streams)
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
    // Pinned host memory is required for async D2H copies to overlap with kernels.
    float *rmsdHostChunk[N_STREAMS];

    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_targets[s], NB_FRAMES_PER_CHUNK * N_atoms * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rmsd[s],    NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cx_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cy_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cz_tgt[s],  NB_FRAMES_PER_CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_G_tgt[s],   NB_FRAMES_PER_CHUNK * sizeof(float)));
        // cudaMallocHost = page-locked; mandatory for cudaMemcpyAsync to actually overlap
        CUDA_CHECK(cudaMallocHost(&rmsdHostChunk[s],
                                  NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK * sizeof(float)));
    }

    dim3 threads(32, 8);
    double centroid_time = 0.0;
    double rmsd_time     = 0.0;
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

    // Pre-extracted coordinate buffers, one slot per stream so we can
    // pre-fetch the next tile while the GPU processes the current one.
    std::vector<std::vector<float>> targets_coords(N_STREAMS);
    std::vector<float> references_coordinates;

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

        // Upload reference tile (blocking; happens once per row)
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        CUDA_CHECK(cudaMemcpy(d_references, references_coordinates.data(),
                              references_coordinates.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Centroid for references (default stream; must finish before any RMSD kernel)
        auto c0 = chrono_time::now();
        computeCentroidsG<<<(nb_ref + 127) / 128, 128>>>(
            d_references, N_atoms, nb_ref,
            d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref);
        CUDA_CHECK(cudaDeviceSynchronize());
        centroid_time += elapsed_s(c0);
        total_centroid_frames += nb_ref;

        // ----------------------------------------------------------------
        // Collect metadata for every col tile in this row, then dispatch
        // them round-robin across streams.
        // ----------------------------------------------------------------
        struct TileMeta {
            size_t start_col, stop_col, nb_tgt;
            int    stream_id;
        };
        std::vector<TileMeta> tiles;

        size_t nb_col_tiles = NB_ROW_ITERATIONS - row;
        tiles.reserve(nb_col_tiles);

        for (size_t col = row; col < NB_ROW_ITERATIONS; col++) {
            int    s         = (int)((col - row) % N_STREAMS);
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt    = stop_col - start_col;

            tiles.push_back({start_col, stop_col, nb_tgt, s});
        }

        // ----------------------------------------------------------------
        // Dispatch: for every tile assigned to stream s, we must wait
        // until any previous tile on that same stream has finished its D2H
        // copy before we overwrite d_targets[s].  We handle this by
        // processing tiles in batches of N_STREAMS and syncing between
        // batches.
        // ----------------------------------------------------------------
        auto centroid_target_time_acc = 0.0;
        auto rmsd_kernel_time_acc     = 0.0;

        for (size_t batch_start = 0; batch_start < tiles.size(); batch_start += N_STREAMS)
        {
            size_t batch_end = std::min(batch_start + (size_t)N_STREAMS, tiles.size());

            // Launch all tiles in this batch
            for (size_t ti = batch_start; ti < batch_end; ti++)
            {
                const TileMeta& tm = tiles[ti];
                int s              = tm.stream_id;

                // Async H2D: targets
                file.extractSnapshotsFastInPlace(tm.start_col, tm.stop_col,
                                                 all_data, targets_coords[s]);
                CUDA_CHECK(cudaMemcpyAsync(d_targets[s], targets_coords[s].data(),
                                          tm.nb_tgt * N_atoms * 3 * sizeof(float),
                                          cudaMemcpyHostToDevice, streams[s]));

                // Centroid for targets (on stream)
                auto c2 = chrono_time::now();
                computeCentroidsG<<<(tm.nb_tgt + 127) / 128, 128, 0, streams[s]>>>(
                    d_targets[s], N_atoms, tm.nb_tgt,
                    d_cx_tgt[s], d_cy_tgt[s], d_cz_tgt[s], d_G_tgt[s]);
                centroid_target_time_acc += elapsed_s(c2);
                total_centroid_frames += tm.nb_tgt;

                // RMSD kernel (on stream)
                dim3 blocks((tm.nb_tgt + threads.x - 1) / threads.x,
                            (nb_ref    + threads.y - 1) / threads.y);
                size_t smem_bytes = 3 * threads.x * threads.y * sizeof(float);

                auto k0 = chrono_time::now();
                RMSD<<<blocks, threads, smem_bytes, streams[s]>>>(
                    d_references, d_targets[s], N_atoms, nb_ref, tm.nb_tgt,
                    d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref,
                    d_cx_tgt[s], d_cy_tgt[s], d_cz_tgt[s], d_G_tgt[s],
                    d_rmsd[s]);
                rmsd_kernel_time_acc += elapsed_s(k0);
                total_rmsd_pairs += nb_ref * tm.nb_tgt;

                // Async D2H: results (pinned memory required)
                CUDA_CHECK(cudaMemcpyAsync(rmsdHostChunk[s], d_rmsd[s],
                                          nb_ref * tm.nb_tgt * sizeof(float),
                                          cudaMemcpyDeviceToHost, streams[s]));
            }

            // Sync all streams before reading host results for this batch
            for (int s = 0; s < N_STREAMS; s++)
                CUDA_CHECK(cudaStreamSynchronize(streams[s]));

            // Pack results into upper-triangle storage
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

        centroid_time += centroid_target_time_acc;
        rmsd_time     += rmsd_kernel_time_acc;
    }

    double pipeline_time = elapsed_s(global_start);

    // -----------------------------------------------------------------------
    // Throughput
    // -----------------------------------------------------------------------
    std::cout << "\n===== PERFORMANCE =====\n";
    std::cout << "Centroid compute : " << total_centroid_frames / centroid_time
              << " molecules/s (" << centroid_time << " s)\n";
    std::cout << "RMSD kernel      : " << total_rmsd_pairs / rmsd_time
              << " RMSD/s (" << rmsd_time << " s)\n";
    std::cout << "Full pipeline    : " << total_rmsd_pairs / pipeline_time
              << " RMSD/s (" << pipeline_time << " s)\n";

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
    }

    CUDA_CHECK(cudaFree(d_references));
    CUDA_CHECK(cudaFree(d_cx_ref)); CUDA_CHECK(cudaFree(d_cy_ref));
    CUDA_CHECK(cudaFree(d_cz_ref)); CUDA_CHECK(cudaFree(d_G_ref));

    delete[] rmsdUpperTriangle;
    delete[] centroids;
    delete[] clusters;

    measure_seconds(global_start, "Total program time");

    return 0;
}