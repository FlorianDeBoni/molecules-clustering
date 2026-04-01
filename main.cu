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
#include <algorithm>

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
    size_t N_frames = 90000;
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
    const size_t window    = 5;
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

        // extractSnapshotsFastInPlace uses exclusive [start, end) convention.
        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        CUDA_CHECK(cudaMemcpy(d_references, references_coordinates.data(),
                              references_coordinates.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        for(size_t col = row; col < NB_ROW_ITERATIONS; col++)
        {
            size_t start_col = col * NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt    = stop_col - start_col;

            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
            CUDA_CHECK(cudaMemcpy(d_targets, targets_coordinates.data(),
                                  targets_coordinates.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));

            dim3 blocks((nb_tgt + threads.x - 1) / threads.x,
                        (nb_ref + threads.y - 1) / threads.y);

            // TILE matches blockDim.x (= threads.x = 32) as used inside the kernel.
            const int TILE      = threads.x;
            size_t smem_bytes   = 3 * TILE * threads.y * sizeof(float);

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
    int K        = 10;
    int MAX_ITER = 50;


    // Allocating GPU arrays
    int* centroids = new int[K];
    int* d_centroids;
    CUDA_CHECK(cudaMalloc(&d_centroids, K * sizeof(int)));

    int* clusters  = new int[N_frames];
    int* d_clusters;
    CUDA_CHECK(cudaMalloc(&d_clusters, N_frames * sizeof(int)));

    const int CHUNK_SIZE = 1024; // tune to fit VRAM: CHUNK_SIZE * N_frames * 4 bytes

    float* d_centroid_rows;
    CUDA_CHECK(cudaMalloc(&d_centroid_rows, (size_t)K * N_frames * sizeof(float)));

    float* d_chunk;
    CUDA_CHECK(cudaMalloc(&d_chunk, (size_t)CHUNK_SIZE * N_frames * sizeof(float)));

    float* h_chunk_pinned;
    CUDA_CHECK(cudaMallocHost(&h_chunk_pinned, (size_t)CHUNK_SIZE * N_frames * sizeof(float)));

    float* d_frame_costs;
    CUDA_CHECK(cudaMalloc(&d_frame_costs, N_frames * sizeof(float)));

    cudaStream_t compute_stream, transfer_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));


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

        // 1. Upload K centroid rows (small, sync is fine)
        uploadCentroidRows(N_frames, K, rmsdUpperTriangle, centroids, d_centroid_rows, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // 2. Assign each frame to nearest centroid
        AssignClusters<<<gridDim, blockDim, 0, compute_stream>>>(
            N_frames, K, d_centroid_rows, d_clusters
        );
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // 3. Compute medoid costs in chunks
        for (int chunk_start = 0; chunk_start < N_frames; chunk_start += CHUNK_SIZE) {
            int chunk_size = std::min((size_t) CHUNK_SIZE, N_frames - chunk_start);
            bool reset     = (chunk_start == 0);

            // Upload chunk on transfer stream
            uploadChunk(
                N_frames, chunk_start, chunk_size,
                rmsdUpperTriangle, d_chunk, h_chunk_pinned,
                transfer_stream
            );
            CUDA_CHECK(cudaStreamSynchronize(transfer_stream)); // chunk ready before kernel

            ComputeMedoidCosts_Chunk<<<gridDim, blockDim, 0, compute_stream>>>(
                N_frames, chunk_start, chunk_size,
                d_chunk, d_clusters, d_frame_costs, reset
            );
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }

        // 4. Update medoids
        UpdateMedoids<<<reducingBlocks, blockDim, sharedMemSize, compute_stream>>>(
            N_frames, d_centroids, d_clusters, d_frame_costs
        );
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // 5. Pull centroids back to CPU (needed for uploadCentroidRows next iter)
        CUDA_CHECK(cudaMemcpyAsync(
            centroids, d_centroids, K * sizeof(int),
            cudaMemcpyDeviceToHost, compute_stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_centroid_rows));
    CUDA_CHECK(cudaFree(d_chunk));
    CUDA_CHECK(cudaFreeHost(h_chunk_pinned));
    CUDA_CHECK(cudaFree(d_frame_costs));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(transfer_stream));

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

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    // CUDA_CHECK(cudaFree(d_rmsdUpperTriangle));
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
