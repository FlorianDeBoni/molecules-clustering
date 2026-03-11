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

static inline double elapsed_s(const chrono_type& start) {
    return std::chrono::duration<double>(chrono_time::now() - start).count();
}

int main(int argc, char** args) {

    chrono_type global_start = chrono_time::now();

    std::string file_name;
    if (argc >= 2) file_name = args[1];
    else {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    FileUtils file(file_name);

    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::vector<float> all_data(N_frames * N_atoms * 3);

    chrono_type t_read = chrono_time::now();
    file.readSnapshotsFastInPlace(0, N_frames - 1, all_data);
    double read_time = elapsed_s(t_read);

    std::cout << "Read dataset : "
              << N_frames/read_time
              << " molecules/s\n";

    const size_t MAX_DATA_CHUNK_SIZE  = 12000;
    const size_t NB_FRAMES_PER_CHUNK  =
        get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);

    const size_t NB_ROW_ITERATIONS =
        (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

    size_t rmsd_chunk_size = NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK;
    float* rmsdHostChunk   = new float[rmsd_chunk_size];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    float* d_references=nullptr;
    float* d_targets=nullptr;
    float* d_rmsd=nullptr;

    cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_targets,    NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_rmsd,       NB_FRAMES_PER_CHUNK*NB_FRAMES_PER_CHUNK*sizeof(float));

    // centroid / G buffers
    float *d_cx_ref,*d_cy_ref,*d_cz_ref,*d_G_ref;
    float *d_cx_tgt,*d_cy_tgt,*d_cz_tgt,*d_G_tgt;

    cudaMalloc(&d_cx_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_ref ,NB_FRAMES_PER_CHUNK*sizeof(float));

    cudaMalloc(&d_cx_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_tgt ,NB_FRAMES_PER_CHUNK*sizeof(float));

    dim3 threads(32,8);

    // CUDA timing
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    float centroid_time_ms = 0.0f;
    float rmsd_time_ms     = 0.0f;

    size_t total_rmsd_pairs = 0;
    size_t total_molecules_processed = 0;

    chrono_type rmsd_pipeline_start = chrono_time::now();

    for (size_t row=0; row<NB_ROW_ITERATIONS; row++) {

        size_t start_row=row*NB_FRAMES_PER_CHUNK;
        size_t stop_row =std::min(start_row+NB_FRAMES_PER_CHUNK,N_frames);
        size_t nb_ref   =stop_row-start_row;

        total_molecules_processed += nb_ref;

        file.extractSnapshotsFastInPlace(
            start_row,stop_row,all_data,references_coordinates);

        cudaMemcpy(d_references,references_coordinates.data(),
                   references_coordinates.size()*sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads1D=128;
        int blocks_ref=(nb_ref+threads1D-1)/threads1D;

        cudaEventRecord(start_evt);

        computeCentroidsG<<<blocks_ref,threads1D>>>(
            d_references,N_atoms,nb_ref,
            d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref);

        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);

        float ms;
        cudaEventElapsedTime(&ms,start_evt,stop_evt);
        centroid_time_ms += ms;

        for(size_t col=row; col<NB_ROW_ITERATIONS; col++) {

            size_t start_col=col*NB_FRAMES_PER_CHUNK;
            size_t stop_col =std::min(start_col+NB_FRAMES_PER_CHUNK,N_frames);
            size_t nb_tgt   =stop_col-start_col;

            total_molecules_processed += nb_tgt;

            file.extractSnapshotsFastInPlace(
                start_col,stop_col,all_data,targets_coordinates);

            cudaMemcpy(d_targets,targets_coordinates.data(),
                       targets_coordinates.size()*sizeof(float),
                       cudaMemcpyHostToDevice);

            int blocks_tgt=(nb_tgt+threads1D-1)/threads1D;

            cudaEventRecord(start_evt);

            computeCentroidsG<<<blocks_tgt,threads1D>>>(
                d_targets,N_atoms,nb_tgt,
                d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt);

            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);

            cudaEventElapsedTime(&ms,start_evt,stop_evt);
            centroid_time_ms += ms;

            dim3 blocks((nb_tgt+threads.x-1)/threads.x,
                        (nb_ref+threads.y-1)/threads.y);

            cudaEventRecord(start_evt);

            RMSD_precomputed<<<blocks,threads>>>(
                d_references,d_targets,
                N_atoms,nb_ref,nb_tgt,
                d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref,
                d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt,
                d_rmsd);

            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);

            cudaEventElapsedTime(&ms,start_evt,stop_evt);
            rmsd_time_ms += ms;

            total_rmsd_pairs += (size_t)nb_ref * (size_t)nb_tgt;

            cudaMemcpy(rmsdHostChunk,d_rmsd,
                       nb_ref*nb_tgt*sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

    double total_pipeline_time = elapsed_s(rmsd_pipeline_start);

    double centroid_time_s = centroid_time_ms / 1000.0;
    double rmsd_time_s     = rmsd_time_ms / 1000.0;

    std::cout << "\n===== PERFORMANCE =====\n";

    std::cout << "Centroid+G compute : "
              << (double)total_molecules_processed / centroid_time_s
              << " molecules/s ("
              << centroid_time_s
              << "s)\n";

    std::cout << "RMSD kernel        : "
              << (double)total_rmsd_pairs / rmsd_time_s
              << " RMSD/s ("
              << rmsd_time_s
              << "s)\n";

    std::cout << "Full RMSD pipeline : "
              << (double)total_rmsd_pairs / total_pipeline_time
              << " RMSD/s ("
              << total_pipeline_time
              << "s)\n";

    std::cout << "Total RMSD pairs   : "
              << total_rmsd_pairs << "\n";

    cudaFree(d_references);
    cudaFree(d_targets);
    cudaFree(d_rmsd);

    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdHostChunk;

    measure_seconds(global_start,"Total program time");

    return 0;
}