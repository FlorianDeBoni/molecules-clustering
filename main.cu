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
    print_throughput("Read .bin", elapsed_s(t_read), N_frames);

    const size_t MAX_DATA_CHUNK_SIZE  = 12000;
    const size_t NB_FRAMES_PER_CHUNK  = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS    = (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

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

    for (size_t row=0; row<NB_ROW_ITERATIONS; row++) {

        size_t start_row=row*NB_FRAMES_PER_CHUNK;
        size_t stop_row =std::min(start_row+NB_FRAMES_PER_CHUNK,N_frames);
        size_t nb_ref   =stop_row-start_row;

        file.extractSnapshotsFastInPlace(start_row,stop_row,all_data,references_coordinates);

        cudaMemcpy(d_references,references_coordinates.data(),
                   references_coordinates.size()*sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads1D=128;
        int blocks_ref=(nb_ref+threads1D-1)/threads1D;

        computeCentroidsG<<<blocks_ref,threads1D>>>(
            d_references,N_atoms,nb_ref,
            d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref);

        for(size_t col=row; col<NB_ROW_ITERATIONS; col++) {

            size_t start_col=col*NB_FRAMES_PER_CHUNK;
            size_t stop_col =std::min(start_col+NB_FRAMES_PER_CHUNK,N_frames);
            size_t nb_tgt   =stop_col-start_col;

            file.extractSnapshotsFastInPlace(start_col,stop_col,all_data,targets_coordinates);

            cudaMemcpy(d_targets,targets_coordinates.data(),
                       targets_coordinates.size()*sizeof(float),
                       cudaMemcpyHostToDevice);

            int blocks_tgt=(nb_tgt+threads1D-1)/threads1D;

            computeCentroidsG<<<blocks_tgt,threads1D>>>(
                d_targets,N_atoms,nb_tgt,
                d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt);

            dim3 blocks((nb_tgt+threads.x-1)/threads.x,
                        (nb_ref+threads.y-1)/threads.y);

            RMSD_precomputed<<<blocks,threads>>>(
                d_references,d_targets,
                N_atoms,nb_ref,nb_tgt,
                d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref,
                d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt,
                d_rmsd);

            cudaMemcpy(rmsdHostChunk,d_rmsd,
                       nb_ref*nb_tgt*sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(d_references);
    cudaFree(d_targets);
    cudaFree(d_rmsd);

    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdHostChunk;

    measure_seconds(global_start,"Total program time");

    return 0;
}