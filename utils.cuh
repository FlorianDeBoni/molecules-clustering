#pragma once

#include "FileUtils.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <stdio.h>
#include <iomanip>
#include <cmath>


// ============================================================================
// UPPER TRIANGLE INDEXING HELPER (inline - can be in header)
// ============================================================================
inline float getRMSD(int i, int j, const float* rmsdPacked, int N_snapshots) {
    if (i == j) return 0.0f;
    if (i > j) std::swap(i, j);
    size_t idx = (size_t)i * N_snapshots
           - ((size_t)i * ((size_t)i + 1)) / 2
           + (j - i - 1);
    return rmsdPacked[idx];
}

// ============================================================================
// FUNCTION DECLARATIONS (implementations in utils.cu)
// ============================================================================

// Tiling helper functions
int trinv(int n);
int triangle_read(int n);
int sum_k(int k);
int col_index_parcours(int i, int bound);
void update_row_col(size_t idx, const size_t N_CHUNKS_PER_ROW, size_t& row, size_t& col);
size_t get_chunk_frame_nb(size_t max_cap, size_t N_atoms, size_t N_dims);

// K-medoids functions
void pickRandomCentroids(int N_frames, int K, int* centroids);
void pickKMedoidsPlusPlus(int N_snapshots, int K, const float* rmsd, int* centroids);
void createClusters(int N_frames, int K, const float* rmsd, const int* centroids, int* clusters);
void updateCentroids(int N_frames, int K, const int* clusters, const float* rmsd, int* centroids);
float daviesBouldinIndex(int N_frames, int K, const int* clusters, const int* centroids, const float* rmsd);
float runKMedoids(int N_frames, int K, const float* rmsd, int MAX_ITER, int* centroids, int* clusters);
float runRandomClustering(int N_frames, int K, const float* rmsd);
float k_analysis(float* rmsd, size_t N_frames, int MAX_ITER, int K_MIN = 2, int K_MAX = 50);
void saveClusters(const int* clusters, int N_frames, const int* centroids, int K);


void saveArrayToFile(const char* filename, float* array, size_t size);