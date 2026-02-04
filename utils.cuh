#pragma once

#include "FileUtils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <iomanip>

void pickKMedoidsPlusPlus(int N_snapshots, int K, const float* rmsd, int* centroids);

void createClusters(
    int N_snapshots,
    int K,
    const float* rmsd,
    const int* centroids,
    int* clusters
);

void updateCentroids(
    int N_snapshots,
    int K,
    const int* clusters,
    const float* rmsdHost,
    int* centroids
);

float daviesBouldinIndex(
    int N_snapshots,
    int K,
    const int* clusters,
    const int* centroids,
    const float* rmsd
);

// Run K-medoids clustering and return DB index
float runKMedoidsInit(int N_snapshots, int K, const float* rmsdHost,
                      int MAX_ITER,
                      const int* init_centroids,
                      int* final_centroids,
                      int* final_clusters);

// Run random clustering and return DB index
float runRandomClusterAssignment(int N_snapshots, int K, const float* rmsdHost);

__host__ __device__
inline int upper_triangle_index(int i, int j, int N) {
    return i * N - (i*(i+1))/2 + (j - i - 1);
}