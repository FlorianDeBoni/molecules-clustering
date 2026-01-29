#include "FileUtils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <iomanip>

float getRMSD(int i, int j, const float* rmsdHost, int N_frames) {
    if (i == j) return 0.0f;
    if (i > j) std::swap(i, j);
    size_t idx = (size_t)i * N_frames 
               - ((size_t)i * ((size_t)i + 1)) / 2 
               + (j - i - 1);
    return rmsdHost[idx];
}

void pickKMedoidsPlusPlus(int N_frames, int K, const float* rmsd, int* centroids) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist0(0, N_frames-1);
    centroids[0] = dist0(gen);

    std::vector<float> minDist(N_frames, 1e30f);

    for (int k = 1; k < K; ++k) {
        for (int i = 0; i < N_frames; ++i) {
            float d = getRMSD(centroids[k-1], i, rmsd, N_frames);
            if (d < minDist[i]) minDist[i] = d * d;
        }

        // Compute cumulative probability
        std::vector<float> cumulative(N_frames, 0.0f);
        cumulative[0] = minDist[0];
        for (int i = 1; i < N_frames; ++i) cumulative[i] = cumulative[i-1] + minDist[i];

        std::uniform_real_distribution<float> dist(0, cumulative[N_frames-1]);
        float r = dist(gen);

        // Pick next centroid
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        centroids[k] = std::distance(cumulative.begin(), it);
    }
}

void createClusters(
    int N_frames,
    int K,
    const float* rmsd,
    const int* centroids,
    int* clusters
) {
    for (int i = 0; i < N_frames; i++) {
        float best = 1e30f;
        int best_k = -1;

        for (int k = 0; k < K; k++) {
            float d = getRMSD(centroids[k], i, rmsd, N_frames);
            if (d < best) {
                best = d;
                best_k = k;
            }
        }
        clusters[i] = best_k;
    }
}

void updateCentroids(
    int N_frames,
    int K,
    const int* clusters,
    const float* rmsdHost,
    int* centroids
) {
    for (int k = 0; k < K; k++) {
        float best_cost = 1e30f;
        int best_idx = -1;

        for (int i = 0; i < N_frames; i++) {
            if (clusters[i] != k) continue;

            float cost = 0.0f;
            for (int j = 0; j < N_frames; j++) {
                if (clusters[j] != k) continue;
                cost += getRMSD(i, j, rmsdHost, N_frames);
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
        }

        if (best_idx != -1)
            centroids[k] = best_idx;
    }
}

float daviesBouldinIndex(
    int N_frames,
    int K,
    const int* clusters,
    const int* centroids,
    const float* rmsd
) {
    std::vector<float> S(K, 0.0f);
    std::vector<int> counts(K, 0);

    for (int i = 0; i < N_frames; i++) {
        int k = clusters[i];
        S[k] += getRMSD(centroids[k], i, rmsd, N_frames);
        counts[k]++;
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0)
            S[k] /= counts[k];
    }

    float db = 0.0f;

    for (int i = 0; i < K; i++) {
        float maxR = 0.0f;

        for (int j = 0; j < K; j++) {
            if (i == j) continue;

            float Mij = getRMSD(centroids[i], centroids[j], rmsd, N_frames);
            if (Mij > 0.0f) {
                float Rij = (S[i] + S[j]) / Mij;
                maxR = std::max(maxR, Rij);
            }
        }

        db += maxR;
    }

    return db / K;
}

// Run K-medoids clustering and return DB index
float runKMedoidsInit(int N_frames, int K, const float* rmsdHost,
                      int MAX_ITER,
                      const int* init_centroids,
                      int* final_centroids,
                      int* final_clusters)
{
    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    memcpy(centroids, init_centroids, K * sizeof(int));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        createClusters(N_frames, K, rmsdHost, centroids, clusters);

        int* old_centroids = new int[K];
        memcpy(old_centroids, centroids, K * sizeof(int));

        updateCentroids(N_frames, K, clusters, rmsdHost, centroids);

        bool converged = true;
        for (int k = 0; k < K; k++) {
            if (centroids[k] != old_centroids[k]) {
                converged = false;
                break;
            }
        }

        delete[] old_centroids;
        if (converged) break;
    }

    float db = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdHost);

    memcpy(final_centroids, centroids, K * sizeof(int));
    memcpy(final_clusters,  clusters,  N_frames * sizeof(int));

    delete[] centroids;
    delete[] clusters;

    return db;
}


// Run random clustering and return DB index
float runRandomClusterAssignment(int N_frames, int K, const float* rmsdHost)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, K-1);

    int* clusters = new int[N_frames];
    for (int i = 0; i < N_frames; i++)
        clusters[i] = dist(gen);  // assign frame i to a random cluster

    // Compute a “dummy” centroid for DB index: pick any frame in each cluster
    int* centroids = new int[K];
    for (int k = 0; k < K; k++)
    {
        centroids[k] = -1;
        for (int i = 0; i < N_frames; i++)
        {
            if (clusters[i] == k)
            {
                centroids[k] = i;
                break;
            }
        }
        if (centroids[k] == -1) centroids[k] = 0; // fallback if cluster is empty
    }

    float db = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsdHost);

    delete[] clusters;
    delete[] centroids;
    return db;
}