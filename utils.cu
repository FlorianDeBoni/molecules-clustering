#include "utils.cuh"

// ============================================================================
// TILING HELPER FUNCTIONS
// ============================================================================

int trinv(int n) {
    return (int) ( ( 1 + std::sqrt( 1 + ( 8 * n ) ) ) / 2);
}

int triangle_read(int n) {
    return n - (trinv(n)*(trinv(n)-1) / 2);
}

int sum_k(int k) {
    return k * (k+1) / 2;
}

int col_index_parcours(int i, int bound) {
    return (int) (bound - triangle_read( ((bound+1)*(bound+2)/2) - i - 1 ));
}

void update_row_col(size_t idx, const size_t N_CHUNKS_PER_ROW, size_t& row, size_t& col)
{
    // total number of elements in packed upper triangle
    size_t total = N_CHUNKS_PER_ROW * (N_CHUNKS_PER_ROW + 1) / 2;

    if (idx >= total) return; // out of range

    col = static_cast<size_t>((std::sqrt(8.0 * idx + 1) - 1) * 0.5);

    // safety clamp (floating precision)
    if (col >= N_CHUNKS_PER_ROW) col = N_CHUNKS_PER_ROW - 1;

    size_t start = col * (col + 1) / 2;
    row = idx - start;
}

size_t get_chunk_frame_nb(size_t max_cap_MB, size_t N_atoms, size_t N_dims)
{
    // Convert memory capacity to number of floats
    double max_floats =
        static_cast<double>(max_cap_MB) * 1024.0 * 1024.0 / sizeof(float);

    // Real memory model:
    // F² + (2*N_atoms*N_dims + 8)F <= max_floats
    double a = 1.0;
    double b = 2.0 * static_cast<double>(N_atoms) *
               static_cast<double>(N_dims) + 8.0;
    double c = -max_floats;

    double delta = b*b - 4.0*a*c;

    if (delta < 0) {
        std::cerr << "Error: memory too small for even one frame!" << std::endl;
        return 0;
    }

    double F = (-b + std::sqrt(delta)) / (2.0 * a);

    size_t result = static_cast<size_t>(std::floor(F));

    // Align to warp size
    result = (result / 32) * 32;

    return result;
}

size_t get_optimal_tile_size(size_t max_cap_MB, size_t N_atoms, size_t N_dims, size_t N_frames) {
    // Convert memory to floats
    double M = static_cast<double>(max_cap_MB) * 1024.0 * 1024.0 / sizeof(float);

    // Solve quadratic: F^2 + 2*(N_atoms*N_dims)*F - 2*M = 0
    double a = 1.0;
    double b = 2.0 * static_cast<double>(N_atoms) * static_cast<double>(N_dims);
    double c = -2.0 * M;
    double delta = b*b - 4*a*c;
    if (delta < 0) {
        std::cerr << "Error: GPU memory too small!" << std::endl;
        return 0;
    }

    double F_tile = (-b + std::sqrt(delta)) / (2*a);

    // Clip to trajectory length
    F_tile = std::min(F_tile, static_cast<double>(N_frames));

    // Clip to CUDA safe size (16x16 threads per block)
    const double MAX_SAFE_TILE = 65000.0;
    F_tile = std::min(F_tile, MAX_SAFE_TILE);

    return static_cast<size_t>(std::floor(F_tile));
}

void pickKMedoidsPlusPlus(int N_snapshots, int K, const float* rmsd, int* centroids) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist0(0, N_snapshots-1);
    centroids[0] = dist0(gen);

    std::vector<float> minDist(N_snapshots, 1e30f);

    for (int k = 1; k < K; ++k) {
        for (int i = 0; i < N_snapshots; ++i) {
            float d = getRMSD(centroids[k-1], i, rmsd, N_snapshots);
            if (d < minDist[i]) minDist[i] = d * d;
        }

        std::vector<float> cumulative(N_snapshots, 0.0f);
        cumulative[0] = minDist[0];
        for (int i = 1; i < N_snapshots; ++i) cumulative[i] = cumulative[i-1] + minDist[i];

        std::uniform_real_distribution<float> dist(0, cumulative[N_snapshots-1]);
        float r = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        centroids[k] = std::distance(cumulative.begin(), it);
    }
}

void createClusters(int N_frames, int K, const float* rmsd, const int* centroids, int* clusters) {
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

void updateCentroids(int N_frames, int K, const int* clusters, const float* rmsd, int* centroids) {
    for (int k = 0; k < K; k++) {
        float best_cost = 1e30f;
        int best_idx = -1;

        for (int i = 0; i < N_frames; i++) {
            if (clusters[i] != k) continue;

            float cost = 0.0f;
            for (int j = 0; j < N_frames; j++) {
                if (clusters[j] != k) continue;
                cost += getRMSD(i, j, rmsd, N_frames);
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

float daviesBouldinIndex(int N_frames, int K, const int* clusters, const int* centroids, const float* rmsd) {
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

float runKMedoids(int N_frames, int K, const float* rmsd, int MAX_ITER, int* centroids, int* clusters) {
    pickKMedoidsPlusPlus(N_frames, K, rmsd, centroids);
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        createClusters(N_frames, K, rmsd, centroids, clusters);
        
        int* old_centroids = new int[K];
        memcpy(old_centroids, centroids, K * sizeof(int));
        
        updateCentroids(N_frames, K, clusters, rmsd, centroids);
        
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
    
    return daviesBouldinIndex(N_frames, K, clusters, centroids, rmsd);
}

float runRandomClustering(int N_frames, int K, const float* rmsd) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, K-1);

    int* clusters = new int[N_frames];
    for (int i = 0; i < N_frames; i++)
        clusters[i] = dist(gen);

    int* centroids = new int[K];
    for (int k = 0; k < K; k++) {
        centroids[k] = -1;
        for (int i = 0; i < N_frames; i++) {
            if (clusters[i] == k) {
                centroids[k] = i;
                break;
            }
        }
        if (centroids[k] == -1) centroids[k] = 0;
    }

    float db = daviesBouldinIndex(N_frames, K, clusters, centroids, rmsd);

    delete[] clusters;
    delete[] centroids;
    return db;
}

float k_analysis(float* rmsd, size_t N_frames, int MAX_ITER, int K_MIN, int K_MAX) {
    std::cout << "\nVerifying RMSD packed format..." << std::endl;
    std::cout << "Sample values:" << std::endl;
    for (int i = 0; i < std::min(5, (int)N_frames); i++) {
        for (int j = i+1; j < std::min(5, (int)N_frames); j++) {
            float val = getRMSD(i, j, rmsd, N_frames);
            std::cout << "  RMSD(" << i << "," << j << ") = " << val << std::endl;
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "DAVIES-BOULDIN INDEX ANALYSIS (K = " << K_MIN 
              << " to " << K_MAX << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::vector<int> K_values;
    std::vector<float> db_kmedoids;
    std::vector<float> db_random;
    
    std::cout << "\nK\tDB_KMedoids\tDB_Random\tDifference\tConvergence" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int K = K_MIN; K <= K_MAX; K++) {
        int* clusters = new int[N_frames];
        int* centroids = new int[K];

        std::cout << K << "\t" << std::flush;
        
        float db_km = runKMedoids(N_frames, K, rmsd, MAX_ITER, centroids, clusters);
        std::cout << std::fixed << std::setprecision(4) << db_km << "\t" << std::flush;
        
        float db_rand = runRandomClustering(N_frames, K, rmsd);
        std::cout << std::fixed << std::setprecision(4) << db_rand << "\t" << std::flush;
        
        float diff = db_rand - db_km;
        std::cout << std::fixed << std::setprecision(4) << diff << "\t" << std::flush;
        std::cout << (diff > 0 ? "✓ Better" : "✗ Worse") << std::endl;
        
        K_values.push_back(K);
        db_kmedoids.push_back(db_km);
        db_random.push_back(db_rand);

        delete[] clusters;
        delete[] centroids;
    }
    
    std::cout << std::string(70, '=') << std::endl;
    
    std::ofstream csv_out("output/db_index_vs_K.csv");
    csv_out << "K,DB_KMedoids,DB_Random,Difference\n";
    for (size_t i = 0; i < K_values.size(); i++) {
        csv_out << K_values[i] << "," 
                << db_kmedoids[i] << "," 
                << db_random[i] << ","
                << (db_random[i] - db_kmedoids[i]) << "\n";
    }
    csv_out.close();
    
    std::cout << "\n✓ Results saved to output/db_index_vs_K.csv" << std::endl;
    
    auto min_it = std::min_element(db_kmedoids.begin(), db_kmedoids.end());
    int optimal_K_db = K_values[std::distance(db_kmedoids.begin(), min_it)];
    float optimal_db = *min_it;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "OPTIMAL K ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Optimal K (lowest DB index): " << optimal_K_db << std::endl;
    std::cout << "DB Index at optimal K: " << optimal_db << std::endl;
    
    int better_count = 0;
    for (size_t i = 0; i < K_values.size(); i++) {
        if (db_kmedoids[i] < db_random[i]) better_count++;
    }
    
    float percent_better = 100.0f * better_count / K_values.size();
    std::cout << "\nK-medoids beats random: " << better_count << "/" 
              << K_values.size() << " times (" << percent_better << "%)" << std::endl;
    
    if (percent_better < 50) {
        std::cout << "\n⚠️  WARNING: K-medoids rarely beats random clustering!" << std::endl;
    }
    
    int* final_centroids = new int[optimal_K_db];
    int* final_clusters = new int[N_frames];
    
    float final_db = runKMedoids(N_frames, optimal_K_db, rmsd, MAX_ITER, 
                                  final_centroids, final_clusters);
    
    std::vector<int> cluster_sizes(optimal_K_db, 0);
    for (int i = 0; i < N_frames; i++) {
        cluster_sizes[final_clusters[i]]++;
    }
    
    std::ofstream cent_out("output/optimal_centroids.txt");
    cent_out << "# Optimal K = " << optimal_K_db << "\n";
    cent_out << "# Davies-Bouldin Index = " << final_db << "\n";
    cent_out << "# Cluster\tFrame_Index\tSize\n";
    for (int k = 0; k < optimal_K_db; k++) {
        cent_out << k << "\t" << final_centroids[k] << "\t" 
                 << cluster_sizes[k] << "\n";
    }
    cent_out.close();

    delete[] final_centroids;
    delete[] final_clusters;

    return final_db;
}

void saveClusters(const int* clusters, int N_frames, const int* centroids, int K) {
    std::ofstream out("output/cluster_assignments.txt");
    out << "# Frame\tCluster\n";
    for (int i = 0; i < N_frames; i++) {
        out << i << "\t" << clusters[i] << "\n";
    }
    out.close();
    std::cout << "✓ Cluster assignments saved to output/cluster_assignments.txt" << std::endl;
}
