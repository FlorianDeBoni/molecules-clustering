#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <omp.h>

#include <functional>
#include <limits>
#include <algorithm>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include "FileUtils.h"
#include "RMSD_func.h"


#define M_PI 3.14159265358979323846

// Pour compilation :
// cd "C:\Users\Paul\Documents\Projet3A\Code\molecules-clustering"
// g++ -fopenmp cpu_clustering.cpp RMSD_func.cpp -o cpu_clustering

using atome_tab = std::array< double , 3 >;
// de taille 3 * N
using molecule_tab = std::vector< std::vector< double > >;
using snapshot_mod = std::vector < molecule_tab >;
using rot_matrix = std::vector< std::vector< double > >;
using matrix_mod = std::vector< std::vector< double > >;

#define M_PI 3.14159265358979323846


// Fonction qui choisit un index proportionnellement à un certain poids
int weighted_choice(const std::vector<double>& weights, std::mt19937_64& rng) {
    double total = 0.0;
    for (double w : weights) total += w;
    if (total <= 0.0) {
        std::uniform_int_distribution<size_t> uni(0, weights.size() - 1);
        return uni(rng);
    }

    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);

    double c = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        c += weights[i];
        if (r <= c) return i;
    }
    return weights.size() - 1;
}


// Initialisation de la methode kmeans, on crée ici les centres
// Le premier centre est choisi aléatoirement parmi les snapshots existantes, puis les autres sont progressivement ajoutés avec une probabilité
// proportionnelle à leur éloignement de l'ensemble des centres existants
std::vector<int> init_kmeanspp_centers(const snapshot_mod& snaps, int K, std::mt19937_64& rng) {
    const int N = snaps.size();
    const int N_atome = (int)snaps[0][0].size();

    if (K <= 0 || (size_t)K > N) throw std::runtime_error("Invalid K");

    std::vector<int> centers;
    centers.reserve(K);

    // Centre choisi aléatoirement
    std::uniform_int_distribution<size_t> uni(0, N - 1);
    centers.push_back(uni(rng));

    // Vecteur qui garde en mémoire pour chaque snapshot à quel point elle est proche des centres, le centre dont elle est le plus proche
    std::vector<double> bestD(N, std::numeric_limits<double>::infinity());

    for (int c = 1; c < K; c++) {
        int lastCenter = centers.back();

        // PARALLELISATION
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; i++) {
            double d = calcul_RMSD(snaps[i], snaps[lastCenter], N_atome);
            if (d < bestD[i]) {
                bestD[i] = d;
            };
        }

        // Calcul des poids
        std::vector<double> weights(N);
        for (int i = 0; i < N; i++) {
            double d = bestD[i];
            weights[i] = d * d;
        }

        // Poids mis à zéro pour éviter de sélectionner 2 fois le même centre
        for (size_t idx : centers) weights[idx] = 0.0;

        int next = weighted_choice(weights, rng);
        centers.push_back(next);
    }

    return centers;
}

// Fonction qui à chaque snapshot lui associe son centre le plus proche, son cluster
std::vector<int> assign_clusters(const snapshot_mod& snaps, const std::vector<int>& centers) {
    const int N = snaps.size();
    const int N_atome = (int)snaps[0][0].size();
    const int K = (int)centers.size();
    std::vector<int> labels(N, -1);
    
    // PARALLELISATION
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < N; i++) {
        double best = std::numeric_limits<double>::infinity();
        int bestk = -1;
        for (int k = 0; k < K; k++) {
            double d = calcul_RMSD(snaps[i], snaps[centers[k]], N_atome);
            if (d < best) {
                best = d;
                bestk = k;
            }
        }
        labels[i] = bestk;
    }
    return labels;
}


// Ici, pour chaque cluster, on choisit aléatoirement num_test_points snapshots appellées candidats.
// Pour chaque candidat, on fait la somme de tous les rmsd entre le candidat et les snapshots de son cluster
// Le nouveau centre devient alors celui qui a la somme la plus faible parmi ces candidats.
std::vector<int> update_centers_by_subsample_medoid(const snapshot_mod& snaps, const std::vector<int>& labels, int K, int num_test_points, 
    std::mt19937_64& rng, const std::vector<int>& fallback_centers) {

    const int N_atome = (int)snaps[0][0].size();

    // On crée un vecteur qui nous donne les membres pour chaque cluster
    std::vector<std::vector<int>> members(K);
    for (int i = 0; i < snaps.size(); i++) {
        int k = labels[i];
        if (k >= 0 && k < K) members[k].push_back(i);
    }

    std::vector<int> new_centers(K, 0);

    // PARALLELISATION
    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < K; k++) {
        auto& M = members[k];
        if (M.empty()) {
            // Empty cluster: keep previous center (simple fallback)
            new_centers[k] = fallback_centers[k];
            continue;
        }

        int m = (int)M.size();
        int t = std::min(num_test_points, m);
        // int t = std::min((int)m/3, m);

        // sample t distinct indices from M
        // simple: shuffle and take first t
        std::vector<int> candidates = M;
        std::shuffle(candidates.begin(), candidates.end(), rng);
        candidates.resize(t);
        // On inclut aussi dans les candidats le centre actuel pour améliorer la stabilité
        // candidates[0] = fallback_centers[k];

        double bestScore = std::numeric_limits<double>::infinity();
        int bestIdx = candidates[0];

        for (int cand : candidates) {
            double score = 0.0;
            for (int j : M) {
                score += calcul_RMSD(snaps[cand], snaps[j], N_atome);
            }
            if (score < bestScore) {
                bestScore = score;
                bestIdx = cand;
            }
        }

        new_centers[k] = bestIdx;
    }

    return new_centers;

}

bool centers_changed(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return true;
    }
    return false;
}

float daviesBouldinIndex(
    snapshot_mod snaps,
    int N_frames,
    int N_atoms,
    int K,
    std::vector<int> clusters,
    std::vector<int> centers
) {
    std::vector<float> S(K, 0.0f);
    std::vector<int> counts(K, 0);

    // --- Compute S_i (intra-cluster scatter)
    for (int i = 0; i < N_frames; i++) {
        int k = clusters[i];
        // S[k] += rmsd[centroids[k] * N_frames + i];
        S[k] += calcul_RMSD(snaps[centers[k]],snaps[i],N_atoms);
        counts[k]++;
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0)
            S[k] /= counts[k];
    }

    // --- Compute DB
    float db = 0.0f;

    for (int i = 0; i < K; i++) {
        float maxR = 0.0f;

        for (int j = 0; j < K; j++) {
            if (i == j) continue;

            // float Mij = rmsd[centroids[i] * N_frames + centroids[j]];
            float Mij = calcul_RMSD(snaps[centers[i]],snaps[centers[j]],N_atoms);
            if (Mij > 0.0f) {
                float Rij = (S[i] + S[j]) / Mij;
                maxR = std::max(maxR, Rij);
            }
        }

        db += maxR;
    }

    return db / K;
}

int main(int argc, char** argv) {

    // Configuration OpenMP
    int num_threads = 8;  
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n\n";

    int max_iters = 100;
    int num_test_points = 20;

    int N_snap = 15000;
    int K = 10;
    int N_atome = 1000;

    std::cout << "============Clustering sur " << N_snap << " molecules avec " << N_atome << " atomes.============\n";

    // FileUtils file; 

    // std::cout << file << std::endl;

    // // size_t N_frames = file.getN_frames();
    // size_t N_frames = 10000;
    // size_t N_atoms  = file.getN_atoms();
    // size_t N_dims   = file.getN_dims();

    // // Load and reorder into X,Y,Z blocks
    // float* frame = file.loadData(N_frames);
    // file.reorderByLine(frame, N_frames);

    // Timers pour la mesure
    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point clust_t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point clust_t1;

    // On génère N snapshots aléatoires
    std::cout << "Generating snapshots..." << "\n";

    snapshot_mod snaps;
    snaps.reserve(N_snap);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int k = 0; k < N_snap; k++) {
        molecule_tab M(3, std::vector<double>(N_atome));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < N_atome; j++) {
                M[i][j] = dist(gen);
            }
        }
        snaps.push_back(std::move(M));
    }

    std::cout << "Snapshots generated !" << "\n";


    // RNG
    std::random_device rd;
    std::mt19937_64 rng((uint64_t)rd());

    // Initialisation Kmeans++
    std::cout << "Initialization of K-means centers...\n";
    std::vector<int> centers = init_kmeanspp_centers(snaps, K, rng);
    std::cout << "Initialized centers.\n";

    // Itérations jusqu'à équilibre
    std::vector<int> labels(snaps.size(), -1);

    std::cout << "Beginning clustering\n";

    clust_t0 = std::chrono::high_resolution_clock::now();


    for (int iter = 0; iter < max_iters; iter++) {
        t0 = std::chrono::high_resolution_clock::now();
        labels = assign_clusters(snaps, centers);
        

        std::cout << "Iter " << iter << " begins\n";
        std::vector<int> new_centers = update_centers_by_subsample_medoid(
            snaps, labels, K, num_test_points, rng, centers);

        bool changed = centers_changed(centers, new_centers);
        centers = std::move(new_centers);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Iteration done in " << std::chrono::duration<double>(t1-t0).count() << " seconds\n";

        float dbi = daviesBouldinIndex(snaps, N_snap, N_atome, K, labels, centers);

        std::cout << "Davies-Bouldin Index : " << dbi << "\n";  

        std::cout << "Iter " << iter << ": centers " << (changed ? "changed" : "unchanged") << "\n";
        if (!changed || iter > max_iters) {
            std::cout << "============CONVERGED. CLUSTERING DONE.============\n";
            break;
        }
    }

    clust_t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Clustering done in " << std::chrono::duration<double>(clust_t1-clust_t0).count() << " seconds\n";


    return 0;
}



