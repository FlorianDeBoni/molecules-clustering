#include <chrono>
#include <random>
#include <iostream>
#include <float.h>
#include <omp.h>
#include <cstdlib>
#include <fstream>

#include <iostream>
#include <forward_list>
#include <functional>
#include <limits>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>

#define M_PI 3.14159265358979323846

// Pour compilation :
// cd "C:\Users\Paul\Documents\Projet3A\Code\molecules-clustering"
// g++ -I "C:\Users\Paul\Documents\Projet3A\Code\molecules-clustering\external\eigen-5.0.0" cpu_clustering.cpp -o cpu_clustering


using atome_tab = std::array< double , 3 >;
// de taille 3 * N
using molecule_tab = std::vector< std::vector< double > >;
using snapshot_mod = std::vector < molecule_tab >;
using rot_matrix = std::vector< std::vector< double > >;
using matrix_mod = std::vector< std::vector< double > >;


// Calcul du produit matriciel carre naif
matrix_mod matprod(const matrix_mod& A, const matrix_mod& B, int size) {
    matrix_mod C(size,std::vector<double>(size));
    for(int i=0; i < size; i++) {
        for(int j=0; j < size; j++) {

            double coeff_sum = 0.0;

            for(int k=0; k < size; k++) {
                coeff_sum += A[i][k] * B[k][j];
            }

            C[i][j] = coeff_sum;

        }
    }

    return C;
}

// Calcul du produit matriciel naif pour matrices de tailles différentes
matrix_mod matprod_diff(const matrix_mod& A, const matrix_mod& B, int row_size1, int com_size, int col_size2) {
    matrix_mod C(row_size1,std::vector<double>(col_size2));
    for(int i=0; i < row_size1; i++) {
        for(int j=0; j < col_size2; j++) {

            double coeff_sum = 0.0;

            for(int k=0; k < com_size; k++) {
                coeff_sum += A[i][k] * B[k][j];
            }

            C[i][j] = coeff_sum;

        }
    }

    return C;
}

// Calcul de la trace
double trace(matrix_mod A, int size) {
    double tr = 0.0;
    for(int i=0; i < size; i++) {
        tr += A[i][i];
    }
    return tr;
}

// Calcul de la transposée
matrix_mod transpose(const matrix_mod& A, int row_size, int col_size) {
    matrix_mod A_t(col_size, std::vector<double>(row_size));
    for(int i=0; i < col_size; ++i) {
        for(int j=0; j < row_size; j++) {
            A_t[i][j] = A[j][i];
        }
    }
    return A_t;
}

// Calcul du déterminant
double det(matrix_mod A) {
    double term1 = A[0][0] * A[1][1] * A[2][2];
    double term2 = A[0][0] * A[1][2] * A[2][1];
    double term3 = A[0][1] * A[1][2] * A[2][0];
    double term4 = A[0][1] * A[1][0] * A[2][2];
    double term5 = A[0][2] * A[1][0] * A[2][1];
    double term6 = A[0][2] * A[1][1] * A[2][0];

    return (term1-term2) + (term3-term4) + (term5-term6);
}

// std::cout << "Just checking\n";

matrix_mod get_rot_matrix(const molecule_tab &mol1, const molecule_tab &mol2, int size) {

    // Step 1
    matrix_mod A(3,std::vector<double>(3));
    A = matprod_diff(mol1,transpose(mol2,3,size),3,size,3);

    // Step 2
    double eigenval1;
    double eigenval2;
    double eigenval3;

    matrix_mod prodAtA = matprod(transpose(A,3,3),A,3);

    double a = -1.0;
    double b = trace(prodAtA,3);
    double c = -(1.0/2.0) * ( std::pow(trace(prodAtA,3),2) - trace(matprod(prodAtA,prodAtA,3),3));
    double d = det(prodAtA);

    double p = (3.0 * a * c - b * b) / (3.0 * a * a);
    double q = (27.0 * a * a * d - 9.0 * a * b * c + 2.0 * pow(b,3)) / (27.0 * pow(a,3));
    double r = std::sqrt(-std::pow(p / 3.0, 3.0));
    double theta = (1.0/3.0) * std::acos(-q / (2*r));
    
    eigenval1 = (2 * std::cbrt(r) * std::cos(theta)) - (b/(3*a));
    eigenval2 = (2 * std::cbrt(r) * std::cos(theta + 2 * M_PI / 3)) - (b/(3*a));
    eigenval3 = (2 * std::cbrt(r) * std::cos(theta + 4 * M_PI / 3)) - (b/(3*a));


    // Step 3
    matrix_mod sigma = {{eigenval1, 0.0, 0.0}, {0.0, eigenval2, 0.0}, {0.0, 0.0, eigenval3}};

    //Step 4
    Eigen::MatrixXd ATA(3,3);
    for(int i=0; i < 2; i++) {
        for(int j=0; j < 2; j++) {
            ATA(i,j) = prodAtA[i][j];
        }   
    }

    // On réalise l'équivalent du Gaussian Elimination Method
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(ATA);

    // Step 5, Eigenvector matrix
    Eigen::MatrixXd V_eigen = es.eigenvectors();

    std::vector<double> eigenvect1 = {V_eigen(0,0), V_eigen(1,0), V_eigen(2,0)};
    std::vector<double> eigenvect2 = {V_eigen(0,1), V_eigen(1,1), V_eigen(2,1)};
    std::vector<double> eigenvect3 = {V_eigen(0,2), V_eigen(1,2), V_eigen(2,2)};

    matrix_mod V = {eigenvect1, eigenvect2, eigenvect3};

    // Step 6
    matrix_mod U = matprod(A,V,3);

    for(int j=0; j < 3; j++) {
        double sing_value = sigma[j][j];
        if (sing_value == 0.0) {
            U[j][0] = 0.0;
            U[j][1] = 0.0;
            U[j][2] = 0.0;
        }
        else {
            for(int i=0; i < 3; i++) {
                U[i][j] = U[i][j] / sing_value;
            }
        }
    }

    return U;
}

// Fonction qui calcule le vecteur multiplié par la matrice de rotation
std::vector<double> rotated_vect(const matrix_mod& X, const matrix_mod& U, int size) {
    matrix_mod Xtemp = matprod_diff(U,X,3,3,1);
    std::vector<double> Xrot = {Xtemp[0][0], Xtemp[1][0], Xtemp[2][0]};
    return Xrot;
}

// Fonction qui calcule la norme euclidienne au carrée entre deux vecteurs
double sq_eucld_norm(std::vector<double> X, std::vector<double> Y) {
    return std::pow(X[0] - Y[0],2) + std::pow(X[1] - Y[1],2) + std::pow(X[2] - Y[2],2);
}

// Fonction qui calcule la RMSD entre deux snapshots
double calcul_RMSD(const molecule_tab& mol1, const molecule_tab& mol2, const matrix_mod &U, int size) {
    double rmsd = 0.0;
    for(int i=0; i < size; i++) {
        matrix_mod Xi = {{mol1[0][i]}, {mol1[1][i]}, {mol1[2][i]}};
        std::vector<double> Yi = {mol2[0][i], mol2[1][i], mol2[2][i]};

        std::vector<double> Xirot = rotated_vect(Xi, U, size);

        rmsd += sq_eucld_norm(Xirot, Yi);
    }

    rmsd = rmsd / size;
    rmsd = std::sqrt(rmsd);

    return rmsd;
}

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
        for (int i = 0; i < N; i++) {
            matrix_mod U = get_rot_matrix(snaps[i], snaps[lastCenter], N_atome);
            double d = calcul_RMSD(snaps[i], snaps[lastCenter], U, N_atome);
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

    for (int i = 0; i < N; i++) {
        double best = std::numeric_limits<double>::infinity();
        int bestk = -1;
        for (int k = 0; k < K; k++) {
            matrix_mod U = get_rot_matrix(snaps[i], snaps[centers[k]], N_atome);
            double d = calcul_RMSD(snaps[i], snaps[centers[k]], U, N_atome);
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
                matrix_mod U = get_rot_matrix(snaps[cand], snaps[j], N_atome);
                score += calcul_RMSD(snaps[cand], snaps[j], U, N_atome);
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

int main(int argc, char** argv) {

    int max_iters = 100;
    int num_test_points = 20;

    int N_snap = 5000;
    int K = 10;
    int N_atome = 100;

    // Timers pour la mesure
    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;

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

        std::cout << "Iter " << iter << ": centers " << (changed ? "changed" : "unchanged") << "\n";
        if (!changed || iter > 6) {
            std::cout << "Converged. Clustering done.\n";
            break;
        }
    }


    return 0;
}



