// =============================================================================
// cluster_cpu.cpp — CPU-only version of the GPU molecule clustering pipeline
//
// Compile:  g++ -O3 -std=c++17 -o cluster_cpu cluster_cpu.cpp
// Usage:    ./cluster_cpu <dataset.bin> [K] [MAX_ITER] [N_FRAMES]
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <omp.h>
#include <limits>

// ---------------------------------------------------------------------------
// Tiny timer
// ---------------------------------------------------------------------------
struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0;
    void start() { t0 = Clock::now(); }
    double elapsed_s() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Packed upper-triangle index  (i < j)
// ---------------------------------------------------------------------------
static inline size_t triIdx(size_t i, size_t j, size_t N) {
    if (i > j) { size_t tmp = i; i = j; j = tmp; }
    return i * N - (i * (i + 1)) / 2 + (j - i - 1);
}

static inline float getRMSD(size_t i, size_t j,
                             const float* packed, size_t N) {
    if (i == j) return 0.f;
    return packed[triIdx(i, j, N)];
}

// ---------------------------------------------------------------------------
// Analytical eigenvalues of a symmetric 3×3 matrix (Cardano).
// Returns lambda[0] >= lambda[1] >= lambda[2].
// ---------------------------------------------------------------------------
static void eigenvalues3x3(float m00, float m01, float m02,
                            float m11, float m12, float m22,
                            float* lambda)
{
    constexpr float PI = 3.14159265358979323846f;

    float trace = m00 + m11 + m22;
    float mean  = trace / 3.f;

    float s00 = m00 - mean, s11 = m11 - mean, s22 = m22 - mean;

    float p = s00*s00 + s11*s11 + s22*s22
            + 2.f * (m01*m01 + m02*m02 + m12*m12);
    p = std::sqrt(p / 6.f);

    float inv = (p > 1e-8f) ? (1.f / p) : 0.f;
    float b00 = s00*inv, b01 = m01*inv, b02 = m02*inv;
    float b11 = s11*inv, b12 = m12*inv, b22 = s22*inv;

    float det = b00*(b11*b22 - b12*b12)
              - b01*(b01*b22 - b12*b02)
              + b02*(b01*b12 - b11*b02);
    det *= 0.5f;
    det  = std::min(1.f, std::max(-1.f, det));

    float phi = std::acos(det) / 3.f;

    lambda[0] = mean + 2.f * p * std::cos(phi);
    lambda[2] = mean + 2.f * p * std::cos(phi + 2.f * PI / 3.f);
    lambda[1] = 3.f * mean - lambda[0] - lambda[2];

    // Sort descending
    if (lambda[0] < lambda[1]) std::swap(lambda[0], lambda[1]);
    if (lambda[1] < lambda[2]) std::swap(lambda[1], lambda[2]);
    if (lambda[0] < lambda[1]) std::swap(lambda[0], lambda[1]);
}

// ---------------------------------------------------------------------------
// Compute centroid and G = sum of squared distances to centroid.
// coords layout: [dim=3][N_atoms][N_frames], frame-stride = 1.
// (Same as the GPU code's interleaved layout.)
// ---------------------------------------------------------------------------
static void computeCentroidG(const float* coords,
                              size_t N_atoms, size_t N_frames, size_t frame,
                              float& cx, float& cy, float& cz, float& G)
{
    float sx = 0.f, sy = 0.f, sz = 0.f;
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t off = a * N_frames + frame;
        sx += coords[0 * N_atoms * N_frames + off];
        sy += coords[1 * N_atoms * N_frames + off];
        sz += coords[2 * N_atoms * N_frames + off];
    }
    cx = sx / N_atoms;
    cy = sy / N_atoms;
    cz = sz / N_atoms;

    float g = 0.f;
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t off = a * N_frames + frame;
        float rx = coords[0 * N_atoms * N_frames + off] - cx;
        float ry = coords[1 * N_atoms * N_frames + off] - cy;
        float rz = coords[2 * N_atoms * N_frames + off] - cz;
        g += rx*rx + ry*ry + rz*rz;
    }
    G = g;
}

// ---------------------------------------------------------------------------
// RMSD between two frames using the QCP / Kabsch eigenvalue approach.
// coords layout: [dim=3][N_atoms][N_frames].
// ---------------------------------------------------------------------------
static float computeRMSD(const float* coords,
                          size_t N_atoms, size_t N_frames,
                          size_t i, size_t j,
                          const float* cx, const float* cy,
                          const float* cz, const float* G)
{
    if (i == j) return 0.f;

    float rcx = cx[i], rcy = cy[i], rcz = cz[i];
    float scx = cx[j], scy = cy[j], scz = cz[j];

    // Build cross-covariance matrix A
    float a00=0,a01=0,a02=0;
    float a10=0,a11=0,a12=0;
    float a20=0,a21=0,a22=0;

    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t offi = a * N_frames + i;
        size_t offj = a * N_frames + j;

        float rx = coords[0*N_atoms*N_frames + offi] - rcx;
        float ry = coords[1*N_atoms*N_frames + offi] - rcy;
        float rz = coords[2*N_atoms*N_frames + offi] - rcz;

        float sx = coords[0*N_atoms*N_frames + offj] - scx;
        float sy = coords[1*N_atoms*N_frames + offj] - scy;
        float sz = coords[2*N_atoms*N_frames + offj] - scz;

        a00 += rx*sx;  a01 += rx*sy;  a02 += rx*sz;
        a10 += ry*sx;  a11 += ry*sy;  a12 += ry*sz;
        a20 += rz*sx;  a21 += rz*sy;  a22 += rz*sz;
    }

    // M = A^T * A  (symmetric)
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    float lambda[3];
    eigenvalues3x3(m00, m01, m02, m11, m12, m22, lambda);

    float sigma = std::sqrt(std::max(lambda[0], 0.f))
                + std::sqrt(std::max(lambda[1], 0.f))
                + std::sqrt(std::max(lambda[2], 0.f));

    float rmsd2 = (G[i] + G[j] - 2.f * sigma) / N_atoms;
    return std::sqrt(std::max(rmsd2, 0.f));
}

// ---------------------------------------------------------------------------
// KMedoids++ initialisation (same logic as the GPU version's host helper)
// ---------------------------------------------------------------------------
static void pickKMedoidsPlusPlus(size_t N, int K,
                                 const float* packed,
                                 std::vector<int>& centroids)
{
    std::mt19937 rng(42);
    centroids.clear();
    centroids.reserve(K);

    // pick first centroid at random
    std::uniform_int_distribution<int> dist(0, (int)N - 1);
    centroids.push_back(dist(rng));

    std::vector<float> minDist(N, std::numeric_limits<float>::max());

    for (int k = 1; k < K; ++k) {
        // Update min distances to nearest chosen centroid
        int last = centroids.back();
        for (size_t f = 0; f < N; ++f) {
            float d = getRMSD((size_t)last, f, packed, N);
            if (d < minDist[f]) minDist[f] = d;
        }
        // Sample proportional to d^2
        float total = 0.f;
        for (float d : minDist) total += d * d;
        std::uniform_real_distribution<float> udist(0.f, total);
        float r = udist(rng);
        float cum = 0.f;
        int chosen = (int)N - 1;
        for (size_t f = 0; f < N; ++f) {
            cum += minDist[f] * minDist[f];
            if (cum >= r) { chosen = (int)f; break; }
        }
        centroids.push_back(chosen);
    }
}

// ---------------------------------------------------------------------------
// Davies-Bouldin index
// ---------------------------------------------------------------------------
static float daviesBouldin(size_t N, int K,
                            const std::vector<int>& clusters,
                            const std::vector<int>& centroids,
                            const float* packed)
{
    // avg intra-cluster distance per cluster
    std::vector<float> s(K, 0.f);
    std::vector<int>   cnt(K, 0);
    for (size_t i = 0; i < N; ++i) {
        int c = clusters[i];
        s[c]  += getRMSD(i, (size_t)centroids[c], packed, N);
        cnt[c]++;
    }
    for (int k = 0; k < K; ++k)
        if (cnt[k] > 0) s[k] /= cnt[k];

    float db = 0.f;
    for (int i = 0; i < K; ++i) {
        float worst = 0.f;
        for (int j = 0; j < K; ++j) {
            if (i == j) continue;
            float dij = getRMSD((size_t)centroids[i],
                                (size_t)centroids[j], packed, N);
            if (dij > 1e-9f)
                worst = std::max(worst, (s[i] + s[j]) / dij);
        }
        db += worst;
    }
    return db / K;
}

// ---------------------------------------------------------------------------
// Read the .bin dataset  (matches FileUtils.cpp header format)
//
//   Header: 3 x size_t  →  n_snapshots_total, N_atoms, N_dims
//   Data:   [dim=3][atom][n_snapshots_total] floats  (already the right layout)
//
//   We only read N_frames frames (the first N_frames of n_snapshots_total).
//   The stride per atom per dim is n_snapshots_total, so we must slice.
// ---------------------------------------------------------------------------
static bool readBin(const char* path,
                    size_t N_frames,
                    size_t& N_atoms,
                    size_t& N_snapshots_total,
                    std::vector<float>& coords)   // [dim][atom][N_frames]
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }

    // Header: three size_t values written by FileUtils.
    // Detect whether the file was written with 32-bit or 64-bit size_t:
    // read 24 bytes, then try interpreting as 64-bit. If n_dims comes out
    // as 0 or nonsensical, fall back to 32-bit (4-byte) size_t.
    uint8_t hdr[24] = {};
    f.read(reinterpret_cast<char*>(hdr), 24);
    if (!f) { std::cerr << "Cannot read header from " << path << "\n"; return false; }

    size_t n_snapshots_file, n_atoms_file, n_dims_file;

    // Try 64-bit first
    uint64_t s64, a64, d64;
    memcpy(&s64, hdr +  0, 8);
    memcpy(&a64, hdr +  8, 8);
    memcpy(&d64, hdr + 16, 8);

    if (d64 == 3 && a64 > 0 && a64 < 1000000 && s64 > 0) {
        // 64-bit size_t — header already fully consumed (24 bytes)
        n_snapshots_file = (size_t)s64;
        n_atoms_file     = (size_t)a64;
        n_dims_file      = (size_t)d64;
        // seek back to right after the 24-byte header (already there)
    } else {
        // Try 32-bit size_t (header is only 12 bytes)
        uint32_t s32, a32, d32;
        memcpy(&s32, hdr + 0, 4);
        memcpy(&a32, hdr + 4, 4);
        memcpy(&d32, hdr + 8, 4);
        n_snapshots_file = (size_t)s32;
        n_atoms_file     = (size_t)a32;
        n_dims_file      = (size_t)d32;
        // Rewind to just after the 12-byte header
        f.seekg(3 * sizeof(uint32_t), std::ios::beg);
    }

    std::cout << "File header: " << n_snapshots_file << " snapshots, "
              << n_atoms_file << " atoms, " << n_dims_file << " dims\n";

    if (n_dims_file != 3) {
        std::cerr << "Expected N_dims=3, got " << n_dims_file << "\n";
        return false;
    }
    if (N_frames > n_snapshots_file) {
        std::cerr << "Requested " << N_frames << " frames but file only has "
                  << n_snapshots_file << "\n";
        return false;
    }

    N_atoms           = n_atoms_file;
    N_snapshots_total = n_snapshots_file;

    // Data layout in file: [dim][atom][n_snapshots_file]
    // We want:             [dim][atom][N_frames]   (first N_frames only)
    coords.resize(3 * N_atoms * N_frames);

    // header_bytes = current file position (right after the header)
    const size_t header_bytes = (size_t)f.tellg();

    for (size_t d = 0; d < 3; ++d) {
        for (size_t a = 0; a < N_atoms; ++a) {
            // File position for dim d, atom a, frame 0
            size_t file_offset = header_bytes
                + (d * N_atoms * n_snapshots_file + a * n_snapshots_file)
                * sizeof(float);
            f.seekg((std::streamoff)file_offset, std::ios::beg);

            float* dst = coords.data() + d * N_atoms * N_frames + a * N_frames;
            f.read(reinterpret_cast<char*>(dst), N_frames * sizeof(float));
            if (!f) {
                std::cerr << "Read error at dim=" << d << " atom=" << a << "\n";
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv)
{

    // Configuration OpenMP
    int num_threads = 16;  
    // if (argc > 1) {
    //     num_threads = std::atoi(argv[1]);
    // }
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset.bin> [K=10] [MAX_ITER=50] [N_FRAMES=10000]\n";
        return 1;
    }

    const int    K         = (argc >= 3) ? std::atoi(argv[2]) : 10;
    const int    MAX_ITER  = (argc >= 4) ? std::atoi(argv[3]) : 50;
    const size_t N_FRAMES  = (argc >= 5) ? (size_t)std::atoi(argv[4]) : 10000;

    // -----------------------------------------------------------------------
    // 1. Load data
    // -----------------------------------------------------------------------
    Timer t_total; t_total.start();
    Timer t; t.start();

    size_t N_atoms = 0, N_snapshots_total = 0;
    std::vector<float> coords;
    if (!readBin(argv[1], N_FRAMES, N_atoms, N_snapshots_total, coords)) return 1;

    printf("\n%s\n", std::string(70, '=').c_str());
    printf("DATASET INFO\n");
    printf("%s\n", std::string(70, '=').c_str());
    printf("Frames : %zu\n", N_FRAMES);
    printf("Atoms  : %zu\n", N_atoms);
    printf("%s\n", std::string(70, '=').c_str());
    printf("Load time : %.3f s\n\n", t.elapsed_s());

    // -----------------------------------------------------------------------
    // 2. Precompute centroids and G
    // -----------------------------------------------------------------------
    t.start();
    std::vector<float> cx(N_FRAMES), cy(N_FRAMES), cz(N_FRAMES), G(N_FRAMES);
    for (size_t i = 0; i < N_FRAMES; ++i)
        computeCentroidG(coords.data(), N_atoms, N_FRAMES, i,
                         cx[i], cy[i], cz[i], G[i]);
    printf("Centroid computation : %.3f s\n", t.elapsed_s());

    // -----------------------------------------------------------------------
    // 3. Compute full pairwise RMSD (upper triangle)
    // -----------------------------------------------------------------------
    t.start();
    size_t tri_size = N_FRAMES * (N_FRAMES - 1) / 2;
    std::vector<float> packed(tri_size, 0.f);

    for (size_t i = 0; i < N_FRAMES; ++i) {
        for (size_t j = i + 1; j < N_FRAMES; ++j) {
            float r = computeRMSD(coords.data(), N_atoms, N_FRAMES,
                                  i, j, cx.data(), cy.data(), cz.data(), G.data());
            packed[triIdx(i, j, N_FRAMES)] = r;
        }
    }
    printf("RMSD computation     : %.3f s\n", t.elapsed_s());

    // Optional: dump first row for validation (same as GPU version)
    {
        std::ofstream out("output/rmsd_row0.txt");
        if (out) {
            for (size_t j = 1; j < std::min((size_t)1000, N_FRAMES); ++j)
                out << j << " " << getRMSD(0, j, packed.data(), N_FRAMES) << "\n";
        }
    }

    // -----------------------------------------------------------------------
    // 4. K-Medoids clustering
    // -----------------------------------------------------------------------
    printf("\n%s\n", std::string(70, '=').c_str());
    printf("K-MEDOIDS CLUSTERING (K=%d, MAX_ITER=%d)\n", K, MAX_ITER);
    printf("%s\n", std::string(70, '=').c_str());

    t.start();

    std::vector<int> centroids;
    pickKMedoidsPlusPlus(N_FRAMES, K, packed.data(), centroids);

    std::vector<int>   clusters(N_FRAMES, 0);
    std::vector<float> frameCosts(N_FRAMES, 0.f);

    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        for (size_t f = 0; f < N_FRAMES; ++f) {
            int   best_k = 0;
            float best_d = std::numeric_limits<float>::max();
            for (int k = 0; k < K; ++k) {
                float d = getRMSD((size_t)centroids[k], f, packed.data(), N_FRAMES);
                if (d < best_d) { best_k = k; best_d = d; }
            }
            clusters[f] = best_k;
        }

        // --- ComputeMedoidCosts ---
        for (size_t f = 0; f < N_FRAMES; ++f) {
            int   c    = clusters[f];
            float cost = 0.f;
            for (size_t j = 0; j < N_FRAMES; ++j)
                if (clusters[j] == c)
                    cost += getRMSD(f, j, packed.data(), N_FRAMES);
            frameCosts[f] = cost;
        }

        // --- UpdateMedoids ---
        for (int k = 0; k < K; ++k) {
            int   best_idx  = -1;
            float best_cost = std::numeric_limits<float>::max();
            for (size_t f = 0; f < N_FRAMES; ++f) {
                if (clusters[f] == k && frameCosts[f] < best_cost) {
                    best_cost = frameCosts[f];
                    best_idx  = (int)f;
                }
            }
            if (best_idx >= 0) centroids[k] = best_idx;
        }
    }

    printf("Clustering time      : %.3f s\n", t.elapsed_s());

    // -----------------------------------------------------------------------
    // 5. Results
    // -----------------------------------------------------------------------
    float db = daviesBouldin(N_FRAMES, K, clusters, centroids, packed.data());

    // Random baseline: shuffle clusters, keep centroids random
    float rnd_db;
    {
        std::vector<int> rnd_clusters(N_FRAMES);
        std::vector<int> rnd_centroids(K);
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> di(0, K - 1);
        for (auto& c : rnd_clusters)  c = di(rng);
        std::uniform_int_distribution<int> df(0, (int)N_FRAMES - 1);
        for (auto& c : rnd_centroids) c = df(rng);
        rnd_db = daviesBouldin(N_FRAMES, K, rnd_clusters, rnd_centroids, packed.data());
    }

    float improvement = (rnd_db - db) / rnd_db * 100.f;

    printf("\n%s\n", std::string(70, '=').c_str());
    printf("CLUSTERING RESULTS\n");
    printf("%s\n", std::string(70, '=').c_str());
    printf("K-medoids Davies-Bouldin : %.6f\n", db);
    printf("Random    Davies-Bouldin : %.6f\n", rnd_db);
    printf("Improvement              : %.2f%% %s\n",
           improvement, improvement > 0 ? "✓ BETTER" : "✗ WORSE");

    std::vector<int> sizes(K, 0);
    for (int f = 0; f < (int)N_FRAMES; ++f) sizes[clusters[f]]++;
    printf("\nCluster centroids and sizes:\n");
    for (int k = 0; k < K; ++k) {
        float pct = 100.f * sizes[k] / N_FRAMES;
        printf("  Cluster %2d | Centroid: frame %6d | Size: %6d (%.2f%%)\n",
               k, centroids[k], sizes[k], pct);
    }
    printf("%s\n", std::string(70, '=').c_str());

    // Save clusters (simple text format)
    {
        std::ofstream out("output/clusters.txt");
        if (out) {
            out << "# frame cluster centroid\n";
            for (size_t f = 0; f < N_FRAMES; ++f)
                out << f << " " << clusters[f] << " " << centroids[clusters[f]] << "\n";
        }
    }

    printf("\nTotal wall time      : %.3f s\n\n", t_total.elapsed_s());
    return 0;
}