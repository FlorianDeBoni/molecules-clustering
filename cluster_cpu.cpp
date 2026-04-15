// =============================================================================
// cluster_cpu.cpp — CPU-only molecule clustering (RMSD + K-Medoids)
//
// Compile:  g++ -O3 -std=c++17 -o cluster_cpu cluster_cpu.cpp
//       or: g++ -O3 -std=c++17 -fopenmp -o cluster_cpu cluster_cpu.cpp
// Usage:    ./cluster_cpu dataset.bin [K=10] [MAX_ITER=50] [N_FRAMES=10000]
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
// Timer
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
// Export Clustering Results
// ---------------------------------------------------------------------------
void exportClusteringToJSON(
    const char* filename,
    const float* frame,
    const std::vector<int>& clusters,
    const std::vector<int>& centroids,
    int N_frames,
    int N_atoms,
    int N_dims,
    int K
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for writing" << std::endl;
        return;
    }

    file << std::setprecision(6) << std::fixed;

    // ── metadata ────────────────────────────────────────────────────────────
    file << "{\n";
    file << "  \"metadata\": {\n";
    file << "    \"n_frames\": "     << N_frames << ",\n";
    file << "    \"n_atoms\": "      << N_atoms  << ",\n";
    file << "    \"n_dimensions\": " << N_dims   << ",\n";
    file << "    \"n_clusters\": "   << K        << "\n";
    file << "  },\n";

    // ── centroids list ───────────────────────────────────────────────────────
    file << "  \"centroids\": [";
    for (int k = 0; k < K; k++) {
        file << centroids[k];
        if (k < K - 1) file << ", ";
    }
    file << "],\n";

    // ── snapshots ────────────────────────────────────────────────────────────
    file << "  \"snapshots\": [\n";

    const int snapshot_stride = N_atoms * N_dims;

    for (int f = 0; f < N_frames; f++) {

        // Check whether this snapshot is a medoid
        bool is_centroid = false;
        for (int k = 0; k < K; k++) {
            if (centroids[k] == f) {
                is_centroid = true;
                break;
            }
        }

        file << "    {\n";
        file << "      \"id\": "          << f            << ",\n";
        file << "      \"cluster\": "     << clusters[f]  << ",\n";
        file << "      \"is_centroid\": " << (is_centroid ? "true" : "false") << ",\n";
        file << "      \"atoms\": [\n";

        const int base = f * snapshot_stride;

        for (int a = 0; a < N_atoms; a++) {
            float x = frame[base + a * 3 + 0];
            float y = frame[base + a * 3 + 1];
            float z = frame[base + a * 3 + 2];

            file << "        {\"x\": " << x
                 << ", \"y\": "        << y
                 << ", \"z\": "        << z << "}";

            if (a < N_atoms - 1) file << ",";
            file << "\n";
        }

        file << "      ]\n";
        file << "    }";

        if (f < N_frames - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Clustering results exported to " << filename << std::endl;
}

 
// ---------------------------------------------------------------------------
// Packed upper-triangle helpers
// ---------------------------------------------------------------------------
static inline size_t triIdx(size_t i, size_t j, size_t N) {
    if (i > j) { size_t t = i; i = j; j = t; }
    return i * N - (i * (i + 1)) / 2 + (j - i - 1);
}
static inline float getRMSD(size_t i, size_t j, const float* packed, size_t N) {
    if (i == j) return 0.f;
    return packed[triIdx(i, j, N)];
}
 
void exportRMSDMatrix(
    const char* filename,
    const float* packed,
    size_t N
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float val = getRMSD(i, j, packed, N);
            file << val;
            if (j < N - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "RMSD matrix exported to " << filename << std::endl;
}

// ---------------------------------------------------------------------------
// Analytical eigenvalues of symmetric 3x3 (Cardano) — descending order
// ---------------------------------------------------------------------------
static void eigenvalues3x3(float m00, float m01, float m02,
                            float m11, float m12, float m22,
                            float* lam)
{
    constexpr float PI = 3.14159265358979323846f;
    float trace = m00 + m11 + m22;
    float mean  = trace / 3.f;
    float s00 = m00-mean, s11 = m11-mean, s22 = m22-mean;
    float p = s00*s00 + s11*s11 + s22*s22 + 2.f*(m01*m01+m02*m02+m12*m12);
    p = std::sqrt(p / 6.f);
    float inv = (p > 1e-8f) ? 1.f/p : 0.f;
    float b00=s00*inv, b01=m01*inv, b02=m02*inv;
    float b11=s11*inv, b12=m12*inv, b22=s22*inv;
    float det = b00*(b11*b22-b12*b12) - b01*(b01*b22-b12*b02) + b02*(b01*b12-b11*b02);
    det = std::min(1.f, std::max(-1.f, det*0.5f));
    float phi = std::acos(det) / 3.f;
    lam[0] = mean + 2.f*p*std::cos(phi);
    lam[2] = mean + 2.f*p*std::cos(phi + 2.f*PI/3.f);
    lam[1] = 3.f*mean - lam[0] - lam[2];
    if (lam[0]<lam[1]) std::swap(lam[0],lam[1]);
    if (lam[1]<lam[2]) std::swap(lam[1],lam[2]);
    if (lam[0]<lam[1]) std::swap(lam[0],lam[1]);
}
 
// ---------------------------------------------------------------------------
// coords layout: [dim][atom][N_frames]  (stride between frames = 1)
// ---------------------------------------------------------------------------
static void computeCentroidG(const float* coords,
                              size_t N_atoms, size_t N_frames, size_t frame,
                              float& cx, float& cy, float& cz, float& G)
{
    float sx=0,sy=0,sz=0;
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t off = a*N_frames + frame;
        sx += coords[0*N_atoms*N_frames + off];
        sy += coords[1*N_atoms*N_frames + off];
        sz += coords[2*N_atoms*N_frames + off];
    }
    cx = sx/N_atoms; cy = sy/N_atoms; cz = sz/N_atoms;
    float g=0;
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t off = a*N_frames + frame;
        float rx = coords[0*N_atoms*N_frames+off]-cx;
        float ry = coords[1*N_atoms*N_frames+off]-cy;
        float rz = coords[2*N_atoms*N_frames+off]-cz;
        g += rx*rx+ry*ry+rz*rz;
    }
    G = g;
}
 
static float computeRMSD(const float* coords,
                          size_t N_atoms, size_t N_frames,
                          size_t i, size_t j,
                          const float* cx, const float* cy,
                          const float* cz, const float* G)
{
    if (i == j) return 0.f;
    float rcx=cx[i], rcy=cy[i], rcz=cz[i];
    float scx=cx[j], scy=cy[j], scz=cz[j];
 
    float a00=0,a01=0,a02=0, a10=0,a11=0,a12=0, a20=0,a21=0,a22=0;

    #pragma omp parallel for schedule(dynamic, 32)
    for (size_t a = 0; a < N_atoms; ++a) {
        size_t oi = a*N_frames+i, oj = a*N_frames+j;
        float rx = coords[0*N_atoms*N_frames+oi]-rcx;
        float ry = coords[1*N_atoms*N_frames+oi]-rcy;
        float rz = coords[2*N_atoms*N_frames+oi]-rcz;
        float sx = coords[0*N_atoms*N_frames+oj]-scx;
        float sy = coords[1*N_atoms*N_frames+oj]-scy;
        float sz = coords[2*N_atoms*N_frames+oj]-scz;
        a00+=rx*sx; a01+=rx*sy; a02+=rx*sz;
        a10+=ry*sx; a11+=ry*sy; a12+=ry*sz;
        a20+=rz*sx; a21+=rz*sy; a22+=rz*sz;
    }
    // M = A^T * A
    float m00=a00*a00+a10*a10+a20*a20;
    float m01=a00*a01+a10*a11+a20*a21;
    float m02=a00*a02+a10*a12+a20*a22;
    float m11=a01*a01+a11*a11+a21*a21;
    float m12=a01*a02+a11*a12+a21*a22;
    float m22=a02*a02+a12*a12+a22*a22;
 
    float lam[3];
    eigenvalues3x3(m00,m01,m02,m11,m12,m22,lam);
    float sigma = std::sqrt(std::max(lam[0],0.f))
                + std::sqrt(std::max(lam[1],0.f))
                + std::sqrt(std::max(lam[2],0.f));
    float rmsd2 = (G[i]+G[j]-2.f*sigma) / (float)N_atoms;
    return std::sqrt(std::max(rmsd2, 0.f));
}
 
// ---------------------------------------------------------------------------
// KMedoids++ init
// ---------------------------------------------------------------------------
static void pickKMedoidsPlusPlus(size_t N, int K,
                                  const float* packed,
                                  std::vector<int>& centroids)
{
    std::mt19937 rng(42);
    centroids.clear(); centroids.reserve(K);
    std::uniform_int_distribution<int> di(0,(int)N-1);
    centroids.push_back(di(rng));
 
    std::vector<float> minD(N, std::numeric_limits<float>::max());
    for (int k = 1; k < K; ++k) {
        int last = centroids.back();
        for (size_t f = 0; f < N; ++f) {
            float d = getRMSD((size_t)last, f, packed, N);
            if (d < minD[f]) minD[f] = d;
        }
        float total = 0.f;
        for (float d : minD) total += d*d;
        std::uniform_real_distribution<float> ud(0.f, total);
        float r = ud(rng), cum = 0.f;
        int chosen = (int)N-1;
        for (size_t f = 0; f < N; ++f) {
            cum += minD[f]*minD[f];
            if (cum >= r) { chosen=(int)f; break; }
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
    std::vector<float> s(K,0.f); std::vector<int> cnt(K,0);
    for (size_t i = 0; i < N; ++i) {
        int c = clusters[i];
        s[c] += getRMSD(i,(size_t)centroids[c],packed,N);
        cnt[c]++;
    }
    for (int k=0;k<K;++k) if(cnt[k]>0) s[k]/=cnt[k];
    float db=0.f;
    for (int i=0;i<K;++i) {
        float worst=0.f;
        for (int j=0;j<K;++j) {
            if(i==j) continue;
            float dij=getRMSD((size_t)centroids[i],(size_t)centroids[j],packed,N);
            if(dij>1e-9f) worst=std::max(worst,(s[i]+s[j])/dij);
        }
        db+=worst;
    }
    return db/K;
}
 
// ---------------------------------------------------------------------------
// Read .bin file (FileUtils format):
//   Header: n_snapshots, n_atoms, n_dims  (each a size_t — 4 or 8 bytes)
//   Data:   [dim=3][atom][n_snapshots_total] floats
//
// Auto-detects 32-bit vs 64-bit size_t from the header.
// Loads only the first N_frames frames per atom.
// Output coords layout: [dim][atom][N_frames]
// ---------------------------------------------------------------------------
static bool readBin(const char* path,
                    size_t N_frames,
                    size_t& N_atoms,
                    std::vector<float>& coords)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }
 
    uint8_t hdr[24] = {};
    f.read(reinterpret_cast<char*>(hdr), 24);
    if (!f) { std::cerr << "Cannot read header\n"; return false; }
 
    size_t n_snap, n_atoms_f, n_dims_f;
    size_t header_bytes;
 
    // Try 64-bit size_t first
    uint64_t s64, a64, d64;
    memcpy(&s64, hdr+ 0, 8);
    memcpy(&a64, hdr+ 8, 8);
    memcpy(&d64, hdr+16, 8);
    if (d64==3 && a64>0 && a64<10000000 && s64>0 && s64<100000000) {
        n_snap=s64; n_atoms_f=a64; n_dims_f=d64; header_bytes=24;
    } else {
        // Fall back to 32-bit size_t
        uint32_t s32, a32, d32;
        memcpy(&s32, hdr+0, 4);
        memcpy(&a32, hdr+4, 4);
        memcpy(&d32, hdr+8, 4);
        n_snap=s32; n_atoms_f=a32; n_dims_f=d32; header_bytes=12;
    }
 
    std::cout << "File: " << n_snap << " snapshots, "
              << n_atoms_f << " atoms, " << n_dims_f << " dims"
              << " (header=" << header_bytes << " bytes)\n";
 
    if (n_dims_f != 3) { std::cerr << "Expected N_dims=3, got " << n_dims_f << "\n"; return false; }
    if (N_frames > n_snap) {
        std::cerr << "Requested " << N_frames << " frames but file has " << n_snap << "\n";
        return false;
    }
 
    N_atoms = n_atoms_f;
    coords.resize(3 * N_atoms * N_frames);
 
    // Read entire dim-block at once, then slice out the first N_frames per atom.
    // This avoids thousands of seeks and is robust to any n_snap value.
    std::vector<float> block(N_atoms * n_snap);
    for (size_t d = 0; d < 3; ++d) {
        size_t file_offset = header_bytes + d * N_atoms * n_snap * sizeof(float);
        f.seekg((std::streamoff)file_offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(block.data()),
               N_atoms * n_snap * sizeof(float));
        if (!f) { std::cerr << "Read error on dim=" << d << "\n"; return false; }
 
        for (size_t a = 0; a < N_atoms; ++a) {
            // In file: atom a starts at block[a * n_snap]
            // In coords: atom a starts at coords[d*N_atoms*N_frames + a*N_frames]
            const float* src = block.data() + a * n_snap;       // first N_frames of this atom
            float*       dst = coords.data() + d*N_atoms*N_frames + a*N_frames;
            std::memcpy(dst, src, N_frames * sizeof(float));
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
    int num_threads = 6;  
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " dataset.bin [K=10] [MAX_ITER=50] [N_FRAMES=10000]\n";
        return 1;
    }
    const int    K        = (argc>=3) ? std::atoi(argv[2]) : 10;
    const int    MAX_ITER = (argc>=4) ? std::atoi(argv[3]) : 50;
    const size_t N_FRAMES = (argc>=5) ? (size_t)std::atoi(argv[4]) : 10000;
 
    Timer t_total; t_total.start();
    Timer t;
 
    // 1. Load
    t.start();
    size_t N_atoms = 0;
    std::vector<float> coords;
    if (!readBin(argv[1], N_FRAMES, N_atoms, coords)) return 1;
    printf("\n%s\n", std::string(70,'=').c_str());
    printf("Frames=%zu  Atoms=%zu  K=%d  MAX_ITER=%d\n", N_FRAMES, N_atoms, K, MAX_ITER);
    printf("%s\n", std::string(70,'=').c_str());
    printf("Load         : %.3f s\n", t.elapsed_s());
 
    // 2. Centroids + G
    t.start();
    std::vector<float> cx(N_FRAMES), cy(N_FRAMES), cz(N_FRAMES), G(N_FRAMES);
    for (size_t i = 0; i < N_FRAMES; ++i)
        computeCentroidG(coords.data(), N_atoms, N_FRAMES, i,
                         cx[i], cy[i], cz[i], G[i]);
    printf("Centroids    : %.3f s\n", t.elapsed_s());
 
    // Sanity check: print a few RMSD values
    printf("Sanity RMSD(0,1)=%.4f  RMSD(0,2)=%.4f  RMSD(1,2)=%.4f\n",
           computeRMSD(coords.data(),N_atoms,N_FRAMES,0,1,cx.data(),cy.data(),cz.data(),G.data()),
           computeRMSD(coords.data(),N_atoms,N_FRAMES,0,2,cx.data(),cy.data(),cz.data(),G.data()),
           computeRMSD(coords.data(),N_atoms,N_FRAMES,1,2,cx.data(),cy.data(),cz.data(),G.data()));
 
    // 3. Full pairwise RMSD
    t.start();
    size_t tri_size = N_FRAMES*(N_FRAMES-1)/2;
    std::vector<float> packed(tri_size, 0.f);

    for (size_t i = 0; i < N_FRAMES; ++i)
        for (size_t j = i+1; j < N_FRAMES; ++j)
            packed[triIdx(i,j,N_FRAMES)] =
                computeRMSD(coords.data(),N_atoms,N_FRAMES,i,j,
                            cx.data(),cy.data(),cz.data(),G.data());
    printf("RMSD matrix  : %.3f s\n", t.elapsed_s());
 
    // Save first row for validation
    {
        std::ofstream out("output/rmsd_row0.txt");
        if (out) for (size_t j=1; j<std::min((size_t)1000,N_FRAMES); ++j)
            out << j << " " << getRMSD(0,j,packed.data(),N_FRAMES) << "\n";
    }
 
    // 4. K-Medoids
    printf("\n%s\nK-MEDOIDS (K=%d, MAX_ITER=%d)\n%s\n",
           std::string(70,'=').c_str(), K, MAX_ITER, std::string(70,'=').c_str());
    t.start();
 
    std::vector<int> centroids;
    pickKMedoidsPlusPlus(N_FRAMES, K, packed.data(), centroids);
 
    std::vector<int>   clusters(N_FRAMES, 0);
    std::vector<float> frameCosts(N_FRAMES, 0.f);
 
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Assign
        for (size_t f = 0; f < N_FRAMES; ++f) {
            int best_k=0; float best_d=std::numeric_limits<float>::max();
            for (int k=0;k<K;++k) {
                float d=getRMSD((size_t)centroids[k],f,packed.data(),N_FRAMES);
                if(d<best_d){best_k=k;best_d=d;}
            }
            clusters[f]=best_k;
        }
        // Costs
        for (size_t f = 0; f < N_FRAMES; ++f) {
            int c=clusters[f]; float cost=0.f;
            for (size_t j=0;j<N_FRAMES;++j)
                if(clusters[j]==c) cost+=getRMSD(f,j,packed.data(),N_FRAMES);
            frameCosts[f]=cost;
        }
        // Update medoids
        for (int k=0;k<K;++k) {
            int best=-1; float best_c=std::numeric_limits<float>::max();
            for (size_t f=0;f<N_FRAMES;++f)
                if(clusters[f]==k && frameCosts[f]<best_c){best_c=frameCosts[f];best=(int)f;}
            if(best>=0) centroids[k]=best;
        }
    }
    printf("Clustering   : %.3f s\n", t.elapsed_s());
 
    // 5. Results
    float db  = daviesBouldin(N_FRAMES, K, clusters, centroids, packed.data());
    float rdb;
    {
        std::vector<int> rc(N_FRAMES), rm(K);
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> di(0,K-1), df(0,(int)N_FRAMES-1);
        for (auto& c:rc) c=di(rng);
        for (auto& c:rm) c=df(rng);
        rdb = daviesBouldin(N_FRAMES, K, rc, rm, packed.data());
    }
    float impr = (rdb-db)/rdb*100.f;
 
    printf("\n%s\nRESULTS\n%s\n", std::string(70,'=').c_str(), std::string(70,'=').c_str());
    printf("K-medoids DB index : %.6f\n", db);
    printf("Random    DB index : %.6f\n", rdb);
    printf("Improvement        : %.2f%% %s\n", impr, impr>0?"✓ BETTER":"✗ WORSE");
 
    std::vector<int> sizes(K,0);
    for (int f=0;f<(int)N_FRAMES;++f) sizes[clusters[f]]++;
    printf("\nCluster centroids and sizes:\n");
    for (int k=0;k<K;++k)
        printf("  Cluster %2d | Centroid: frame %6d | Size: %6d (%.2f%%)\n",
               k, centroids[k], sizes[k], 100.f*sizes[k]/N_FRAMES);
    printf("%s\n", std::string(70,'=').c_str());
 
    // Save
    {
        std::ofstream out("output/clusters.txt");
        if (out) {
            out << "# frame cluster centroid\n";
            for (size_t f=0;f<N_FRAMES;++f)
                out << f << " " << clusters[f] << " " << centroids[clusters[f]] << "\n";
        }
    }

    // exportClusteringToJSON(
    //     "output/clustering_results.json",
    //     coords.data(),
    //     clusters,
    //     centroids,
    //     N_FRAMES,
    //     N_atoms,
    //     3,
    //     K
    // );

    // exportRMSDMatrix("output/rmsd_matrix.csv", packed.data(), N_FRAMES);
 
    printf("\nTotal wall time : %.3f s\n\n", t_total.elapsed_s());
    return 0;
}