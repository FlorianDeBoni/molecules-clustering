#include <fstream>
#include <iostream>
#include <iomanip>

/*
 * Exports clustering results to JSON format
 * 
 * Parameters:
 *   - filename: output JSON file path
 *   - frame: array containing all molecular coordinates (reordered by dimension)
 *   - clusters: array mapping each snapshot to its cluster ID
 *   - centroids: array of centroid snapshot indices
 *   - N_frames: total number of snapshots
 *   - N_atoms: number of atoms per molecule
 *   - N_dims: number of dimensions (should be 3)
 *   - K: number of clusters
 */
void exportClusteringToJSON(
    const char* filename,
    const float* frame,
    const int* clusters,
    const int* centroids,
    int N_frames,
    int N_atoms,
    int N_dims,
    int K
) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    file << std::setprecision(6) << std::fixed;
    
    file << "{\n";
    file << "  \"metadata\": {\n";
    file << "    \"n_frames\": " << N_frames << ",\n";
    file << "    \"n_atoms\": " << N_atoms << ",\n";
    file << "    \"n_dimensions\": " << N_dims << ",\n";
    file << "    \"n_clusters\": " << K << "\n";
    file << "  },\n";
    
    // Export centroids
    file << "  \"centroids\": [";
    for (int k = 0; k < K; k++) {
        file << centroids[k];
        if (k < K - 1) file << ", ";
    }
    file << "],\n";
    
    // Export snapshots
    file << "  \"snapshots\": [\n";
    
    int block = N_atoms * N_frames;
    
    for (int f = 0; f < N_frames; f++) {
        file << "    {\n";
        file << "      \"id\": " << f << ",\n";
        file << "      \"cluster\": " << clusters[f] << ",\n";
        file << "      \"is_centroid\": ";
        
        // Check if this snapshot is a centroid
        bool is_centroid = false;
        for (int k = 0; k < K; k++) {
            if (centroids[k] == f) {
                is_centroid = true;
                break;
            }
        }
        file << (is_centroid ? "true" : "false") << ",\n";
        
        file << "      \"atoms\": [\n";
        
        for (int a = 0; a < N_atoms; a++) {
            // Calculate indices in the reordered array
            int idx_x = 0 * block + a * N_frames + f;
            int idx_y = 1 * block + a * N_frames + f;
            int idx_z = 2 * block + a * N_frames + f;
            
            file << "        {\"x\": " << frame[idx_x] 
                 << ", \"y\": " << frame[idx_y] 
                 << ", \"z\": " << frame[idx_z] << "}";
            
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