// FileUtils.cpp
#include "FileUtils.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string.h>

FileUtils::FileUtils(const std::string& file_name) 
    : file_name(file_name), file(file_name, std::ios::binary) 
{
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    // Read header information
    file.read(reinterpret_cast<char*>(&n_frames), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_atoms), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_dims), sizeof(size_t));

    if (!file) {
        throw std::runtime_error("Error reading header from" + file_name);
    }
    
    std::cout << "Loaded binary file: " << n_frames << " frames, " 
              << n_atoms << " atoms, " << n_dims << " dimensions" << std::endl;
}

size_t FileUtils::getN_atoms() const { return n_atoms; }
size_t FileUtils::getN_frames() const { return n_frames; }
size_t FileUtils::getN_dims() const { return n_dims; }
std::ifstream& FileUtils::getFile() const { return const_cast<std::ifstream&>(file); }

std::ostream& operator<<(std::ostream& os, const FileUtils& f) {
    size_t n_atoms = f.getN_atoms();
    size_t n_dims = f.getN_dims();
    size_t n_frames = f.getN_frames();
    std::ifstream& file = f.getFile();

    std::vector<float> frame_data(n_atoms * n_dims);
    
    // Frame 1 (index 0)
    file.clear();  // Clear any error flags
    file.seekg(3 * sizeof(size_t) + 0 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));
    
    if (!file) {
        os << "Error reading frame 1" << std::endl;
        return os;
    }
    
    os << "Snapshot 1:" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
    
    // Frame 2 (index 1)
    if (n_frames > 1) {
        file.clear();
        file.seekg(3 * sizeof(size_t) + 1 * n_atoms * n_dims * sizeof(float), std::ios::beg);
        file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));
        
        if (!file) {
            os << "Error reading frame 2" << std::endl;
            return os;
        }

        os << "Snapshot 2:" << std::endl
           << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
           << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
           << "..." << std::endl
           << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
        os << "..." << std::endl << std::endl;
    }

    // Last frame (index n_frames - 1)
    if (n_frames > 0) {
        file.clear();
        file.seekg(3 * sizeof(size_t) + (n_frames - 1) * n_atoms * n_dims * sizeof(float), std::ios::beg);
        file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));
        
        if (!file) {
            os << "Error reading last frame" << std::endl;
            return os;
        }

        os << "Snapshot " << n_frames << ":" << std::endl
           << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
           << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
           << "..." << std::endl
           << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")";
    }
    
    return os;
}

std::vector<float> FileUtils::readFrame(size_t frame_idx) {
    if (frame_idx >= n_frames) {
        throw std::out_of_range("Frame index " + std::to_string(frame_idx) + 
                                " out of range [0, " + std::to_string(n_frames) + ")");
    }

    std::vector<float> frame_data(n_atoms * n_dims);
    
    // Clear any error flags before seeking
    file.clear();
    
    // Seek to the correct frame (was always reading frame 0!)
    file.seekg(3 * sizeof(size_t) + frame_idx * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    if (!file) {
        throw std::runtime_error("Error reading frame " + std::to_string(frame_idx));
    }

    return frame_data;
}

/*
Before:
Frame0: [atom0_x, atom0_y, atom0_z, atom1_x, atom1_y, atom1_z, ...]
Frame1: [atom0_x, atom0_y, atom0_z, atom1_x, atom1_y, atom1_z, ...]

After:
All X coords: [atom0_frame0, atom0_frame1, ..., atom0_frameN, atom1_frame0, atom1_frame1, ...]
All Y coords: [atom0_frame0, atom0_frame1, ..., atom0_frameN, atom1_frame0, atom1_frame1, ...]
All Z coords: [atom0_frame0, atom0_frame1, ..., atom0_frameN, atom1_frame0, atom1_frame1, ...]
*/
void FileUtils::reorderByLine(float* frame_data, const size_t n_subset_frames) {

    const size_t n_coords = 3;
    const size_t frame_size = n_atoms * n_coords;
    const size_t total = n_subset_frames * frame_size;

    std::vector<float> tmp(total);
    memcpy(tmp.data(), frame_data, total * sizeof(float));

    // Indexing helper lambdas
    auto old_index = [&](size_t f, size_t a, size_t c) {
        return f * frame_size + a * n_coords + c; // original layout
    };

    auto new_index = [&](size_t a, size_t f, size_t c) {
        return c * n_atoms * n_subset_frames
             + a * n_subset_frames
             + f;
    };

    for (size_t f = 0; f < n_subset_frames; ++f) {
        for (size_t a = 0; a < n_atoms; ++a) {
            for (size_t c = 0; c < n_coords; ++c) {
                frame_data[new_index(a, f, c)] = tmp[old_index(f, a, c)];
            }
        }
    }
}

/*
* Loads n_subset_frames (*52 Kbytes) data into memory (RAM) 
* must be <= n_frames which is the total
* number of frames in the file.
* returns a pointer to a detached C-like array
* (must be called with delete[] later)
*/
float* FileUtils::loadData(size_t n_subset_frames) {
    if (n_subset_frames > n_frames) {
        std::cerr << "Error: number of frames requested " << n_subset_frames 
                  << " > " << n_frames << std::endl;
        throw std::invalid_argument("Requested frames exceed available frames");
    }
    
    if (n_subset_frames == 0) {
        std::cerr << "Error: cannot load 0 frames" << std::endl;
        throw std::invalid_argument("n_subset_frames must be > 0");
    }

    size_t n_elements = n_subset_frames * n_atoms * n_dims;
    float* data = new (std::nothrow) float[n_elements];
    if (!data) {
        std::cerr << "Error allocating " << n_elements * sizeof(float) / (1024*1024) 
                  << " MB" << std::endl;
        throw std::bad_alloc();
    }

    // Reset file state and seek to data start
    file.clear(); 
    file.seekg(3 * sizeof(size_t), std::ios::beg);

    file.read(reinterpret_cast<char*>(data), n_elements * sizeof(float));
    if (!file) {
        std::cerr << "Error reading frame data from file" << std::endl;
        std::cerr << "Attempted to read " << n_elements * sizeof(float) / (1024*1024) 
                  << " MB" << std::endl;
        std::cerr << "File state - fail: " << file.fail() 
                  << ", eof: " << file.eof() 
                  << ", bad: " << file.bad() << std::endl;
        delete[] data;
        throw std::runtime_error("Failed to read frame data");
    }

    std::cout << "Successfully loaded " << n_elements * sizeof(float) / (1024*1024) 
              << " MB (" << n_subset_frames << " frames)" << std::endl;

    return data;
}

// Frame :
// Frame0: [atom0_x, atom0_y, atom0_z, atom1_x, atom1_y, atom1_z, ...]
// Frame1: [atom0_x, atom0_y, atom0_z, atom1_x, atom1_y, atom1_z, ...]
// On utilise le fait que la matrice soit triangulaire supérieure, donc col >= row
float* FileUtils::getFrameSubset(float* frames, int row_begin, int row_end, int col_begin, int col_end, size_t N_frames) {

    float* frame_subset = nullptr;

    // Cas où les frames de la colonne et de la ligne sont les memes.
    if(row_begin == col_begin) {
        int subset_size = row_end - row_begin;
        int frame_arr_size = n_dims * n_atoms;
        frame_subset = new (std::nothrow) float[subset_size*frame_arr_size];

        for(int i=row_begin; i < row_end; ++i) {
            for(int j=0; j < frame_arr_size; ++j) {
                frame_subset[((i - row_begin)*frame_arr_size) + j] = frames[ (i * frame_arr_size) + j ];
            }
        }
    }

    // 2 ensembles disjoints de frames entre les lignes et les colonnes
    else {
        int subset_size = (row_end - row_begin) + (col_end - col_begin);
        int frame_arr_size = n_dims * n_atoms;
        frame_subset = new (std::nothrow) float[subset_size*frame_arr_size];

        for(int i=row_begin; i < row_end; ++i) {
            for(int j=0; j < frame_arr_size; ++j) {
                frame_subset[((i - row_begin)*frame_arr_size) + j] = frames[ (i * frame_arr_size) + j ];
            }
        }

        for(int i=col_begin; i < col_end; ++i) {
            for(int j=0; j < frame_arr_size; ++j) {
                frame_subset[((i + row_end - row_begin - col_begin)*frame_arr_size) + j] = frames[ (i * frame_arr_size) + j];
            }
        }

    }

    return frame_subset;
}

/*
* Writes clustering result's centroid and cluster indices into a binary file.
* These indices are used to read actual molecule shapes 
* stored in the original dataset file.
* The file layout is as follows:
* - Metadata: sizeof(int) bytes storing K | sizeof(int) bytes storing N_frames
* - Data: K*sizeof(int) bytes storing centroids indices | N_frames*sizeof(int) bytes storing cluster
* indices of each frame
*/
void saveClusters(const int* clusters, int N_frames, const int* centroids, int K) {
    std::ofstream outFile("output/clusters.bin", std::ios::binary);
    
    if (outFile.is_open()) {
        // metadata
        outFile.write(reinterpret_cast<const char*>(&K), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&N_frames), sizeof(int));

        // Write the arrays
        outFile.write(reinterpret_cast<const char*>(centroids), K * sizeof(int));
        outFile.write(reinterpret_cast<const char*>(clusters), N_frames * sizeof(int));
        
        outFile.close();
    }

    std::cout << "Results saved to output/clusters.bin\n";
}

/*
* Loads cluster labels from the clusters.bin file.
* returns a vector<int> of size N_frames that was saved in the file.
*/
std::vector<int> loadClusterLabels() {
    std::ifstream inFile("output/clusters.bin", std::ios::binary);
    if (!inFile) return {};

    int K, N_frames;

    // Read metadata
    inFile.read(reinterpret_cast<char*>(&K), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&N_frames), sizeof(int));

    // Skip centroids
    inFile.seekg(K * sizeof(int), std::ios::cur);

    // Read labels
    std::vector<int> labels(N_frames);
    inFile.read(reinterpret_cast<char*>(labels.data()),
                N_frames * sizeof(int));

    return labels;
}

/*
* Loads cluster centroids from the clusters.bin file.
* returns a vector<int> of size K that was saved in the file.
*/
std::vector<int> loadClusterCentroids() {
    std::ifstream inFile("output/clusters.bin", std::ios::binary);
    if (!inFile) return {};

    int K, N_frames;

    // Read metadata
    inFile.read(reinterpret_cast<char*>(&K), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&N_frames), sizeof(int));

    // Read centroids
    std::vector<int> centroids(K);
    inFile.read(reinterpret_cast<char*>(centroids.data()),
                K * sizeof(int));

    return centroids;
}