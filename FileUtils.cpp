// FileUtils.cpp
#include "FileUtils.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string.h>

FileUtils::FileUtils() 
    : file("output/snapshots_coords.bin", std::ios::binary) 
{
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output/snapshots_coords.bin");
    }

    // Read header information
    file.read(reinterpret_cast<char*>(&n_frames), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_atoms), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_dims), sizeof(size_t));

    if (!file) {
        throw std::runtime_error("Error reading header from snapshots_coords.bin");
    }
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
    file.seekg(3 * sizeof(size_t) + 0 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));
    
    os << "Snapshot 1:" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
    
    file.seekg(3 * sizeof(size_t) + 1 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    os << "Snapshot 2:" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
    os << "..." << std::endl << std::endl;

    file.seekg(3 * sizeof(size_t) + n_frames * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    os << "Snapshot " << n_frames << ":" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")";
    return os;
}

std::vector<float> FileUtils::readFrame(size_t frame_idx) {

    std::vector<float> frame_data(n_atoms * n_dims);
    file.seekg(3 * sizeof(size_t) + 0 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    return frame_data;
}

/*
* Reorders a flat frame_data array from
* (N * n * 3) to (N * 3 * n)
*/
void FileUtils::reorderByLine(float* frame_data, const size_t n_subset_frames) {

    // buffer
    std::vector<float> tmp(3 * n_atoms);

    for (size_t frame_idx = 0; frame_idx < n_subset_frames; frame_idx++){
        float* base = frame_data + frame_idx * n_atoms * 3;

        // copy original (N * n * 3) block
        memcpy(tmp.data(), base, 3 * n_atoms * sizeof(float));

        // now write in (N * 3 * n) order
        for (size_t a = 0; a < n_atoms; ++a) {
            base[a]             = tmp[3 * a + 0];
            base[a + n_atoms]   = tmp[3 * a + 1];
            base[a + 2*n_atoms] = tmp[3 * a + 2];
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
        std::cerr << "Error: number of frames requested " << n_subset_frames << " > " << n_frames << " in the file" << std::endl;
        exit(1);
    }
    size_t data_size_bytes = n_subset_frames * n_atoms * n_dims * sizeof(float);

    // no throw to handle error ourselves
    float* data = new (std::nothrow) float[data_size_bytes];

    if (data == nullptr) {
        std::cerr << "Error allocating " << data_size_bytes / (1000*1000) << " Mb" << std::endl;
        exit(1);
    }

    // skipping metadata 
    size_t offset = (3 * sizeof(size_t));
    file.seekg(offset, std::ios::beg);
    
    // extract data
    file.read(reinterpret_cast<char*>(data), data_size_bytes);

    std::cout << "Successfully Loaded " << data_size_bytes / (1000*1000) << " Mb" << std::endl;

    return data;
}