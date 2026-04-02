// FileUtils.cpp
#include "FileUtils.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string.h>
#include <cstring>

FileUtils::FileUtils(const std::string& file_name)
    : file_name(file_name), file(file_name, std::ios::binary)
{
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + file_name);

    file.read(reinterpret_cast<char*>(&n_snapshots), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_atoms),     sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_dims),      sizeof(size_t));

    if (!file)
        throw std::runtime_error("Error reading header from " + file_name);

    std::cout << "Loaded binary file: " << n_snapshots << " snapshots, "
              << n_atoms << " atoms, " << n_dims << " dimensions" << std::endl;
}

size_t FileUtils::getN_atoms()     const { return n_atoms;     }
size_t FileUtils::getN_snapshots() const { return n_snapshots; }
size_t FileUtils::getN_dims()      const { return n_dims;      }
std::ifstream& FileUtils::getFile() const { return const_cast<std::ifstream&>(file); }

std::ostream& operator<<(std::ostream& os, const FileUtils& f)
{
    size_t n_atoms     = f.getN_atoms();
    size_t n_dims      = f.getN_dims();
    size_t n_snapshots = f.getN_snapshots();
    std::ifstream& file = f.getFile();

    const size_t header_size = 3 * sizeof(size_t);
    const size_t block_size  = n_atoms * n_snapshots;

    if (n_dims != 3) {
        os << "Unsupported dimension count: " << n_dims << std::endl;
        return os;
    }

    auto print_snapshot = [&](size_t frame_index, const std::string& title)
    {
        std::vector<float> snapshot_data(n_atoms * 3);

        for (size_t a = 0; a < n_atoms; ++a) {
            for (size_t c = 0; c < 3; ++c) {
                size_t idx        = c * block_size + a * n_snapshots + frame_index;
                std::streampos pos = header_size + idx * sizeof(float);

                file.clear();
                file.seekg(pos, std::ios::beg);

                float value;
                file.read(reinterpret_cast<char*>(&value), sizeof(float));
                if (!file) { os << "Error reading snapshot data" << std::endl; return; }

                snapshot_data[a * 3 + c] = value;
            }
        }

        os << title << "\n"
           << "Atom 1: (" << snapshot_data[0] << ", "
                          << snapshot_data[1] << ", "
                          << snapshot_data[2] << ")\n"
           << "Atom 2: (" << snapshot_data[3] << ", "
                          << snapshot_data[4] << ", "
                          << snapshot_data[5] << ")\n"
           << "...\n"
           << "Atom " << n_atoms << ": ("
           << snapshot_data[(n_atoms-1)*3+0] << ", "
           << snapshot_data[(n_atoms-1)*3+1] << ", "
           << snapshot_data[(n_atoms-1)*3+2] << ")\n\n";
    };

    if (n_snapshots > 0) print_snapshot(0, "Snapshot 1:");
    if (n_snapshots > 1) { print_snapshot(1, "Snapshot 2:"); os << "...\n\n"; }
    if (n_snapshots > 2) print_snapshot(n_snapshots-1, "Snapshot " + std::to_string(n_snapshots) + ":");

    return os;
}

std::vector<int> loadClusterLabels()
{
    std::ifstream inFile("output/clusters.bin", std::ios::binary);
    if (!inFile) return {};

    int K, N_frames;
    inFile.read(reinterpret_cast<char*>(&K),        sizeof(int));
    inFile.read(reinterpret_cast<char*>(&N_frames), sizeof(int));
    inFile.seekg(K * sizeof(int), std::ios::cur);   // skip centroids

    std::vector<int> labels(N_frames);
    inFile.read(reinterpret_cast<char*>(labels.data()), N_frames * sizeof(int));
    return labels;
}

std::vector<int> loadClusterCentroids()
{
    std::ifstream inFile("output/clusters.bin", std::ios::binary);
    if (!inFile) return {};

    int K, N_frames;
    inFile.read(reinterpret_cast<char*>(&K),        sizeof(int));
    inFile.read(reinterpret_cast<char*>(&N_frames), sizeof(int));

    std::vector<int> centroids(K);
    inFile.read(reinterpret_cast<char*>(centroids.data()), K * sizeof(int));
    return centroids;
}

// ---------------------------------------------------------------------------
// readSnapshotsFastInPlace
//   start : first frame index (inclusive)
//   end   : last  frame index (inclusive)
//   Reads frames [start, end] into result with layout:
//     result[c * n_atoms * n_frames + a * n_frames + f]
// ---------------------------------------------------------------------------
void FileUtils::readSnapshotsFastInPlace(size_t start, size_t end,
                                         std::vector<float>& result)
{
    if (start > end || end >= n_snapshots)
        throw std::out_of_range("readSnapshotsFastInPlace: invalid range ["
                                + std::to_string(start) + ", "
                                + std::to_string(end) + "] for "
                                + std::to_string(n_snapshots) + " snapshots");
    if (n_dims != 3)
        throw std::runtime_error("Unsupported dimension count");

    size_t n_frames    = end - start + 1;          // inclusive count
    size_t block_size  = n_atoms * n_snapshots;
    const size_t header_size = 3 * sizeof(size_t);

    result.resize(n_frames * n_atoms * 3);

    std::vector<float> coord_block(block_size);

    for (size_t c = 0; c < 3; ++c) {
        std::streampos pos = header_size + c * block_size * sizeof(float);
        file.clear();
        file.seekg(pos, std::ios::beg);
        file.read(reinterpret_cast<char*>(coord_block.data()),
                  block_size * sizeof(float));
        if (!file)
            throw std::runtime_error("Error reading coordinate block");

        for (size_t a = 0; a < n_atoms; ++a) {
            for (size_t f = 0; f < n_frames; ++f) {
                size_t idx_file = a * n_snapshots + (start + f);
                size_t idx_out  = c * (n_atoms * n_frames) + a * n_frames + f;
                result[idx_out] = coord_block[idx_file];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// extractSnapshotsFastInPlace
//   start : first frame index (inclusive)
//   end   : one-past-last frame index (EXCLUSIVE)  → n_frames = end - start
//
//   Slices frames [start, end) from all_data (which holds N_total frames
//   with stride n_snapshots_data = all_data.size()/(3*n_atoms)) into result
//   with the same layout convention:
//     result[c * n_atoms * n_frames + a * n_frames + f]
// ---------------------------------------------------------------------------
void FileUtils::extractSnapshotsFastInPlace(
    size_t start,
    size_t end,                      // EXCLUSIVE
    const std::vector<float>& all_data,
    std::vector<float>& result)
{
    // Derive total frames stored in all_data from its size.
    const size_t n_snapshots_data = all_data.size() / (3 * n_atoms);

    if (start > end || end > n_snapshots_data)
        throw std::out_of_range("extractSnapshotsFastInPlace: invalid range ["
                                + std::to_string(start) + ", "
                                + std::to_string(end) + ") for "
                                + std::to_string(n_snapshots_data) + " frames");
    if (n_dims != 3)
        throw std::runtime_error("Unsupported dimension count");

    const size_t n_extracted = end - start;   // number of frames to extract

    result.resize(n_extracted * n_atoms * 3);

    for (size_t c = 0; c < 3; ++c) {
        // Base pointer for this coordinate dimension in all_data,
        // offset to the first requested frame.
        const float* src_base =
            all_data.data() + c * (n_atoms * n_snapshots_data) + start;

        float* dst_base =
            result.data() + c * (n_atoms * n_extracted);

        for (size_t a = 0; a < n_atoms; ++a) {
            const float* src = src_base + a * n_snapshots_data;
            float*       dst = dst_base + a * n_extracted;
            memcpy(dst, src, n_extracted * sizeof(float));
        }
    }
}
