#pragma once

#include <ostream>
#include <fstream>
#include <vector>
#include <iostream>

class FileUtils {
private:
    size_t n_frames;
    size_t n_atoms;
    size_t n_dims;
    friend std::ostream& operator<<(std::ostream& os, const FileUtils& f);
    std::ifstream file;
    std::string file_name;

public:
    FileUtils(const std::string& file_name);
    size_t getN_atoms() const;
    size_t getN_frames() const;
    size_t getN_dims() const;
    std::ifstream& getFile() const;
    std::vector<float> readFrame(size_t frame_idx);
    float* loadData(size_t n_subset_frames);
    float* getFrameSubset(float* frames, int row_begin, int row_end, int col_begin, int col_end, size_t N_frames);
    void reorderByLine(float* frame_data, const size_t n_subset_frames);
};

// Cluster I/O
void saveClusters(const int* clusters, int N_frames,
                  const int* centroids, int K);

std::vector<int> loadClusterLabels();
std::vector<int> loadClusterCentroids();
