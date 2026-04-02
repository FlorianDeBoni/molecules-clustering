#pragma once

#include <ostream>
#include <fstream>
#include <vector>
#include <iostream>

class FileUtils {
private:
    size_t n_snapshots;
    size_t n_atoms;
    size_t n_dims;
    friend std::ostream& operator<<(std::ostream& os, const FileUtils& f);
    std::ifstream file;
    std::string file_name;

public:
    FileUtils(const std::string& file_name);
    size_t getN_atoms() const;
    size_t getN_snapshots() const;
    size_t getN_dims() const;
    std::ifstream& getFile() const;
    
    void readSnapshotsFastInPlace(size_t start, size_t end, std::vector<float>& result);
    void extractSnapshotsFastInPlace(size_t start, size_t end, const std::vector<float>& all_data, std::vector<float>& result);

    float* getFrameSubset(float* frames, int row_begin, int row_end, int col_begin, int col_end, size_t N_frames);
};

std::vector<int> loadClusterLabels();
std::vector<int> loadClusterCentroids();