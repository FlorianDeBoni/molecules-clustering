// FileUtils.h

#ifndef FILEUTILS_H
#define FILEUTILS_H

#include <ostream>

#pragma once
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

public:
    FileUtils();
    size_t getN_atoms() const;
    size_t getN_frames() const;
    size_t getN_dims() const;
    std::ifstream& getFile() const;
    std::vector<float> readFrame(size_t frame_idx);
    float* loadData(size_t n_subset_frames);
    void reorderByLine(float* frame_data, const size_t n_subset_frames);
};

#endif