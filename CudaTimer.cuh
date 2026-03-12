#pragma once
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdio.h>

struct CudaTimer {
    struct Measure {
        cudaEvent_t start, stop;
        std::string label;
        bool stopped = false;
    };

    std::vector<Measure> measures;  // preserves insertion order for printing
    std::unordered_map<std::string, size_t> index;

    void start(const std::string& label) {
        if (index.count(label)) {
            // Reuse existing events (e.g. in a loop)
            auto& m = measures[index[label]];
            m.stopped = false;
            cudaEventRecord(m.start);
            return;
        }
        Measure m;
        m.label = label;
        cudaEventCreate(&m.start);
        cudaEventCreate(&m.stop);
        cudaEventRecord(m.start);
        index[label] = measures.size();
        measures.push_back(m);
    }

    void stop(const std::string& label) {
        auto& m = measures[index[label]];
        cudaEventRecord(m.stop);
        m.stopped = true;
    }

    void print() {
        cudaDeviceSynchronize();
        printf("\n%-30s %10s\n", "Stage", "Time (s)");
        printf("%s\n", std::string(42, '-').c_str());
        for (auto& m : measures) {
            if (!m.stopped) continue;
            float ms = 0;
            cudaEventElapsedTime(&ms, m.start, m.stop);
            printf("%-30s %10.3f s\n", m.label.c_str(), ms/1000);
        }
    }

    ~CudaTimer() {
        for (auto& m : measures) {
            cudaEventDestroy(m.start);
            cudaEventDestroy(m.stop);
        }
    }
};