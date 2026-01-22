/*
    Compile with:
    g++ -std=c++17 -O3 md_reader.cpp -lchemfiles -o md_reader

    Run:
    ./md_reader
*/

#include <chemfiles.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

struct TrajectoryInfo {
    size_t n_frames;
    size_t n_atoms;
    std::streampos data_start_pos;  // Track where data was written
};

// Read trajectory and write directly to file (streaming approach)
TrajectoryInfo write_trajectory_to_file(const std::string& trajectory_file,
                                         const chemfiles::Topology& topology,
                                         std::ofstream& outfile) {
    TrajectoryInfo info = {0, 0, outfile.tellp()};
    
    try {
        chemfiles::Trajectory traj(trajectory_file, 'r');
        traj.set_topology(topology);
        
        while (!traj.done()) {
            auto frame = traj.read();
            auto positions = frame.positions();
            
            if (info.n_frames == 0) {
                info.n_atoms = positions.size();
                std::cout << "  Atoms in trajectory: " << info.n_atoms << std::endl;
            }
            
            // Verify atom count consistency within trajectory
            if (positions.size() != info.n_atoms) {
                throw std::runtime_error("Inconsistent atom count within trajectory at frame " + 
                                       std::to_string(info.n_frames));
            }
            
            for (size_t j = 0; j < positions.size(); j++) {
                float coords[3] = {
                    static_cast<float>(positions[j][0]), 
                    static_cast<float>(positions[j][1]), 
                    static_cast<float>(positions[j][2])
                };
                outfile.write(reinterpret_cast<const char*>(coords), 3 * sizeof(float));
            }
            
            info.n_frames++;
            
            if (info.n_frames % 1000 == 0) {
                std::cout << "  Progress: " << info.n_frames << " frames" << std::endl;
                std::cout.flush();
            }
        }
        
        std::cout << "  Completed: " << info.n_frames << " frames" << std::endl;
        
    } catch (const chemfiles::Error& e) {
        throw std::runtime_error("Chemfiles error reading " + trajectory_file + ": " + std::string(e.what()));
    }
    
    return info;
}

int main() {
    try {
        std::string dataset_root = "./dataset";
        std::string output_file = "output/snapshots_coords_all.bin";

        // Ensure output folder exists
        fs::create_directories(fs::path(output_file).parent_path());

        std::ofstream outfile(output_file, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }

        // Reserve space for header (3 size_t values)
        size_t total_frames = 0;
        size_t n_atoms = 0;
        size_t n_dims = 3;

        std::streampos header_pos = outfile.tellp();
        outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_atoms), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_dims), sizeof(size_t));

        std::cout << "Scanning dataset folder: " << dataset_root << std::endl;

        // Collect and sort subdirectories for consistent ordering
        std::vector<fs::path> subdirs;
        for (const auto& entry : fs::directory_iterator(dataset_root)) {
            if (entry.is_directory()) {
                subdirs.push_back(entry.path());
            }
        }
        std::sort(subdirs.begin(), subdirs.end());

        if (subdirs.empty()) {
            std::cout << "Warning: No subdirectories found in " << dataset_root << std::endl;
        }

        for (const auto& subdir : subdirs) {
            fs::path pdb_file = subdir / "minimal.pdb";
            fs::path xtc_file = subdir / "minimal.xtc";

            if (!fs::exists(pdb_file) || !fs::exists(xtc_file)) {
                std::cout << "Skipping " << subdir.filename() 
                          << " (missing minimal.pdb or minimal.xtc)" << std::endl;
                continue;
            }

            std::cout << "\nProcessing folder: " << subdir.filename() << std::endl;

            // Read topology
            chemfiles::Trajectory topology_traj(pdb_file.string(), 'r', "PDB");
            auto topology_frame = topology_traj.read();
            auto topology = topology_frame.topology();

            std::cout << "  Topology atoms: " << topology.size() << std::endl;

            // Get current position before writing
            std::streampos before_write = outfile.tellp();

            try {
                auto info = write_trajectory_to_file(
                    xtc_file.string(),
                    topology,
                    outfile
                );

                // Verify atom count consistency across trajectories
                if (n_atoms == 0) {
                    n_atoms = info.n_atoms;
                    std::cout << "  Set reference atom count: " << n_atoms << std::endl;
                } else if (n_atoms != info.n_atoms) {
                    // Atom count mismatch - revert written data
                    std::cerr << "ERROR: Atom count mismatch in " << subdir.filename() 
                              << " (expected " << n_atoms << ", got " << info.n_atoms 
                              << "). Reverting data." << std::endl;
                    
                    // Seek back to position before this trajectory
                    outfile.seekp(before_write);
                    continue;
                }

                total_frames += info.n_frames;

            } catch (const std::exception& e) {
                std::cerr << "ERROR processing " << subdir.filename() << ": " 
                          << e.what() << ". Reverting data." << std::endl;
                // Seek back to position before this trajectory
                outfile.seekp(before_write);
                continue;
            }
        }

        // Verify we have data
        if (total_frames == 0 || n_atoms == 0) {
            throw std::runtime_error("No valid trajectory data was processed");
        }

        // Write final header
        outfile.seekp(header_pos);
        outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_atoms), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_dims), sizeof(size_t));

        outfile.close();

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total frames: " << total_frames << std::endl;
        std::cout << "Atoms per frame: " << n_atoms << std::endl;
        std::cout << "Shape: (" << total_frames << ", " << n_atoms << ", 3)" << std::endl;
        std::cout << "Output file: " << output_file << std::endl;

        // Verify file size
        std::ifstream check(output_file, std::ios::binary | std::ios::ate);
        size_t file_size = check.tellg();
        size_t expected_size = 3 * sizeof(size_t) + total_frames * n_atoms * 3 * sizeof(float);
        
        std::cout << "File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Expected size: " << (expected_size / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        if (file_size != expected_size) {
            std::cerr << "WARNING: File size mismatch! Data may be corrupted." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}