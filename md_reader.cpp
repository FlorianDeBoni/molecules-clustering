/*
    Compile with:
    g++ -std=c++11 -O3 md_reader.cpp -lchemfiles -o md_reader

    Run:
    ./md_reader
*/

#include <chemfiles.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

struct TrajectoryInfo {
    size_t n_frames;
    size_t n_atoms;
};

// Read trajectory and write directly to file (streaming approach)
TrajectoryInfo write_trajectory_to_file(const std::string& trajectory_file,
                                         chemfiles::Topology& topology,
                                         std::ofstream& outfile) {
    TrajectoryInfo info = {0, 0};
    
    try {
        // Open trajectory
        chemfiles::Trajectory traj(trajectory_file, 'r');
        traj.set_topology(topology);
        
        // Read and write frames one at a time
        while (!traj.done()) {
            auto frame = traj.read();
            auto positions = frame.positions();
            
            if (info.n_frames == 0) {
                info.n_atoms = positions.size();
            }
            
            // Write coordinates directly to file as float (4 bytes) instead of double (8 bytes)
            // This saves 50% space and memory bandwidth
            for (size_t j = 0; j < positions.size(); j++) {
                float coords[3] = {
                    static_cast<float>(positions[j][0]), 
                    static_cast<float>(positions[j][1]), 
                    static_cast<float>(positions[j][2])
                };
                outfile.write(reinterpret_cast<const char*>(coords), 3 * sizeof(float));
            }
            
            info.n_frames++;
            
            // Progress indicator every 1000 frames
            if (info.n_frames % 1000 == 0) {
                std::cout << "  Progress: " << info.n_frames << " frames" << std::endl;
                std::cout.flush();
            }
        }
        
        std::cout << "  Completed: " << info.n_frames << " frames" << std::endl;
    } catch (const chemfiles::Error& e) {
        throw std::runtime_error("Chemfiles error: " + std::string(e.what()));
    }
    
    return info;
}

int main() {
    try {
        std::string topology_file = "./dataset/1k5n_A.pdb";
        std::string output_file = "output/snapshots_coords.bin";
        
        // Load topology once
        std::cout << "Loading topology..." << std::endl;
        chemfiles::Trajectory topology_traj(topology_file, 'r', "PDB");
        auto topology_frame = topology_traj.read();
        auto topology = topology_frame.topology();
        std::cout << "Topology loaded: " << topology.size() << " atoms" << std::endl;
        
        // Create output file
        std::ofstream outfile(output_file, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }
        
        // Reserve space for header (will write later)
        size_t total_frames = 0;
        size_t n_atoms = 0;
        size_t header_pos = outfile.tellp();
        outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_atoms), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(size_t)); // placeholder
        
        // Process R1
        std::cout << "\nReading R1 trajectory..." << std::endl;
        auto info_R1 = write_trajectory_to_file("./dataset/1k5n_A_prod_R1_fit.xtc", topology, outfile);
        total_frames += info_R1.n_frames;
        n_atoms = info_R1.n_atoms;
        
        // Process R2
        std::cout << "\nReading R2 trajectory..." << std::endl;
        auto info_R2 = write_trajectory_to_file("./dataset/1k5n_A_prod_R2_fit.xtc", topology, outfile);
        total_frames += info_R2.n_frames;
        
        // Process R3
        std::cout << "\nReading R3 trajectory..." << std::endl;
        auto info_R3 = write_trajectory_to_file("./dataset/1k5n_A_prod_R3_fit.xtc", topology, outfile);
        total_frames += info_R3.n_frames;
        
        // Write header with final dimensions
        outfile.seekp(header_pos);
        outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(size_t));
        outfile.write(reinterpret_cast<const char*>(&n_atoms), sizeof(size_t));
        size_t n_dims = 3;
        outfile.write(reinterpret_cast<const char*>(&n_dims), sizeof(size_t));
        
        outfile.close();
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total frames: " << total_frames << std::endl;
        std::cout << "Atoms per frame: " << n_atoms << std::endl;
        std::cout << "Shape: (" << total_frames << ", " << n_atoms << ", 3)" << std::endl;
        std::cout << "Data saved to: " << output_file << std::endl;
        
        // File size info
        std::ifstream check(output_file, std::ios::binary | std::ios::ate);
        size_t file_size = check.tellg();
        std::cout << "File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "\nNote: Coordinates stored as float32 (4 bytes per value)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

