// /*
// Compile with:
// g++ -std=c++17 -O3 md_reader_optimized.cpp \
//     -I/usr/users/gpumol/deboni_flo/Bureau/include \
//     -L/usr/users/gpumol/deboni_flo/Bureau/lib -lchemfiles \
//     -o md_reader

// Run:
// ./md_reader

// Optimized for: 200,000 snapshots × 4,600 atoms
// Strategy: Process trajectories in chunks, write directly to file positions
// */

// #include <chemfiles.hpp>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <filesystem>
// #include <algorithm>
// #include <cstring>

// namespace fs = std::filesystem;

// constexpr size_t CHUNK_SIZE = 10000; // Process 10k frames at a time

// // ------------------------------------------------------------
// // Count snapshots in a trajectory
// // ------------------------------------------------------------
// size_t count_snapshots(const std::string &trajectory_file,
//                        const chemfiles::Topology &topology)
// {
//     chemfiles::Trajectory traj(trajectory_file, 'r');
//     traj.set_topology(topology);
//     size_t count = 0;
//     while (!traj.done())
//     {
//         traj.read();
//         ++count;
//     }
//     return count;
// }

// // ------------------------------------------------------------
// // Write trajectory chunk with atom-major layout
// // Uses direct file positioning for each atom's data
// // ------------------------------------------------------------
// void write_trajectory_chunk(const std::string &trajectory_file,
//                             const chemfiles::Topology &topology,
//                             const std::string &output_file,
//                             size_t n_atoms,
//                             size_t total_snapshots,
//                             size_t start_frame,
//                             size_t header_size)
// {
//     chemfiles::Trajectory traj(trajectory_file, 'r');
//     traj.set_topology(topology);

//     // Allocate buffers for this chunk
//     std::vector<std::vector<float>> chunkX(n_atoms);
//     std::vector<std::vector<float>> chunkY(n_atoms);
//     std::vector<std::vector<float>> chunkZ(n_atoms);
    
//     for (size_t a = 0; a < n_atoms; ++a)
//     {
//         chunkX[a].reserve(CHUNK_SIZE);
//         chunkY[a].reserve(CHUNK_SIZE);
//         chunkZ[a].reserve(CHUNK_SIZE);
//     }

//     size_t frames_read = 0;
    
//     // Read trajectory into chunk buffers
//     while (!traj.done())
//     {
//         auto snapshot = traj.read();
//         auto pos = snapshot.positions();
        
//         if (pos.size() != n_atoms)
//             throw std::runtime_error("Atom count mismatch");

//         for (size_t a = 0; a < n_atoms; ++a)
//         {
//             chunkX[a].push_back((float)pos[a][0]);
//             chunkY[a].push_back((float)pos[a][1]);
//             chunkZ[a].push_back((float)pos[a][2]);
//         }
        
//         ++frames_read;
//     }

//     if (frames_read == 0)
//         return;

//     // Now write to file at correct positions
//     std::fstream outfile(output_file, std::ios::binary | std::ios::in | std::ios::out);
//     if (!outfile.is_open())
//         throw std::runtime_error("Cannot open output file for writing chunk");

//     // Calculate byte offsets
//     // Layout: [header] [all X coords] [all Y coords] [all Z coords]
//     // Each coordinate section: atom0_all_frames, atom1_all_frames, ...
    
//     size_t bytes_per_atom = total_snapshots * sizeof(float);
//     size_t x_section_start = header_size;
//     size_t y_section_start = x_section_start + n_atoms * bytes_per_atom;
//     size_t z_section_start = y_section_start + n_atoms * bytes_per_atom;

//     // Write each atom's data at the correct position
//     for (size_t a = 0; a < n_atoms; ++a)
//     {
//         // Write X coordinates for this atom
//         size_t x_offset = x_section_start + a * bytes_per_atom + start_frame * sizeof(float);
//         outfile.seekp(x_offset);
//         outfile.write(reinterpret_cast<char*>(chunkX[a].data()), 
//                      frames_read * sizeof(float));

//         // Write Y coordinates for this atom
//         size_t y_offset = y_section_start + a * bytes_per_atom + start_frame * sizeof(float);
//         outfile.seekp(y_offset);
//         outfile.write(reinterpret_cast<char*>(chunkY[a].data()), 
//                      frames_read * sizeof(float));

//         // Write Z coordinates for this atom
//         size_t z_offset = z_section_start + a * bytes_per_atom + start_frame * sizeof(float);
//         outfile.seekp(z_offset);
//         outfile.write(reinterpret_cast<char*>(chunkZ[a].data()), 
//                      frames_read * sizeof(float));
//     }

//     outfile.close();
// }

// // ------------------------------------------------------------
// // Main
// // ------------------------------------------------------------
// int main()
// {
//     try
//     {
//         std::string dataset_root = "./dataset";
//         std::string output_file = "output/snapshots_coords_all.bin";

//         fs::create_directories(fs::path(output_file).parent_path());

//         std::vector<fs::path> subdirs;
//         for (auto &entry : fs::directory_iterator(dataset_root))
//             if (entry.is_directory())
//                 subdirs.push_back(entry.path());

//         std::sort(subdirs.begin(), subdirs.end());

//         if (subdirs.empty())
//         {
//             std::cout << "Warning: no subdirectories found\n";
//             return 1;
//         }

//         // ---------------- FIRST PASS: count frames & atoms ----------------
//         std::cout << "=== PASS 1: Counting frames and atoms ===\n";
        
//         size_t total_snapshots = 0;
//         size_t n_atoms = 0;
        
//         struct TrajectoryInfo {
//             fs::path subdir;
//             fs::path pdb_file;
//             fs::path xtc_file;
//             size_t n_frames;
//         };
        
//         std::vector<TrajectoryInfo> trajectories;

//         for (auto &subdir : subdirs)
//         {
//             fs::path pdb_file = subdir / "minimal.pdb";
//             fs::path xtc_file = subdir / "minimal.xtc";

//             if (!fs::exists(pdb_file) || !fs::exists(xtc_file))
//             {
//                 std::cout << "Skipping " << subdir.filename() << " (missing files)\n";
//                 continue;
//             }

//             chemfiles::Trajectory top_traj(pdb_file.string(), 'r');
//             auto snapshot = top_traj.read();
//             auto topology = snapshot.topology();

//             if (n_atoms == 0)
//                 n_atoms = topology.size();
//             else if (n_atoms != topology.size())
//                 throw std::runtime_error("Atom count mismatch across trajectories");

//             size_t n = count_snapshots(xtc_file.string(), topology);
//             total_snapshots += n;
            
//             trajectories.push_back({subdir, pdb_file, xtc_file, n});

//             std::cout << "  " << subdir.filename() << ": " << n << " frames\n";
//         }

//         if (total_snapshots == 0)
//             throw std::runtime_error("No snapshots found");

//         std::cout << "\nTotal frames: " << total_snapshots << "\n";
//         std::cout << "Total atoms: " << n_atoms << "\n";
        
//         size_t header_size = 3 * sizeof(size_t);
//         size_t data_size = total_snapshots * n_atoms * 3 * sizeof(float);
//         double size_mb = (header_size + data_size) / (1024.0 * 1024.0);
//         double size_gb = size_mb / 1024.0;
        
//         std::cout << "Output file size: " << size_gb << " GB (" << size_mb << " MB)\n";
//         std::cout << "Memory per chunk: " << (3.0 * n_atoms * CHUNK_SIZE * sizeof(float)) / (1024.0 * 1024.0) << " MB\n\n";

//         // ---------------- Allocate output file ----------------
//         std::cout << "=== Allocating output file ===\n";
        
//         std::ofstream outfile(output_file, std::ios::binary | std::ios::trunc);
//         if (!outfile.is_open())
//             throw std::runtime_error("Cannot open output file");

//         // Pre-allocate file size
//         outfile.seekp(header_size + data_size - 1);
//         outfile.write("", 1);
//         outfile.flush();

//         // Write header
//         size_t n_dims = 3;
//         outfile.seekp(0);
//         outfile.write(reinterpret_cast<char*>(&total_snapshots), sizeof(size_t));
//         outfile.write(reinterpret_cast<char*>(&n_atoms), sizeof(size_t));
//         outfile.write(reinterpret_cast<char*>(&n_dims), sizeof(size_t));
//         outfile.close();
        
//         std::cout << "File allocated successfully\n\n";

//         // ---------------- SECOND PASS: write trajectories ----------------
//         std::cout << "=== PASS 2: Writing trajectory data ===\n";
        
//         size_t global_frame = 0;
        
//         for (auto &traj_info : trajectories)
//         {
//             std::cout << "Processing " << traj_info.subdir.filename() 
//                       << " (" << traj_info.n_frames << " frames, starting at frame " 
//                       << global_frame << ")..." << std::flush;

//             chemfiles::Trajectory top_traj(traj_info.pdb_file.string(), 'r');
//             auto snapshot = top_traj.read();
//             auto topology = snapshot.topology();

//             write_trajectory_chunk(
//                 traj_info.xtc_file.string(),
//                 topology,
//                 output_file,
//                 n_atoms,
//                 total_snapshots,
//                 global_frame,
//                 header_size);

//             global_frame += traj_info.n_frames;
            
//             std::cout << " done\n";
//         }

//         std::cout << "\n=== Complete ===\n";
//         std::cout << "Output file: " << output_file << "\n";
//         std::cout << "Total frames written: " << global_frame << "\n";
//         std::cout << "File size: " << size_gb << " GB\n";
        
//         // Verify file size
//         size_t actual_size = fs::file_size(output_file);
//         size_t expected_size = header_size + data_size;
//         if (actual_size == expected_size)
//         {
//             std::cout << "✓ File size verification: PASSED\n";
//         }
//         else
//         {
//             std::cout << "✗ File size verification: FAILED\n";
//             std::cout << "  Expected: " << expected_size << " bytes\n";
//             std::cout << "  Actual: " << actual_size << " bytes\n";
//         }

//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "\nError: " << e.what() << "\n";
//         return 1;
//     }

//     return 0;
// }

/*
Compile with:
g++ -std=c++17 -O3 md_reader_optimized.cpp \
    -I/usr/users/gpumol/deboni_flo/Bureau/include \
    -L/usr/users/gpumol/deboni_flo/Bureau/lib -lchemfiles \
    -o md_reader

Run:
./md_reader

Optimized for: 200,000 snapshots × 4,600 atoms
Strategy: Process trajectories in chunks, write directly to file positions
*/

#include <chemfiles.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

constexpr size_t CHUNK_SIZE = 10000; // Process 10k frames at a time

// ------------------------------------------------------------
// Count snapshots in a trajectory
// ------------------------------------------------------------
size_t count_snapshots(const std::string &trajectory_file,
                       const chemfiles::Topology &topology)
{
    chemfiles::Trajectory traj(trajectory_file, 'r');
    traj.set_topology(topology);
    size_t count = 0;
    while (!traj.done())
    {
        traj.read();
        ++count;
    }
    return count;
}

// ------------------------------------------------------------
// Write trajectory chunk with atom-major layout
// Uses direct file positioning for each atom's data
// ------------------------------------------------------------
void write_trajectory_chunk(const std::string &trajectory_file,
                            const chemfiles::Topology &topology,
                            const std::string &output_file,
                            size_t n_atoms,
                            size_t total_snapshots,
                            size_t start_frame,
                            size_t header_size)
{
    chemfiles::Trajectory traj(trajectory_file, 'r');
    traj.set_topology(topology);

    // Allocate buffers for this chunk
    std::vector<std::vector<float>> chunkX(n_atoms);
    std::vector<std::vector<float>> chunkY(n_atoms);
    std::vector<std::vector<float>> chunkZ(n_atoms);
    
    for (size_t a = 0; a < n_atoms; ++a)
    {
        chunkX[a].reserve(CHUNK_SIZE);
        chunkY[a].reserve(CHUNK_SIZE);
        chunkZ[a].reserve(CHUNK_SIZE);
    }

    size_t frames_read = 0;
    
    // Read trajectory into chunk buffers
    while (!traj.done())
    {
        auto snapshot = traj.read();
        auto pos = snapshot.positions();
        
        if (pos.size() != n_atoms)
            throw std::runtime_error("Atom count mismatch");

        for (size_t a = 0; a < n_atoms; ++a)
        {
            chunkX[a].push_back((float)pos[a][0]);
            chunkY[a].push_back((float)pos[a][1]);
            chunkZ[a].push_back((float)pos[a][2]);
        }
        
        ++frames_read;
    }

    if (frames_read == 0)
        return;

    // Now write to file at correct positions
    std::fstream outfile(output_file, std::ios::binary | std::ios::in | std::ios::out);
    if (!outfile.is_open())
        throw std::runtime_error("Cannot open output file for writing chunk");

    // Calculate byte offsets
    // Layout: [header] [all X coords] [all Y coords] [all Z coords]
    // Each coordinate section: atom0_all_frames, atom1_all_frames, ...
    
    size_t bytes_per_atom = total_snapshots * sizeof(float);
    size_t x_section_start = header_size;
    size_t y_section_start = x_section_start + n_atoms * bytes_per_atom;
    size_t z_section_start = y_section_start + n_atoms * bytes_per_atom;

    // Write each atom's data at the correct position
    for (size_t a = 0; a < n_atoms; ++a)
    {
        // Write X coordinates for this atom
        size_t x_offset = x_section_start + a * bytes_per_atom + start_frame * sizeof(float);
        outfile.seekp(x_offset);
        outfile.write(reinterpret_cast<char*>(chunkX[a].data()), 
                     frames_read * sizeof(float));

        // Write Y coordinates for this atom
        size_t y_offset = y_section_start + a * bytes_per_atom + start_frame * sizeof(float);
        outfile.seekp(y_offset);
        outfile.write(reinterpret_cast<char*>(chunkY[a].data()), 
                     frames_read * sizeof(float));

        // Write Z coordinates for this atom
        size_t z_offset = z_section_start + a * bytes_per_atom + start_frame * sizeof(float);
        outfile.seekp(z_offset);
        outfile.write(reinterpret_cast<char*>(chunkZ[a].data()), 
                     frames_read * sizeof(float));
    }

    outfile.close();
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main()
{
    try
    {
        std::string dataset_root = "./dataset";
        std::string output_file = "output/snapshots_coords_20811.bin";

        fs::create_directories(fs::path(output_file).parent_path());

        std::vector<fs::path> subdirs;
        for (auto &entry : fs::directory_iterator(dataset_root))
            if (entry.is_directory())
                subdirs.push_back(entry.path());

        std::sort(subdirs.begin(), subdirs.end());

        if (subdirs.empty())
        {
            std::cout << "Warning: no subdirectories found\n";
            return 1;
        }

        // ---------------- FIRST PASS: count frames & atoms ----------------
        std::cout << "=== PASS 1: Counting frames and atoms ===\n";
        
        size_t total_snapshots = 0;
        size_t n_atoms = 0;
        
        struct TrajectoryInfo {
            fs::path subdir;
            fs::path pdb_file;
            fs::path xtc_file;
            size_t n_frames;
        };
        
        std::vector<TrajectoryInfo> trajectories;

        for (auto &subdir : subdirs)
        {
            fs::path pdb_file = subdir / "minimal.pdb";
            fs::path xtc_file = subdir / "minimal.xtc";

            if (!fs::exists(pdb_file) || !fs::exists(xtc_file))
            {
                std::cout << "Skipping " << subdir.filename() << " (missing files)\n";
                continue;
            }

            chemfiles::Trajectory top_traj(pdb_file.string(), 'r');
            auto snapshot = top_traj.read();
            auto topology = snapshot.topology();

            if (n_atoms == 0)
                n_atoms = topology.size();
            else if (n_atoms != topology.size())
                throw std::runtime_error("Atom count mismatch across trajectories");

            size_t n = count_snapshots(xtc_file.string(), topology);
            total_snapshots += n;
            
            trajectories.push_back({subdir, pdb_file, xtc_file, n});

            std::cout << "  " << subdir.filename() << ": " << n << " frames\n";
        }

        if (total_snapshots == 0)
            throw std::runtime_error("No snapshots found");

        std::cout << "\nTotal frames: " << total_snapshots << "\n";
        std::cout << "Total atoms: " << n_atoms << "\n";
        
        size_t header_size = 3 * sizeof(size_t);
        size_t data_size = total_snapshots * n_atoms * 3 * sizeof(float);
        double size_mb = (header_size + data_size) / (1024.0 * 1024.0);
        double size_gb = size_mb / 1024.0;
        
        std::cout << "Output file size: " << size_gb << " GB (" << size_mb << " MB)\n";
        std::cout << "Memory per chunk: " << (3.0 * n_atoms * CHUNK_SIZE * sizeof(float)) / (1024.0 * 1024.0) << " MB\n\n";

        // ---------------- Allocate output file ----------------
        std::cout << "=== Allocating output file ===\n";
        
        std::ofstream outfile(output_file, std::ios::binary | std::ios::trunc);
        if (!outfile.is_open())
            throw std::runtime_error("Cannot open output file");

        // Pre-allocate file size
        outfile.seekp(header_size + data_size - 1);
        outfile.write("", 1);
        outfile.flush();

        // Write header
        size_t n_dims = 3;
        outfile.seekp(0);
        outfile.write(reinterpret_cast<char*>(&total_snapshots), sizeof(size_t));
        outfile.write(reinterpret_cast<char*>(&n_atoms), sizeof(size_t));
        outfile.write(reinterpret_cast<char*>(&n_dims), sizeof(size_t));
        outfile.close();
        
        std::cout << "File allocated successfully\n\n";

        // ---------------- SECOND PASS: write trajectories ----------------
        std::cout << "=== PASS 2: Writing trajectory data ===\n";
        
        size_t global_frame = 0;
        
        for (auto &traj_info : trajectories)
        {
            std::cout << "Processing " << traj_info.subdir.filename() 
                      << " (" << traj_info.n_frames << " frames, starting at frame " 
                      << global_frame << ")..." << std::flush;

            chemfiles::Trajectory top_traj(traj_info.pdb_file.string(), 'r');
            auto snapshot = top_traj.read();
            auto topology = snapshot.topology();

            write_trajectory_chunk(
                traj_info.xtc_file.string(),
                topology,
                output_file,
                n_atoms,
                total_snapshots,
                global_frame,
                header_size);

            global_frame += traj_info.n_frames;
            
            std::cout << " done\n";
        }

        std::cout << "\n=== Complete ===\n";
        std::cout << "Output file: " << output_file << "\n";
        std::cout << "Total frames written: " << global_frame << "\n";
        std::cout << "File size: " << size_gb << " GB\n";
        
        // Verify file size
        size_t actual_size = fs::file_size(output_file);
        size_t expected_size = header_size + data_size;
        if (actual_size == expected_size)
        {
            std::cout << "✓ File size verification: PASSED\n";
        }
        else
        {
            std::cout << "✗ File size verification: FAILED\n";
            std::cout << "  Expected: " << expected_size << " bytes\n";
            std::cout << "  Actual: " << actual_size << " bytes\n";
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }

    return 0;
}