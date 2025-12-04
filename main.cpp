#include "utils.h"
#include "lib/cnpy.h"
#include <iostream>
#include <vector>

using namespace std;


int main() {
    const string filename = "snapshots_coords.npy";

    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        cout << "Successfully loaded file: " << filename << "\n";

        // Data pointer
        float* data_ptr = arr.data<float>();
        
        size_t num_snapshots = arr.shape[0]; // 30003
        size_t num_atoms = arr.shape[1];     // 4366
        size_t num_dims = arr.shape[2];      // 3

        size_t molecule_id = 0;

        for (size_t i = 0; i < num_atoms; i++) {
            float x = data_ptr[i * (num_atoms * num_dims) + num_dims*0];
            float y = data_ptr[i * (num_atoms * num_dims) + num_dims*1];
            float z = data_ptr[i * (num_atoms * num_dims) + num_dims*2];
            std::cout << "Atom " << i << ": X=" << x << ", Y=" << y << ", Z=" << z << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error loading or processing NPY file: " << e.what() << "\n";
        return 1;
    }

    return 0;
}