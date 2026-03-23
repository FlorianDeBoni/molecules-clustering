# Molecules Clustering
Clustering molecules dynamics on GPU

# Setup
This project uses `g++` and `nvcc` for compilation.  
In the resulting folder from cloning this repository, do the following:
### 1. Download the dataset
For the complete dataset of ~200 000 snapshots, download 4 zip files using the following links:  
- [subet_1](https://mdrepo.org/api/v1/get_download_instance/3257.zip)
- [subset_2](https://mdrepo.org/api/v1/get_download_instance/3258.zip)
- [subset_3](https://mdrepo.org/api/v1/get_download_instance/3259.zip)
- [subset_4](https://mdrepo.org/api/v1/get_download_instance/3260.zip)  
  
Then unzip them into a `dataset/` folder at the root of the project.  
[For more info on the data](https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/1k5n_A/1k5n_A.html)

### 2. Install cmake
Make sure to have cmake installed, it's used to build the [chemfiles](https://github.com/chemfiles/chemfiles) library locally to read the raw data.  
For linux:
```bash
sudo apt install cmake
```

### 3. Run make
To run the clustering using the GPU, the default make command does just that:  
```bash
make
```
This will compile the chemfiles library, convert the raw data into a binary file for ease of use and subsequently compiles and runs the main program.  
The program prints relevent information on the clustering and saves the results in a text file `output/cluster_assignments.txt`.  
