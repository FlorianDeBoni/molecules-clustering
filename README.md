# Molecules Clustering
Clustering molecules dynamics on GPU

# Setup
This project uses `g++` and `nvcc` for compilation.  
In the resulting folder from cloning this repository, do the following:
### 1. Download the dataset
[Download raw dataset](https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/1k5n_A/1k5n_A_protein.zip)
and unzip it in a `/dataset` folder

[For more info on the data](https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/1k5n_A/1k5n_A.html)

### 2. Install cmake
Make sure to have cmake installed, it's used to build the [chemfiles](https://github.com/chemfiles/chemfiles) library locally to read the raw data.  
For linux:
```bash
sudo apt install cmake
```

### 3. Run make
```bash
make
```
This will compile the chemfiles library, convert the raw data into a binary file for ease of use and subsequently compiles and runs the main program.
