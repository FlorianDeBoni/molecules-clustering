# Molecules Clustering
Clustering molecules dynamics on GPU

# Download the dataset
https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/1k5n_A/1k5n_A_protein.zip
and unzip it in a `/dataset` folder

Dataset website
https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/1k5n_A/1k5n_A.html

# Building the chemfiles library to read the .xtc files:
```bash
git clone https://github.com/chemfiles/chemfiles
cd chemfiles
mkdir build
cd build
cmake .. # various options are allowed here
cmake --build .
# if you whant to run the tests before installing:
ctest
cmake --build . --target install
```

# Read files:
1. Create a folder named `output`
2. Compile and run `md_reader.cpp` to convert the .xtc files into a binary one
3. Compile and run  `reading_coordinates` to test
