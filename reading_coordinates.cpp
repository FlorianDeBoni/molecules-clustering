/*
    Compile with:
    g++ -std=c++11 -O3 reading_coordinates.cpp FileUtils.cpp -lchemfiles -o reading_coordinates

    Run:
    ./reading_coordinates
*/

#include "FileUtils.h"
#include <iostream>

int main() {
    
    FileUtils file; 

    float* frame = file.loadData(1000);

    std::cout << frame[0] << "," << frame[1] << "," << frame[2] << std::endl << std::endl; 

    std::cout << file << std::endl;

    return 0;
}