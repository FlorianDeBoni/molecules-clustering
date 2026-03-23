# Compilers
CPUCC = g++
GPUCC = nvcc

# Architecture Flags 
# CUDA_TARGET_FLAGS = -arch=sm_61       # GTX 1080
# CUDA_TARGET_FLAGS = -arch=sm_75      # RTX 2080-Ti
# CUDA_TARGET_FLAGS = -arch=sm_86      # RTX 3080

# Compiler flags
CXXFLAGS = -DDP -I. -I./lib -I/usr/include/x86_64-linux-gnu

CXXFLAGS += -O3 -I./chemfiles/include -I./chemfiles/build/include 
CUDA_CXXFLAGS = -O3 $(CUDA_TARGET_FLAGS)

# Linker flags
CC_LDFLAGS = -L./chemfiles/build -lchemfiles
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu
CUDA_LIBS = -lcudart

# Source files
CC_SOURCES = FileUtils.cpp
CUDA_SOURCES = main.cu gpu.cu utils.cu

# Chemfiles lib
CHEMFILES_GIT = https://github.com/chemfiles/chemfiles
CHEMFILES_DIR = ./chemfiles
CHEMFILES_BUILD_DIR = $(CHEMFILES_DIR)/build
BIN_DIR = output


# Object Lists
OBJECT_DIR = objects
CC_OBJECTS = $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(CC_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,$(OBJECT_DIR)/%.o,$(CUDA_SOURCES))

EXECNAME = main

# For modularity, data prep is seperate 
# Currently using the 200k dataset & saving a bin file
DATASET_READER = md_reader.cpp
BIN_FILE = $(BIN_DIR)/snapshots_coords_all.bin

all: $(CHEMFILES_BUILD_DIR)/libchemfiles.a $(EXECNAME)
	@./$(EXECNAME) $(BIN_FILE)

# Header dependecies
$(OBJECT_DIR)/main.o: main.cu utils.cuh gpu.cuh FileUtils.hpp
$(OBJECT_DIR)/gpu.o: gpu.cu utils.cuh gpu.cuh
$(OBJECT_DIR)/utils.o: utils.cu utils.cuh FileUtils.hpp
$(OBJECT_DIR)/FileUtils.o: FileUtils.cpp FileUtils.hpp

# C++ Compilation Rule
$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@$(CPUCC) -c -g -O0 $< $(CXXFLAGS) -o $@

# Linking Rule
$(EXECNAME): $(CC_OBJECTS) $(CUDA_OBJECTS) $(BIN_FILE)
	@$(GPUCC) -o $(EXECNAME) -g -O0 $(CC_OBJECTS) $(CUDA_OBJECTS) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CUDA_LIBS)


# GPU Compilation Rule
$(OBJECT_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	@$(GPUCC) -c $< $(CUDA_CXXFLAGS) -Xcompiler "$(CXXFLAGS)" -o $@


# Chemfiles build rule
$(CHEMFILES_BUILD_DIR)/libchemfiles.a:
	@if [ ! -d "$(CHEMFILES_DIR)" ]; then \
		echo "==> Cloning Chemfiles..."; \
		git clone $(CHEMFILES_GIT) $(CHEMFILES_DIR); \
	fi
	@mkdir -p $(CHEMFILES_BUILD_DIR)
	@echo "==> Configuring and building Chemfiles...";
	cmake -S $(CHEMFILES_DIR) -B $(CHEMFILES_BUILD_DIR)
	cmake --build $(CHEMFILES_BUILD_DIR) --target chemfiles

# Building binary snapshot file rule
$(BIN_FILE):
	@echo "==> Writing snapshot data onto $(BIN_FILE)"
	@mkdir -p $(BIN_DIR)
	$(CPUCC) -o md_reader $(DATASET_READER) $(CXXFLAGS) $(CC_LDFLAGS) 
	./md_reader
	rm md_reader

# Clean 
clean:
	rm -rf $(OBJECT_DIR) $(EXECNAME) $(CHEMFILES_DIR)