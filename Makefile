# Compilers
CPUCC = g++
GPUCC = /usr/local/cuda/bin/nvcc

# Architecture Flags 
# CUDA_TARGET_FLAGS = -arch=sm_61      # GTX 1080
# CUDA_TARGET_FLAGS = -arch=sm_75      # RTX 2080-Ti
# CUDA_TARGET_FLAGS = -arch=sm_86      # RTX 3080

# Compiler flags
# CXXFLAGS = -DDP -I. -I./lib -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
CXXFLAGS = -DDP -I. -I./lib
CC_CXXFLAGS = -Ofast -fopenmp
CUDA_CXXFLAGS = -O3 $(CUDA_TARGET_FLAGS)

# Linker flags
# cnpy requires -lz e.i zlib 
CC_LDFLAGS = -fopenmp -lz
# CUDA_LDFLAGS = -L/usr/local/cuda/lib64
# CUDA_LIBS = -lcudart -lcuda

# Source files
CC_SOURCES = main.cpp utils.cpp lib/cnpy.cpp
# CUDA_SOURCES = gpu.cu

# Dirs
OBJECT_DIR = objects
LIB_DIR = lib

# Object Lists
CC_OBJECTS = $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(CC_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,$(OBJECT_DIR)/%.o,$(CUDA_SOURCES))

EXECNAME = main

all: $(EXECNAME)
	./$(EXECNAME)

# Linking Rule
$(EXECNAME): $(CC_OBJECTS) $(CUDA_OBJECTS)
	$(CPUCC) -o $@ $^ $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CUDA_LIBS)

# C++ Compilation Rule
$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CPUCC) -c $< $(CXXFLAGS) $(CC_CXXFLAGS) -o $@

# GPU Compilation Rule
# $(OBJECT_DIR)/%.o: %.cu
# 	@mkdir -p $(dir $@)
# 	$(GPUCC) -c $< $(CXXFLAGS) $(CUDA_CXXFLAGS) -o $@

# Clean 
clean:
	rm -rf $(OBJECT_DIR) $(EXECNAME)