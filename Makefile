CPUCC = g++
GPUCC = /usr/local/cuda/bin/nvcc

#CUDA_TARGET_FLAGS = -arch=sm_61      #GTX 1080 on Cameron cluster
#CUDA_TARGET_FLAGS = -arch=sm_75      #RTX 2080-Ti on Tx cluster
#CUDA_TARGET_FLAGS = -arch=sm_86      #RTX 3080 on Sh cluster

CXXFLAGS = -DDP
CXXFLAGS += -I. -I/usr/local/cuda/include/ -I/usr/include/x86_64-linux-gnu/
CC_CXXFLAGS = -Ofast -fopenmp
CUDA_CXXFLAGS = -O3 $(CUDA_TARGET_FLAGS)

CC_LDFLAGS =  -fopenmp -L/usr/local/x86_64-linux-gnu
CUDA_LDFLAGS = -L/usr/local/cuda/lib64/ 

CUDA_LIBS = -lcudart -lcuda

CC_SOURCES =  main.cpp utils.cpp
CUDA_SOURCES = gpu.cu 
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

EXECNAME = main


all:
	$(CPUCC) -c $(CC_SOURCES) $(CXXFLAGS) $(CC_CXXFLAGS)
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CC_OBJECTS)

# gpu stuff
# 	$(GPUCC) -c $(CUDA_SOURCES) $(CXXFLAGS) $(CUDA_CXXFLAGS)
# 	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CC_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) $(CC_LIBS)


clean:
	rm -f *.o $(EXECNAME)