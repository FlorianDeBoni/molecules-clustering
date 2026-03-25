#ifndef GPU_CUH
#define GPU_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Unified CUDA error-checking macro (used by both host code and gpu.cu).
// Prints file, line, and the CUDA error string, then exits.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Computes per-frame centroid (cx, cy, cz) and the inner-product G = Σ|r-c|²
// for N_frames frames, each with N_atoms atoms.
// Data layout: coords[dim * N_atoms * N_frames + atom * N_frames + frame]
__global__
void computeCentroidsG(const float* __restrict__ coords,
                       size_t N_atoms,
                       size_t N_frames,
                       float* __restrict__ centroids_x,
                       float* __restrict__ centroids_y,
                       float* __restrict__ centroids_z,
                       float* __restrict__ G);

// Computes the N_ref × N_tgt RMSD matrix using the Kabsch/QCP inner-product
// formula.  Shared memory size must be 3 * blockDim.x * blockDim.y * sizeof(float).
__global__
void RMSD(const float* __restrict__ refs,
          const float* __restrict__ tgts,
          size_t N_atoms,
          size_t N_ref,
          size_t N_tgt,
          const float* __restrict__ cx_ref,
          const float* __restrict__ cy_ref,
          const float* __restrict__ cz_ref,
          const float* __restrict__ G_ref,
          const float* __restrict__ cx_tgt,
          const float* __restrict__ cy_tgt,
          const float* __restrict__ cz_tgt,
          const float* __restrict__ G_tgt,
          float* __restrict__ rmsd);

#endif // GPU_CUH
