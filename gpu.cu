#include "gpu.cuh"
#include <cuda_runtime.h>

__global__
void loadSnapshotKernel(
    const float* src,
    float* dst,
    int N_frames,
    int N_atoms,
    int N_dims
)
{
    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    if (snap >= N_frames) return;

    for (int c = 0; c < N_dims; ++c) {
        size_t coord_offset = (size_t)c * N_atoms * N_frames;

        for (int a = 0; a < N_atoms; ++a) {

            size_t in  = coord_offset + a * N_frames + snap;
            size_t out = snap * (size_t)(N_atoms * N_dims) + c * N_atoms + a;

            // dst is organized by snapshots all x for atom then all y then all z
            dst[out] = src[in];
        }
    }

    // First atom coordinate for this snapshot (for testing)
    // printf("%f \n", src[snap * (size_t)(N_atoms * N_dims)]);
}
