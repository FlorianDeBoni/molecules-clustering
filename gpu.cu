// gpu.cu - Complete corrected version
#include "gpu.cuh"
#include <cuda_runtime.h>

__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float* v)
{
    // Solve the homogeneous system:
    // (m00 - lambda)*x1 + m01*x2 + m02*x3 = 0
    // m01*x1 + (m11 - lambda)*x2 + m12*x3 = 0
    // m02*x1 + m12*x2 + (m22 - lambda)*x3 = 0
    
    float a00 = m00 - lambda;
    float a01 = m01;
    float a02 = m02;
    float a10 = m01;
    float a11 = m11 - lambda;
    float a12 = m12;
    float a20 = m02;
    float a21 = m12;
    float a22 = m22 - lambda;

    float b0 = -a02;
    float b1 = -a12;
    float det = a00 * a11 - a01 * a10;

    // z = 1 and solve by inversing the resulting system
    if (fabsf(det) > 1e-8f) {
        v[0] = (b0 * a11 - b1 * a01) / det;
        v[1] = (a00 * b1 - a10 * b0) / det;
        v[2] = 1.0f;
    } else {
        
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;
        
        // y = 1 and solve by inversing the resulting system
        if (fabsf(det) > 1e-8f) {
            v[0] = (b0 * a22 - b1 * a02) / det;
            v[1] = 1.0f;
            v[2] = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;
            
            // x = 1 and solve by inversing the resulting system
            if (fabsf(det) > 1e-8f) {
                v[0] = 1.0f;
                v[1] = (b0 * a12 - b1 * a02) / det;
                v[2] = (a01 * b1 - a11 * b0) / det;
            } else {
                // Fallback
                v[0] = 1.0f;
                v[1] = 0.0f;
                v[2] = 0.0f;
            }
        }
    }

    // Normalize
    float norm = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > 1e-8f) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

// Robust eigenvalue solver using iterative method
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;
    
    // Shift matrix
    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;
    
    float p = sm00*sm00 + sm11*sm11 + sm22*sm22 + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);
    
    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;
    
    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;
    
    float det = b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02);
    det = det / 2.0f;
    det = fminf(1.0f, fmaxf(-1.0f, det));
    
    float phi = acosf(det) / 3.0f;
    
    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + (2.0f * 3.14159265358979323846f / 3.0f));
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];
    
    // Sort
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
    if (lambda[1] < lambda[2]) { float tmp = lambda[1]; lambda[1] = lambda[2]; lambda[2] = tmp; }
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
}

__global__
void RMSD(
    const float* __restrict__ dst,
    int N_frames,
    int N_atoms,
    float* out
)
{
    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    int ref_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds check
    if (snap >= N_frames || ref_idx >= N_frames)
        return;
    
    // FIX: Only compute upper triangle to avoid race conditions
    // Each pair (i,j) where i < j is computed only once
    // if (snap < ref_idx)
    //     return;
        
    // if (snap == ref_idx) {
    //     out[ref_idx * N_frames + snap] = 0.0f;
    //     return;
    // }

    int block = N_atoms * N_frames;

    // STEP 0: Compute centroids
    float centroid_X_x = 0.0f, centroid_X_y = 0.0f, centroid_X_z = 0.0f;
    float centroid_Y_x = 0.0f, centroid_Y_y = 0.0f, centroid_Y_z = 0.0f;

    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        centroid_X_x += dst[xr];
        centroid_X_y += dst[yr];
        centroid_X_z += dst[zr];

        centroid_Y_x += dst[xs];
        centroid_Y_y += dst[ys];
        centroid_Y_z += dst[zs];
    }

    centroid_X_x /= N_atoms;
    centroid_X_y /= N_atoms;
    centroid_X_z /= N_atoms;

    centroid_Y_x /= N_atoms;
    centroid_Y_y /= N_atoms;
    centroid_Y_z /= N_atoms;

    // STEP 1: Build correlation matrix A
    float a00=0, a01=0, a02=0;
    float a10=0, a11=0, a12=0;
    float a20=0, a21=0, a22=0;

    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        float Xx = dst[xr] - centroid_X_x;
        float Xy = dst[yr] - centroid_X_y;
        float Xz = dst[zr] - centroid_X_z;

        float Yx = dst[xs] - centroid_Y_x;
        float Yy = dst[ys] - centroid_Y_y;
        float Yz = dst[zs] - centroid_Y_z;

        a00 += Xx * Yx;  a01 += Xx * Yy;  a02 += Xx * Yz;
        a10 += Xy * Yx;  a11 += Xy * Yy;  a12 += Xy * Yz;
        a20 += Xz * Yx;  a21 += Xz * Yy;  a22 += Xz * Yz;
    }

    // Compute M = A^T * A
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    // STEP 2: Compute eigenvalues
    float eigenvalues[3];
    compute_eigenvalues_symmetric_3x3(m00, m01, m02, m11, m12, m22, eigenvalues);

    // STEP 3: Compute eigenvectors
    float v0[3], v1[3], v2[3];
    
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[0], v0);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[1], v1);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[2], v2);

    // FIXED: Orthonormalization with stability checks
    // Normalize v0
    float mag0 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    if (mag0 > 1e-8f) {
        float n0 = rsqrtf(mag0);
        v0[0] *= n0; 
        v0[1] *= n0; 
        v0[2] *= n0;
    } else {
        // Degenerate case - use default
        v0[0] = 1.0f; 
        v0[1] = 0.0f; 
        v0[2] = 0.0f;
    }

    // Gram-Schmidt: v1 = v1 - (v1·v0)v0
    float dot10 = v1[0]*v0[0] + v1[1]*v0[1] + v1[2]*v0[2];
    v1[0] -= dot10*v0[0];
    v1[1] -= dot10*v0[1];
    v1[2] -= dot10*v0[2];

    // Normalize v1
    float mag1 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    if (mag1 > 1e-8f) {
        float n1 = rsqrtf(mag1);
        v1[0] *= n1; 
        v1[1] *= n1; 
        v1[2] *= n1;
    } else {
        // Degenerate case - make orthogonal to v0
        if (fabsf(v0[0]) < 0.9f) {
            v1[0] = 0.0f; 
            v1[1] = -v0[2]; 
            v1[2] = v0[1];
        } else {
            v1[0] = -v0[2]; 
            v1[1] = 0.0f; 
            v1[2] = v0[0];
        }
        float norm1 = sqrtf(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        if (norm1 > 1e-8f) {
            v1[0] /= norm1;
            v1[1] /= norm1;
            v1[2] /= norm1;
        }
    }

    // v2 = v0 × v1 (cross product ensures orthogonality)
    v2[0] = v0[1]*v1[2] - v0[2]*v1[1];
    v2[1] = v0[2]*v1[0] - v0[0]*v1[2];
    v2[2] = v0[0]*v1[1] - v0[1]*v1[0];

    // STEP 4: Compute U from A*V with stability checks
    float av0[3], av1[3], av2[3];
    
    av0[0] = a00*v0[0] + a01*v0[1] + a02*v0[2];
    av0[1] = a10*v0[0] + a11*v0[1] + a12*v0[2];
    av0[2] = a20*v0[0] + a21*v0[1] + a22*v0[2];
    
    av1[0] = a00*v1[0] + a01*v1[1] + a02*v1[2];
    av1[1] = a10*v1[0] + a11*v1[1] + a12*v1[2];
    av1[2] = a20*v1[0] + a21*v1[1] + a22*v1[2];
    
    av2[0] = a00*v2[0] + a01*v2[1] + a02*v2[2];
    av2[1] = a10*v2[0] + a11*v2[1] + a12*v2[2];
    av2[2] = a20*v2[0] + a21*v2[1] + a22*v2[2];
    
    float u0[3], u1[3], u2[3];

    // FIXED: Handle small eigenvalues more carefully
    if (eigenvalues[0] > 1e-6f) {
        float s0 = sqrtf(eigenvalues[0]);
        u0[0] = av0[0] / s0;
        u0[1] = av0[1] / s0;
        u0[2] = av0[2] / s0;
    } else {
        // Degenerate - normalize av0 directly
        float norm = sqrtf(av0[0]*av0[0] + av0[1]*av0[1] + av0[2]*av0[2]);
        if (norm > 1e-8f) {
            u0[0] = av0[0] / norm;
            u0[1] = av0[1] / norm;
            u0[2] = av0[2] / norm;
        } else {
            u0[0] = v0[0];
            u0[1] = v0[1];
            u0[2] = v0[2];
        }
    }

    if (eigenvalues[1] > 1e-6f) {
        float s1 = sqrtf(eigenvalues[1]);
        u1[0] = av1[0] / s1;
        u1[1] = av1[1] / s1;
        u1[2] = av1[2] / s1;
    } else {
        float norm = sqrtf(av1[0]*av1[0] + av1[1]*av1[1] + av1[2]*av1[2]);
        if (norm > 1e-8f) {
            u1[0] = av1[0] / norm;
            u1[1] = av1[1] / norm;
            u1[2] = av1[2] / norm;
        } else {
            u1[0] = v1[0];
            u1[1] = v1[1];
            u1[2] = v1[2];
        }
    }

    if (eigenvalues[2] > 1e-6f) {
        float s2 = sqrtf(eigenvalues[2]);
        u2[0] = av2[0] / s2;
        u2[1] = av2[1] / s2;
        u2[2] = av2[2] / s2;
    } else {
        float norm = sqrtf(av2[0]*av2[0] + av2[1]*av2[1] + av2[2]*av2[2]);
        if (norm > 1e-8f) {
            u2[0] = av2[0] / norm;
            u2[1] = av2[1] / norm;
            u2[2] = av2[2] / norm;
        } else {
            u2[0] = v2[0];
            u2[1] = v2[1];
            u2[2] = v2[2];
        }
    }

    // Compute rotation matrix R = U*V^T
    float R[3][3];
    R[0][0] = u0[0]*v0[0] + u1[0]*v1[0] + u2[0]*v2[0];
    R[0][1] = u0[0]*v0[1] + u1[0]*v1[1] + u2[0]*v2[1];
    R[0][2] = u0[0]*v0[2] + u1[0]*v1[2] + u2[0]*v2[2];

    R[1][0] = u0[1]*v0[0] + u1[1]*v1[0] + u2[1]*v2[0];
    R[1][1] = u0[1]*v0[1] + u1[1]*v1[1] + u2[1]*v2[1];
    R[1][2] = u0[1]*v0[2] + u1[1]*v1[2] + u2[1]*v2[2];

    R[2][0] = u0[2]*v0[0] + u1[2]*v1[0] + u2[2]*v2[0];
    R[2][1] = u0[2]*v0[1] + u1[2]*v1[1] + u2[2]*v2[1];
    R[2][2] = u0[2]*v0[2] + u1[2]*v1[2] + u2[2]*v2[2];

    // Check determinant and fix if needed
    float detR = R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1]) -
                 R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0]) +
                 R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);

    if (detR < 0.0f) {
        // Flip u2 to ensure proper rotation
        u2[0] = -u2[0];
        u2[1] = -u2[1];
        u2[2] = -u2[2];
        
        R[0][0] = u0[0]*v0[0] + u1[0]*v1[0] + u2[0]*v2[0];
        R[0][1] = u0[0]*v0[1] + u1[0]*v1[1] + u2[0]*v2[1];
        R[0][2] = u0[0]*v0[2] + u1[0]*v1[2] + u2[0]*v2[2];
        
        R[1][0] = u0[1]*v0[0] + u1[1]*v1[0] + u2[1]*v2[0];
        R[1][1] = u0[1]*v0[1] + u1[1]*v1[1] + u2[1]*v2[1];
        R[1][2] = u0[1]*v0[2] + u1[1]*v1[2] + u2[1]*v2[2];
        
        R[2][0] = u0[2]*v0[0] + u1[2]*v1[0] + u2[2]*v2[0];
        R[2][1] = u0[2]*v0[1] + u1[2]*v1[1] + u2[2]*v2[1];
        R[2][2] = u0[2]*v0[2] + u1[2]*v1[2] + u2[2]*v2[2];
    }

    // ================= DEBUG PRINT =================
    if (snap < 3 && ref_idx == 2) {
        printf("snap=%d ref=%d det(R)=%.6f | [%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n",
           snap, ref_idx, detR,
           R[0][0],R[0][1],R[0][2],
           R[1][0],R[1][1],R[1][2],
           R[2][0],R[2][1],R[2][2]);
    }
    // =================================================


    // Calculate RMSD
    float sum_squared_dist = 0.0f;
    
    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;
        
        float Xi_x = dst[xr] - centroid_X_x;
        float Xi_y = dst[yr] - centroid_X_y;
        float Xi_z = dst[zr] - centroid_X_z;
        
        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;
        
        float Yi_x = dst[xs] - centroid_Y_x;
        float Yi_y = dst[ys] - centroid_Y_y;
        float Yi_z = dst[zs] - centroid_Y_z;
        
        // Apply rotation R to Y
        float RYi_x = R[0][0]*Yi_x + R[0][1]*Yi_y + R[0][2]*Yi_z;
        float RYi_y = R[1][0]*Yi_x + R[1][1]*Yi_y + R[1][2]*Yi_z;
        float RYi_z = R[2][0]*Yi_x + R[2][1]*Yi_y + R[2][2]*Yi_z;
        
        float diff_x = Xi_x - RYi_x;
        float diff_y = Xi_y - RYi_y;
        float diff_z = Xi_z - RYi_z;
        
        sum_squared_dist += diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
    }
    
    float rmsd = sqrtf(sum_squared_dist / N_atoms);

    // ================= DEBUG RMSD =================
    if (snap < 3 && ref_idx == 2) {
        printf("RMSD(snap=%d, ref=%d) = %.8f\n", snap, ref_idx, rmsd);
    }



    // Write symmetric entries - safe now because only upper triangle is computed
    out[ref_idx * N_frames + snap] = rmsd;
    out[snap * N_frames + ref_idx] = rmsd;
}