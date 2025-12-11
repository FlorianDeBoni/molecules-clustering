#include "gpu.cuh"
#include <cuda_runtime.h>

__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float* v)
{
    // Compute (M - lambda*I)
    float a00 = m00 - lambda;
    float a01 = m01;
    float a02 = m02;
    float a11 = m11 - lambda;
    float a12 = m12;
    float a22 = m22 - lambda;

    // Rows of (M - lambda*I)
    float r0[3] = {a00, a01, a02};
    float r1[3] = {a01, a11, a12};
    float r2[3] = {a02, a12, a22};

    // Cross products of pairs of rows
    float c0[3], c1[3], c2[3];
    
    c0[0] = r0[1]*r1[2] - r0[2]*r1[1];
    c0[1] = r0[2]*r1[0] - r0[0]*r1[2];
    c0[2] = r0[0]*r1[1] - r0[1]*r1[0];
    
    c1[0] = r0[1]*r2[2] - r0[2]*r2[1];
    c1[1] = r0[2]*r2[0] - r0[0]*r2[2];
    c1[2] = r0[0]*r2[1] - r0[1]*r2[0];
    
    c2[0] = r1[1]*r2[2] - r1[2]*r2[1];
    c2[1] = r1[2]*r2[0] - r1[0]*r2[2];
    c2[2] = r1[0]*r2[1] - r1[1]*r2[0];

    float len0 = c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2];
    float len1 = c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2];
    float len2 = c2[0]*c2[0] + c2[1]*c2[1] + c2[2]*c2[2];

    float* best = c0;
    float best_len = len0;
    
    if (len1 > best_len) {
        best = c1;
        best_len = len1;
    }
    if (len2 > best_len) {
        best = c2;
        best_len = len2;
    }

    float norm = sqrtf(best_len);
    if (norm > 1e-8f) {
        v[0] = best[0] / norm;
        v[1] = best[1] / norm;
        v[2] = best[2] / norm;
    } else {
        v[0] = 1.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    }
}

__global__
void computeA(
    const float* __restrict__ dst,
    float* __restrict__ outA,
    int N_frames,
    int N_atoms,
    int ref_idx
)
{
    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    if (snap >= N_frames || snap == ref_idx)
        return;

    int block = N_atoms * N_frames;

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

        float Xx = dst[xr];
        float Xy = dst[yr];
        float Xz = dst[zr];

        float Yx = dst[xs];
        float Yy = dst[ys];
        float Yz = dst[zs];

        a00 += Xx * Yx;  a01 += Xx * Yy;  a02 += Xx * Yz;
        a10 += Xy * Yx;  a11 += Xy * Yy;  a12 += Xy * Yz;
        a20 += Xz * Yx;  a21 += Xz * Yy;  a22 += Xz * Yz;
    }

    // Compute M = A^T A
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    // Characteristic polynomial coefficients
    float tr = m00 + m11 + m22;
    float trM2 = m00*m00 + 2*m01*m01 + 2*m02*m02 + m11*m11 + 2*m12*m12 + m22*m22;
    
    float p = (tr*tr - trM2) / 2.0f;
    float q = (m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02));

    // Eigenvalues via Cardano
    float a_coef = -tr;
    float b_coef = p;
    float c_coef = -q;
    
    float Q = (3*b_coef - a_coef*a_coef) / 9.0f;
    float R = (9*a_coef*b_coef - 27*c_coef - 2*a_coef*a_coef*a_coef) / 54.0f;
    float D = Q*Q*Q + R*R;
    
    float lambda0, lambda1, lambda2;
    
    if (D < 0) {
        // Three real roots
        float theta = acosf(R / sqrtf(-Q*Q*Q));
        float sqrt_Q = sqrtf(-Q);
        lambda0 = 2*sqrt_Q*cosf(theta/3.0f) - a_coef/3.0f;
        lambda1 = 2*sqrt_Q*cosf((theta + 2*3.14159265f)/3.0f) - a_coef/3.0f;
        lambda2 = 2*sqrt_Q*cosf((theta + 4*3.14159265f)/3.0f) - a_coef/3.0f;
    } else {
        // Use stable computation
        float S = cbrtf(R + sqrtf(D));
        float T = cbrtf(R - sqrtf(D));
        lambda0 = S + T - a_coef/3.0f;
        lambda1 = lambda0;
        lambda2 = lambda0;
    }

    // Ensure eigenvalues are non-negative
    lambda0 = fmaxf(lambda0, 0.0f);
    lambda1 = fmaxf(lambda1, 0.0f);
    lambda2 = fmaxf(lambda2, 0.0f);

    // Eigenvectors
    float v0[3], v1[3], v2[3];
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda0, v0);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda1, v1);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda2, v2);

    // Singular values
    float s0 = sqrtf(lambda0);
    float s1 = sqrtf(lambda1);
    float s2 = sqrtf(lambda2);
    
    // Compute U = A*V*Σ^(-1)
    float u0[3], u1[3], u2[3];
    
    u0[0] = a00*v0[0] + a01*v0[1] + a02*v0[2];
    u0[1] = a10*v0[0] + a11*v0[1] + a12*v0[2];
    u0[2] = a20*v0[0] + a21*v0[1] + a22*v0[2];
    
    u1[0] = a00*v1[0] + a01*v1[1] + a02*v1[2];
    u1[1] = a10*v1[0] + a11*v1[1] + a12*v1[2];
    u1[2] = a20*v1[0] + a21*v1[1] + a22*v1[2];
    
    u2[0] = a00*v2[0] + a01*v2[1] + a02*v2[2];
    u2[1] = a10*v2[0] + a11*v2[1] + a12*v2[2];
    u2[2] = a20*v2[0] + a21*v2[1] + a22*v2[2];
    
    // Normalize
    if (s0 > 1e-8f) {
        u0[0] /= s0; u0[1] /= s0; u0[2] /= s0;
    } else {
        // Make orthonormal if singular value is too small
        float len = sqrtf(u0[0]*u0[0] + u0[1]*u0[1] + u0[2]*u0[2]);
        if (len > 1e-8f) {
            u0[0] /= len; u0[1] /= len; u0[2] /= len;
        }
    }
    
    if (s1 > 1e-8f) {
        u1[0] /= s1; u1[1] /= s1; u1[2] /= s1;
    } else {
        float len = sqrtf(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]);
        if (len > 1e-8f) {
            u1[0] /= len; u1[1] /= len; u1[2] /= len;
        }
    }
    
    if (s2 > 1e-8f) {
        u2[0] /= s2; u2[1] /= s2; u2[2] /= s2;
    } else {
        float len = sqrtf(u2[0]*u2[0] + u2[1]*u2[1] + u2[2]*u2[2]);
        if (len > 1e-8f) {
            u2[0] /= len; u2[1] /= len; u2[2] /= len;
        }
    }
    
    // Store U (row-major)
    float* U = outA + snap * 9;
    U[0] = u0[0]; U[1] = u1[0]; U[2] = u2[0];
    U[3] = u0[1]; U[4] = u1[1]; U[5] = u2[1];
    U[6] = u0[2]; U[7] = u1[2]; U[8] = u2[2];

    // Debug print for multiple frames
    if (snap < 3) {
        printf("\n=== Frame %d ===\n", snap);
        printf("Matrix A:\n%f %f %f\n%f %f %f\n%f %f %f\n",
               a00, a01, a02, a10, a11, a12, a20, a21, a22);
        printf("Matrix M (A^T A):\n%f %f %f\n%f %f %f\n%f %f %f\n",
               m00, m01, m02, m01, m11, m12, m02, m12, m22);
        printf("Eigenvalues: %f %f %f\n", lambda0, lambda1, lambda2);
        printf("Singular values: %f %f %f\n", s0, s1, s2);
        printf("U matrix:\n%f %f %f\n%f %f %f\n%f %f %f\n",
               U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8]);
               
        // Print first atom coordinates for debugging
        int xr = 0*block + 0*N_frames + ref_idx;
        int yr = 1*block + 0*N_frames + ref_idx;
        int zr = 2*block + 0*N_frames + ref_idx;
        int xs = 0*block + 0*N_frames + snap;
        int ys = 1*block + 0*N_frames + snap;
        int zs = 2*block + 0*N_frames + snap;
        printf("First atom - Ref: (%f, %f, %f), Snap: (%f, %f, %f)\n",
               dst[xr], dst[yr], dst[zr], dst[xs], dst[ys], dst[zs]);
    }

    // Compute RMSD
    float sum_sq = 0.0f;
    
    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;
        
        float Xref = dst[xr];
        float Yref = dst[yr];
        float Zref = dst[zr];
        
        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;
        
        float Xsnap = dst[xs];
        float Ysnap = dst[ys];
        float Zsnap = dst[zs];
        
        float Xrot = U[0]*Xsnap + U[1]*Ysnap + U[2]*Zsnap;
        float Yrot = U[3]*Xsnap + U[4]*Ysnap + U[5]*Zsnap;
        float Zrot = U[6]*Xsnap + U[7]*Ysnap + U[8]*Zsnap;
        
        float dx = Xrot - Xref;
        float dy = Yrot - Yref;
        float dz = Zrot - Zref;
        
        sum_sq += dx*dx + dy*dy + dz*dz;
    }
    
    float rmsd = sqrtf(sum_sq / (float)N_atoms);
    // printf("RMSD for frame %d: %.10f\n", snap, rmsd);
}