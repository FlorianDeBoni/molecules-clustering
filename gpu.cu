#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

// =======================================================
// Eigenvector solver
// =======================================================
__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float &x, float &y, float &z)
{
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

    if (fabsf(det) > 1e-8f) {
        x = (b0 * a11 - b1 * a01) / det;
        y = (a00 * b1 - a10 * b0) / det;
        z = 1.0f;
    } else {
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;

        if (fabsf(det) > 1e-8f) {
            x = (b0 * a22 - b1 * a02) / det;
            y = 1.0f;
            z = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;

            if (fabsf(det) > 1e-8f) {
                x = 1.0f;
                y = (b0 * a12 - b1 * a02) / det;
                z = (a01 * b1 - a11 * b0) / det;
            } else {
                x = 1.0f;
                y = 0.0f;
                z = 0.0f;
            }
        }
    }

    float norm = sqrtf(x*x + y*y + z*z);
    if (norm > 1e-8f) {
        x /= norm; y /= norm; z /= norm;
    }
}

// =======================================================
// Eigenvalues solver (symmetric 3x3)
// =======================================================
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;

    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;

    float p = sm00*sm00 + sm11*sm11 + sm22*sm22
            + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);

    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;

    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;

    float det = b00*(b11*b22 - b12*b12)
              - b01*(b01*b22 - b12*b02)
              + b02*(b01*b12 - b11*b02);
    det *= 0.5f;
    det = fminf(1.0f, fmaxf(-1.0f, det));

    float phi = acosf(det) / 3.0f;

    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + 2.0f * CUDART_PI_F / 3.0f);
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];

    if (lambda[0] < lambda[1]) { float t=lambda[0];lambda[0]=lambda[1];lambda[1]=t; }
    if (lambda[1] < lambda[2]) { float t=lambda[1];lambda[1]=lambda[2];lambda[2]=t; }
    if (lambda[0] < lambda[1]) { float t=lambda[0];lambda[0]=lambda[1];lambda[1]=t; }
}

// =======================================================
// RMSD kernel (packed upper triangle)
// =======================================================
__global__
void RMSD(const float* __restrict__ references,
          const float* __restrict__ targets,
          size_t N_references_subset,
          size_t N_targets_subset,
          size_t N_atoms,
          float* rmsd_device)
{
    int snap_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ref_idx  = blockIdx.y * blockDim.y + threadIdx.y;

    if (snap_idx >= N_targets_subset || ref_idx >= N_references_subset)
        return;

    // STEP 0: Compute centroids
    float rcx=0, rcy=0, rcz=0;
    float scx=0, scy=0, scz=0;

    for (int a=0;a<N_atoms;a++) {
        size_t b_references = a * N_references_subset;
        rcx += references[0*N_atoms*N_references_subset + b_references + ref_idx];
        rcy += references[1*N_atoms*N_references_subset + b_references + ref_idx];
        rcz += references[2*N_atoms*N_references_subset + b_references + ref_idx];
        
        size_t b_targets = a * N_targets_subset;
        scx += targets[0*N_atoms*N_targets_subset + b_targets + snap_idx];
        scy += targets[1*N_atoms*N_targets_subset + b_targets + snap_idx];
        scz += targets[2*N_atoms*N_targets_subset + b_targets + snap_idx];
    }

    rcx/=N_atoms; rcy/=N_atoms; rcz/=N_atoms;
    scx/=N_atoms; scy/=N_atoms; scz/=N_atoms;

    // STEP 1: Build correlation matrix A
    float a00=0,a01=0,a02=0,a10=0,a11=0,a12=0,a20=0,a21=0,a22=0;

    for (int a=0;a<N_atoms;a++) {
        size_t b_references = a * N_references_subset;

        float rx = references[0*N_atoms*N_references_subset + b_references + ref_idx]-rcx;
        float ry = references[1*N_atoms*N_references_subset + b_references + ref_idx]-rcy;
        float rz = references[2*N_atoms*N_references_subset + b_references + ref_idx]-rcz;

        size_t b_targets = a * N_targets_subset;

        float sx = targets[0*N_atoms*N_targets_subset + b_targets + snap_idx]-scx;
        float sy = targets[1*N_atoms*N_targets_subset + b_targets + snap_idx]-scy;
        float sz = targets[2*N_atoms*N_targets_subset + b_targets + snap_idx]-scz;

        a00+=sx*rx; a01+=sx*ry; a02+=sx*rz;
        a10+=sy*rx; a11+=sy*ry; a12+=sy*rz;
        a20+=sz*rx; a21+=sz*ry; a22+=sz*rz;
    }

    // Compute M = A^T * A
    float m00=a00*a00+a10*a10+a20*a20;
    float m01=a00*a01+a10*a11+a20*a21;
    float m02=a00*a02+a10*a12+a20*a22;
    float m11=a01*a01+a11*a11+a21*a21;
    float m12=a01*a02+a11*a12+a21*a22;
    float m22=a02*a02+a12*a12+a22*a22;

    // STEP 2: Compute eigenvalues
    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

    // STEP 3: Compute eigenvectors
    float3 v0,v1,v2;
    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[0],v0.x,v0.y,v0.z);
    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[1],v1.x,v1.y,v1.z);

    // STEP 5: Orthonormalize V
    float d=v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
    v1.x-=d*v0.x; v1.y-=d*v0.y; v1.z-=d*v0.z;

    float n=sqrtf(v1.x*v1.x+v1.y*v1.y+v1.z*v1.z);
    if(n>1e-8f){v1.x/=n;v1.y/=n;v1.z/=n;}

    v2.x=v0.y*v1.z-v0.z*v1.y;
    v2.y=v0.z*v1.x-v0.x*v1.z;
    v2.z=v0.x*v1.y-v0.y*v1.x;

    // STEP 6: Compute U = A*V
    float av0x=a00*v0.x+a01*v0.y+a02*v0.z;
    float av0y=a10*v0.x+a11*v0.y+a12*v0.z;
    float av0z=a20*v0.x+a21*v0.y+a22*v0.z;

    float av1x=a00*v1.x+a01*v1.y+a02*v1.z;
    float av1y=a10*v1.x+a11*v1.y+a12*v1.z;
    float av1z=a20*v1.x+a21*v1.y+a22*v1.z;

    float s0=sqrtf(fmaxf(lambda[0],1e-8f));
    float s1=sqrtf(fmaxf(lambda[1],1e-8f));

    float3 u0={av0x/s0,av0y/s0,av0z/s0};
    float3 u1={av1x/s1,av1y/s1,av1z/s1};
    float3 u2={u0.y*u1.z-u0.z*u1.y,
               u0.z*u1.x-u0.x*u1.z,
               u0.x*u1.y-u0.y*u1.x};

    // STEP 7: Compute R = U*V^T
    float R00=u0.x*v0.x+u1.x*v1.x+u2.x*v2.x;
    float R01=u0.x*v0.y+u1.x*v1.y+u2.x*v2.y;
    float R02=u0.x*v0.z+u1.x*v1.z+u2.x*v2.z;

    float R10=u0.y*v0.x+u1.y*v1.x+u2.y*v2.x;
    float R11=u0.y*v0.y+u1.y*v1.y+u2.y*v2.y;
    float R12=u0.y*v0.z+u1.y*v1.z+u2.y*v2.z;

    float R20=u0.z*v0.x+u1.z*v1.x+u2.z*v2.x;
    float R21=u0.z*v0.y+u1.z*v1.y+u2.z*v2.y;
    float R22=u0.z*v0.z+u1.z*v1.z+u2.z*v2.z;

    // STEP 8: Compute RMSD
    float sum2=0;
    for(int a=0;a<N_atoms;a++){
        size_t b_references = a * N_references_subset;
        float rx=references[0*N_atoms*N_references_subset+b_references+ref_idx]-rcx;
        float ry=references[1*N_atoms*N_references_subset+b_references+ref_idx]-rcy;
        float rz=references[2*N_atoms*N_references_subset+b_references+ref_idx]-rcz;

        size_t b_targets = a * N_targets_subset;
        float sx=targets[0*N_atoms*N_targets_subset+b_targets+snap_idx]-scx;
        float sy=targets[1*N_atoms*N_targets_subset+b_targets+snap_idx]-scy;
        float sz=targets[2*N_atoms*N_targets_subset+b_targets+snap_idx]-scz;

        float x=R00*sx+R01*sy+R02*sz;
        float y=R10*sx+R11*sy+R12*sz;
        float z=R20*sx+R21*sy+R22*sz;

        float dx=rx-x,dy=ry-y,dz=rz-z;
        sum2+=dx*dx+dy*dy+dz*dz;
    }

    float rmsd=sqrtf(sum2/N_atoms);

    size_t idx = ref_idx * N_targets_subset + snap_idx;
    rmsd_device[idx] = rmsd;

}
