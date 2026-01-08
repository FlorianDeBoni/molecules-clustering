#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <vector>

#include "RMSD_func.h"

#define M_PI 3.14159265358979323846

// Calcul d'un vecteur propre pour une matrice symétrique 3x3
void compute_eigenvector(double m00, double m01, double m02,
                         double m11, double m12, double m22,
                         double lambda, double* v)
{
    // Résout le système homogène:
    // (m00 - lambda)*x1 + m01*x2 + m02*x3 = 0
    // m01*x1 + (m11 - lambda)*x2 + m12*x3 = 0
    // m02*x1 + m12*x2 + (m22 - lambda)*x3 = 0
    
    double a00 = m00 - lambda;
    double a01 = m01;
    double a02 = m02;
    double a10 = m01;
    double a11 = m11 - lambda;
    double a12 = m12;
    double a20 = m02;
    double a21 = m12;
    double a22 = m22 - lambda;

    double b0 = -a02;
    double b1 = -a12;
    double det = a00 * a11 - a01 * a10;

    // z = 1 et résolution par inversion du système résultant
    if (std::abs(det) > 1e-8) {
        v[0] = (b0 * a11 - b1 * a01) / det;
        v[1] = (a00 * b1 - a10 * b0) / det;
        v[2] = 1.0;
    } else {
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;
        
        // y = 1 et résolution par inversion du système résultant
        if (std::abs(det) > 1e-8) {
            v[0] = (b0 * a22 - b1 * a02) / det;
            v[1] = 1.0;
            v[2] = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;
            
            // x = 1 et résolution par inversion du système résultant
            if (std::abs(det) > 1e-8) {
                v[0] = 1.0;
                v[1] = (b0 * a12 - b1 * a02) / det;
                v[2] = (a01 * b1 - a11 * b0) / det;
            } else {
                // Fallback
                v[0] = 1.0;
                v[1] = 0.0;
                v[2] = 0.0;
            }
        }
    }

    // Normalisation
    double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > 1e-8) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

// Calcul robuste des valeurs propres pour une matrice symétrique 3x3
void compute_eigenvalues_symmetric_3x3(double m00, double m01, double m02,
                                       double m11, double m12, double m22,
                                       double* lambda)
{
    double trace = m00 + m11 + m22;
    double mean = trace / 3.0;
    
    // Décalage de la matrice
    double sm00 = m00 - mean;
    double sm11 = m11 - mean;
    double sm22 = m22 - mean;
    
    double p = sm00*sm00 + sm11*sm11 + sm22*sm22 + 2.0*(m01*m01 + m02*m02 + m12*m12);
    p = std::sqrt(p / 6.0);
    
    double invp = (p > 1e-8) ? (1.0 / p) : 0.0;
    
    double b00 = sm00 * invp;
    double b01 = m01 * invp;
    double b02 = m02 * invp;
    double b11 = sm11 * invp;
    double b12 = m12 * invp;
    double b22 = sm22 * invp;
    
    double det = b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02);
    det = det / 2.0;
    det = std::min(1.0, std::max(-1.0, det));
    
    double phi = std::acos(det) / 3.0;
    
    lambda[0] = mean + 2.0 * p * std::cos(phi);
    lambda[2] = mean + 2.0 * p * std::cos(phi + (2.0 * M_PI / 3.0));
    lambda[1] = 3.0 * mean - lambda[0] - lambda[2];
    
    // Tri décroissant
    if (lambda[0] < lambda[1]) std::swap(lambda[0], lambda[1]);
    if (lambda[1] < lambda[2]) std::swap(lambda[1], lambda[2]);
    if (lambda[0] < lambda[1]) std::swap(lambda[0], lambda[1]);
}

// Calcul de la matrice de rotation optimale entre deux molécules
// mol1 et mol2 sont des tableaux [3][N_atoms] contenant les coordonnées x, y, z
void get_rotation_matrix(const std::vector<std::vector<double>>& mol1,
                        const std::vector<std::vector<double>>& mol2,
                        int N_atoms,
                        double R[3][3])
{
    // STEP 0: Calculer les centroïdes
    double centroid_X[3] = {0.0, 0.0, 0.0};
    double centroid_Y[3] = {0.0, 0.0, 0.0};
    
    for (int a = 0; a < N_atoms; ++a) {
        centroid_X[0] += mol1[0][a];
        centroid_X[1] += mol1[1][a];
        centroid_X[2] += mol1[2][a];
        
        centroid_Y[0] += mol2[0][a];
        centroid_Y[1] += mol2[1][a];
        centroid_Y[2] += mol2[2][a];
    }
    
    centroid_X[0] /= N_atoms;
    centroid_X[1] /= N_atoms;
    centroid_X[2] /= N_atoms;
    
    centroid_Y[0] /= N_atoms;
    centroid_Y[1] /= N_atoms;
    centroid_Y[2] /= N_atoms;
    
    // STEP 1: Construction de la matrice de corrélation A = X * Y^T (centrée)
    double a00=0, a01=0, a02=0;
    double a10=0, a11=0, a12=0;
    double a20=0, a21=0, a22=0;
    
    for (int a = 0; a < N_atoms; ++a) {
        double Xx = mol1[0][a] - centroid_X[0];
        double Xy = mol1[1][a] - centroid_X[1];
        double Xz = mol1[2][a] - centroid_X[2];
        
        double Yx = mol2[0][a] - centroid_Y[0];
        double Yy = mol2[1][a] - centroid_Y[1];
        double Yz = mol2[2][a] - centroid_Y[2];
        
        a00 += Xx * Yx;  a01 += Xx * Yy;  a02 += Xx * Yz;
        a10 += Xy * Yx;  a11 += Xy * Yy;  a12 += Xy * Yz;
        a20 += Xz * Yx;  a21 += Xz * Yy;  a22 += Xz * Yz;
    }
    
    // Calculer M = A^T * A
    double m00 = a00*a00 + a10*a10 + a20*a20;
    double m01 = a00*a01 + a10*a11 + a20*a21;
    double m02 = a00*a02 + a10*a12 + a20*a22;
    double m11 = a01*a01 + a11*a11 + a21*a21;
    double m12 = a01*a02 + a11*a12 + a21*a22;
    double m22 = a02*a02 + a12*a12 + a22*a22;
    
    // STEP 2: Calculer les valeurs propres
    double eigenvalues[3];
    compute_eigenvalues_symmetric_3x3(m00, m01, m02, m11, m12, m22, eigenvalues);
    
    // STEP 3: Calculer les vecteurs propres
    double v0[3], v1[3], v2[3];
    
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[0], v0);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[1], v1);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, eigenvalues[2], v2);
    
    // Orthonormalisation: Gram-Schmidt
    double n0 = 1.0 / std::sqrt(v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
    v0[0] *= n0; v0[1] *= n0; v0[2] *= n0;
    
    double dot10 = v1[0]*v0[0] + v1[1]*v0[1] + v1[2]*v0[2];
    v1[0] -= dot10*v0[0];
    v1[1] -= dot10*v0[1];
    v1[2] -= dot10*v0[2];
    
    double n1 = 1.0 / std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    v1[0] *= n1; v1[1] *= n1; v1[2] *= n1;
    
    // Produit vectoriel pour v2
    v2[0] = v0[1]*v1[2] - v0[2]*v1[1];
    v2[1] = v0[2]*v1[0] - v0[0]*v1[2];
    v2[2] = v0[0]*v1[1] - v0[1]*v1[0];
    
    // STEP 4: Calculer U à partir de A*V
    double av0[3], av1[3], av2[3];
    
    av0[0] = a00*v0[0] + a01*v0[1] + a02*v0[2];
    av0[1] = a10*v0[0] + a11*v0[1] + a12*v0[2];
    av0[2] = a20*v0[0] + a21*v0[1] + a22*v0[2];
    
    av1[0] = a00*v1[0] + a01*v1[1] + a02*v1[2];
    av1[1] = a10*v1[0] + a11*v1[1] + a12*v1[2];
    av1[2] = a20*v1[0] + a21*v1[1] + a22*v1[2];
    
    av2[0] = a00*v2[0] + a01*v2[1] + a02*v2[2];
    av2[1] = a10*v2[0] + a11*v2[1] + a12*v2[2];
    av2[2] = a20*v2[0] + a21*v2[1] + a22*v2[2];
    
    double u0[3], u1[3], u2[3];
    
    double s0 = std::sqrt(std::max(eigenvalues[0], 1e-8));
    double s1 = std::sqrt(std::max(eigenvalues[1], 1e-8));
    double s2 = std::sqrt(std::max(eigenvalues[2], 1e-8));
    
    u0[0] = av0[0] / s0;
    u0[1] = av0[1] / s0;
    u0[2] = av0[2] / s0;
    
    u1[0] = av1[0] / s1;
    u1[1] = av1[1] / s1;
    u1[2] = av1[2] / s1;
    
    u2[0] = av2[0] / s2;
    u2[1] = av2[1] / s2;
    u2[2] = av2[2] / s2;
    
    // Calculer la matrice de rotation R = U*V^T
    R[0][0] = u0[0]*v0[0] + u1[0]*v1[0] + u2[0]*v2[0];
    R[0][1] = u0[0]*v0[1] + u1[0]*v1[1] + u2[0]*v2[1];
    R[0][2] = u0[0]*v0[2] + u1[0]*v1[2] + u2[0]*v2[2];
    
    R[1][0] = u0[1]*v0[0] + u1[1]*v1[0] + u2[1]*v2[0];
    R[1][1] = u0[1]*v0[1] + u1[1]*v1[1] + u2[1]*v2[1];
    R[1][2] = u0[1]*v0[2] + u1[1]*v1[2] + u2[1]*v2[2];
    
    R[2][0] = u0[2]*v0[0] + u1[2]*v1[0] + u2[2]*v2[0];
    R[2][1] = u0[2]*v0[1] + u1[2]*v1[1] + u2[2]*v2[1];
    R[2][2] = u0[2]*v0[2] + u1[2]*v1[2] + u2[2]*v2[2];
    
    // Vérifier le déterminant (doit être +1 pour une rotation propre)
    double detR = R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1]) -
                  R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0]) +
                  R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);
    
    if (detR < 0.0) {
        // Inverser le dernier vecteur propre
        u2[0] = -u2[0];
        u2[1] = -u2[1];
        u2[2] = -u2[2];
        
        // Recalculer R
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
}

// Calcul du RMSD entre deux molécules
double calcul_RMSD(const std::vector<std::vector<double>>& mol1,
                   const std::vector<std::vector<double>>& mol2,
                   int N_atoms)
{
    // Calculer les centroïdes
    double centroid_X[3] = {0.0, 0.0, 0.0};
    double centroid_Y[3] = {0.0, 0.0, 0.0};
    
    for (int a = 0; a < N_atoms; ++a) {
        centroid_X[0] += mol1[0][a];
        centroid_X[1] += mol1[1][a];
        centroid_X[2] += mol1[2][a];
        
        centroid_Y[0] += mol2[0][a];
        centroid_Y[1] += mol2[1][a];
        centroid_Y[2] += mol2[2][a];
    }
    
    centroid_X[0] /= N_atoms;
    centroid_X[1] /= N_atoms;
    centroid_X[2] /= N_atoms;
    
    centroid_Y[0] /= N_atoms;
    centroid_Y[1] /= N_atoms;
    centroid_Y[2] /= N_atoms;
    
    // Obtenir la matrice de rotation optimale
    double R[3][3];
    get_rotation_matrix(mol1, mol2, N_atoms, R);
    
    // Calculer le RMSD avec la rotation
    double sum_squared_dist = 0.0;
    
    for (int a = 0; a < N_atoms; ++a) {
        double Xi_x = mol1[0][a] - centroid_X[0];
        double Xi_y = mol1[1][a] - centroid_X[1];
        double Xi_z = mol1[2][a] - centroid_X[2];
        
        double Yi_x = mol2[0][a] - centroid_Y[0];
        double Yi_y = mol2[1][a] - centroid_Y[1];
        double Yi_z = mol2[2][a] - centroid_Y[2];
        
        // Appliquer la rotation
        double RYi_x = R[0][0]*Yi_x + R[0][1]*Yi_y + R[0][2]*Yi_z;
        double RYi_y = R[1][0]*Yi_x + R[1][1]*Yi_y + R[1][2]*Yi_z;
        double RYi_z = R[2][0]*Yi_x + R[2][1]*Yi_y + R[2][2]*Yi_z;
        
        double diff_x = Xi_x - RYi_x;
        double diff_y = Xi_y - RYi_y;
        double diff_z = Xi_z - RYi_z;
        
        sum_squared_dist += diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
    }
    
    return std::sqrt(sum_squared_dist / N_atoms);
}