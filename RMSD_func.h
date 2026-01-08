#ifndef RMSD_func
#define RMSD_func

#include <vector>

void compute_eigenvector(double m00, double m01, double m02,
                         double m11, double m12, double m22,
                         double lambda, double* v);

void compute_eigenvalues_symmetric_3x3(double m00, double m01, double m02,
                                       double m11, double m12, double m22,
                                       double* lambda);

void get_rotation_matrix(const std::vector<std::vector<double>>& mol1,
                        const std::vector<std::vector<double>>& mol2,
                        int N_atoms,
                        double R[3][3]);

double calcul_RMSD(const std::vector<std::vector<double>>& mol1,
                   const std::vector<std::vector<double>>& mol2,
                   int N_atoms);

#endif