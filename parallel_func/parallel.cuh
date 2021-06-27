#include "cuda_runtime.h"

cudaError_t bfieldWithCuda(double* h_by, double* h_bz, double* h_ey, double* h_ez, int m, double c);
cudaError_t efieldWithCuda(
	double* h_ex, double* h_ey, double* h_ez,
	double* h_by, double* h_bz,
	double* h_jxe, double* h_jye, double* h_jze,
	double* h_jxi, double* h_jyi, double* h_jzi,
	int m, double c);
cudaError_t moverWithCuda(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, \
	double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m);
