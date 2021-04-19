#include "cuda_runtime.h"

cudaError_t bfieldWithCuda(double* h_by, double* h_bz, double* h_ey, double* h_ez, int m, double c);
cudaError_t efieldWithCuda(
	double* h_ex, double* h_ey, double* h_ez,
	double* h_by, double* h_bz,
	double* h_jxe, double* h_jye, double* h_jze,
	double* h_jxi, double* h_jyi, double* h_jzi,
	int m, double c);
