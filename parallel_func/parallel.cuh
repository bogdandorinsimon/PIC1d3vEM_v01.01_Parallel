#include "cuda_runtime.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

cudaError_t bfieldWithCuda(double* h_by, double* h_bz, double* h_ey, double* h_ez, int m, double c);
cudaError_t efieldWithCuda(
	double* h_ex, double* h_ey, double* h_ez,
	double* h_by, double* h_bz,
	double* h_jxe, double* h_jye, double* h_jze,
	double* h_jxi, double* h_jyi, double* h_jzi,
	int m, double c);
cudaError_t moverWithCuda(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, \
	double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m);
cudaError_t currentWithCuda(double *h_jxe_s, double *h_jye_s, double *h_jze_s, double *h_jxi_s, double *h_jyi_s, double *h_jzi_s, double *h_x, double *h_vx, double *h_vy, \
    double *h_vz, double qse, double qsi, int np, int m);
