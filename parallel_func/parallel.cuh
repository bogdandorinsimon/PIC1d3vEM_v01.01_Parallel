#include "cuda_runtime.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val)
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

// kernels
__global__ inline void bfieldKernel(double* by, double* bz, double* ey, double* ez, int m, double c);
__global__ inline void efieldKernel(
	double* ex, double* ey, double* ez,
	double* by, double* bz,
	double* jxe, double* jye, double* jze,
	double* jxi, double* jyi, double* jzi,
	int m, double c);
__global__ inline void moverKernel(double* x, double* vx, double* vy, double* vz, double* ex, double* ey, double* ez, double* by, double* bz, \
    double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m);
__global__ inline void currentInitializationKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, int m);
__global__ inline void currentSmoothingKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, int m);
__global__ inline void currentKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, \
    double* jxi_s, double* jyi_s, double* jzi_s, double* x, double* vx, double* vy, double* vz, double qse, double qsi, int np, int m);

// parallel functions
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

// all parallel
void initialize(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, double* h_jxe, double* h_jye, double* h_jze, double* h_jxi, double* h_jyi, double* h_jzi, int m, int np);
void parallelFunctions(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, double* h_jxe, double* h_jye, double* h_jze, double* h_jxi, double* h_jyi, double* h_jzi, \
	double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double qse, double qsi, double c, int m, int np, bool copyToHost);
void freeMemory();