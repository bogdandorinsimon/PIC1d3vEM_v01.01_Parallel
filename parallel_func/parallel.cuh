#include "bfield.cu"
#include "cuda_runtime.h"

#if !defined( _RVGS_ )
#define _RVGS_
cudaError_t bfieldWithCuda(double* h_by, double* h_bz, double* h_ey, double* h_ez, int m, double c);
#endif
