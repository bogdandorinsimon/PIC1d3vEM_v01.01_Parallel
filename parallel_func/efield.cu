#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void efieldKernel(
	double* ex, double* ey, double* ez,
	double* by, double* bz, 
	double* jxe, double* jye, double* jze, 
	double* jxi, double* jyi, double* jzi, 
	int m, double c) {
	int i = threadIdx.x;

	if (i >= 2 && i < m - 3) {
		ex[i] = ex[i] - (jxe[i] + jxi[i]);
		ey[i] = ey[i] - (jye[i] + jyi[i]) - c * (bz[i + 1] - bz[i]);
		ez[i] = ez[i] - (jze[i] + jzi[i]) + c * (by[i + 1] - by[i]);
	}


	if (i == m - 1) {
		ex[i] = ex[4];
		ey[i] = ey[4];
		ez[i] = ez[4];
	}

	if (i == m - 2) {
		ex[i] = ex[3];
		ey[i] = ey[3];
		ez[i] = ez[3];
	}

	if (i == m - 3) {
		ex[i] = ex[2];
		ey[i] = ey[2];
		ez[i] = ez[2];
	}

	switch (i) {
	case 0:
		ex[i] = ex[m - 5];
		ey[i] = ey[m - 5];
		ez[i] = ez[m - 5];
		break;
	case 1:
		ex[i] = ex[m - 4];
		ey[i] = ey[m - 4];
		ez[i] = ez[m - 4];
		break;
	}
}

cudaError_t efieldWithCuda(
	double* h_ex, double* h_ey, double* h_ez, 
	double* h_by, double* h_bz, 
	double* h_jxe, double* h_jye, double* h_jze, 
	double* h_jxi, double* h_jyi, double* h_jzi, 
	int m, double c) {

	double* d_ex, * d_ey, * d_ez;
	double* d_by, * d_bz;
	double* d_jxe, * d_jye, * d_jze;
	double* d_jxi, * d_jyi, * d_jzi;

	cudaError_t cudaStatus;

	const unsigned ARRAY_BITES = m * sizeof(double);
	cudaMalloc((void**)&d_ex, ARRAY_BITES);
	cudaMalloc((void**)&d_ey, ARRAY_BITES);
	cudaMalloc((void**)&d_ez, ARRAY_BITES);

	cudaMalloc((void**)&d_by, ARRAY_BITES);
	cudaMalloc((void**)&d_bz, ARRAY_BITES);

	cudaMalloc((void**)&d_jxe, ARRAY_BITES);
	cudaMalloc((void**)&d_jye, ARRAY_BITES);
	cudaMalloc((void**)&d_jze, ARRAY_BITES);

	cudaMalloc((void**)&d_jxi, ARRAY_BITES);
	cudaMalloc((void**)&d_jyi, ARRAY_BITES);
	cudaMalloc((void**)&d_jzi, ARRAY_BITES);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_ex, h_ex, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BITES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_by, h_by, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BITES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_jxe, h_jxe, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jye, h_jye, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jze, h_jze, ARRAY_BITES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_jxi, h_jxi, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jyi, h_jyi, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jzi, h_jzi, ARRAY_BITES, cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	efieldKernel << <1, m >> > (d_ex, d_ey, d_ez, d_by, d_bz, d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, m, c);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: efieldKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching efieldKernel!\n", cudaStatus);
		goto Error;
	}

	cudaMemcpy(h_ex, d_ex, ARRAY_BITES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ey, d_ey, ARRAY_BITES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ez, d_ez, ARRAY_BITES, cudaMemcpyDeviceToHost);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: cudaMemcpyDeviceToHost failed!");
		goto Error;
	}

Error:
	cudaFree(d_ex);
	cudaFree(d_ey);
	cudaFree(d_ez);
	cudaFree(d_by);
	cudaFree(d_bz);
	cudaFree(d_jxe);
	cudaFree(d_jye);
	cudaFree(d_jze);
	cudaFree(d_jxi);
	cudaFree(d_jyi);
	cudaFree(d_jzi);

	return cudaStatus;
}
