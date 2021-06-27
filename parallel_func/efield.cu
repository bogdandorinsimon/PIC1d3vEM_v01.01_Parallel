#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void efieldKernel(
	double* ex, double* ey, double* ez,
	double* by, double* bz, 
	double* jxe, double* jye, double* jze, 
	double* jxi, double* jyi, double* jzi, 
	int m, double c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= m) {
		return;
	}

	if (index >= 2 && index < m - 3) {
		ex[index] = ex[index] - (jxe[index] + jxi[index]);
		ey[index] = ey[index] - (jye[index] + jyi[index]) - c * (bz[index + 1] - bz[index]);
		ez[index] = ez[index] - (jze[index] + jzi[index]) + c * (by[index + 1] - by[index]);
	}


	if (index == m - 1) {
		ex[index] = ex[4];
		ey[index] = ey[4];
		ez[index] = ez[4];
	}

	if (index == m - 2) {
		ex[index] = ex[3];
		ey[index] = ey[3];
		ez[index] = ez[3];
	}

	if (index == m - 3) {
		ex[index] = ex[2];
		ey[index] = ey[2];
		ez[index] = ez[2];
	}

	switch (index) {
	case 0:
		ex[index] = ex[m - 5];
		ey[index] = ey[m - 5];
		ez[index] = ez[m - 5];
		break;
	case 1:
		ex[index] = ex[m - 4];
		ey[index] = ey[m - 4];
		ez[index] = ez[m - 4];
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

	const unsigned long ARRAY_BYTES = m * sizeof(double);
	const unsigned long BLOCK_SIZE = 256;
	const unsigned long NUM_OF_BLOCKS = (m - 1) / BLOCK_SIZE + 1;

	cudaMalloc((void**)&d_ex, ARRAY_BYTES);
	cudaMalloc((void**)&d_ey, ARRAY_BYTES);
	cudaMalloc((void**)&d_ez, ARRAY_BYTES);

	cudaMalloc((void**)&d_by, ARRAY_BYTES);
	cudaMalloc((void**)&d_bz, ARRAY_BYTES);

	cudaMalloc((void**)&d_jxe, ARRAY_BYTES);
	cudaMalloc((void**)&d_jye, ARRAY_BYTES);
	cudaMalloc((void**)&d_jze, ARRAY_BYTES);

	cudaMalloc((void**)&d_jxi, ARRAY_BYTES);
	cudaMalloc((void**)&d_jyi, ARRAY_BYTES);
	cudaMalloc((void**)&d_jzi, ARRAY_BYTES);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_ex, h_ex, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_by, h_by, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_jxe, h_jxe, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jye, h_jye, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jze, h_jze, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_jxi, h_jxi, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jyi, h_jyi, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jzi, h_jzi, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "efieldWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	efieldKernel << <NUM_OF_BLOCKS, BLOCK_SIZE>> > (d_ex, d_ey, d_ez, d_by, d_bz, d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, m, c);
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

	cudaMemcpy(h_ex, d_ex, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ey, d_ey, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ez, d_ez, ARRAY_BYTES, cudaMemcpyDeviceToHost);

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
