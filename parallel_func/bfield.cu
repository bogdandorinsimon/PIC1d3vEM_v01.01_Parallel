#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void bfieldKernel(double *by, double *bz, double *ey, double *ez, int m, double c) {
	int i = threadIdx.x;

	if (i >= 3 && i <= m - 3) {
		by[i] = by[i] + 0.5 * c * (ez[i] - ez[i - 1]);
		bz[i] = bz[i] - 0.5 * c * (ey[i] - ey[i - 1]);
	}

	if (i == m - 1) {
		by[i] = by[4];
		bz[i] = bz[4];
	}

	if (i == m - 2) {
		by[i] = by[3];
		bz[i] = bz[3];
	}

	switch (i) {
	case 0:
		by[i] = by[m - 5];
		bz[i] = bz[m - 5];
		break;
	case 1:
		by[i] = by[m - 4];
		bz[i] = bz[m - 4];
		break;
	case 2:
		by[i] = by[m - 3];
		bz[i] = bz[m - 3];
		break;
	}
}

cudaError_t bfieldWithCuda(double *h_by, double *h_bz, double *h_ey, double *h_ez, int m, double c) {
	double *d_by, *d_bz, *d_ey, *d_ez;
	cudaError_t cudaStatus;

	const unsigned ARRAY_BITES = m * sizeof(double);
	cudaMalloc((void**) &d_by, ARRAY_BITES);
	cudaMalloc((void**) &d_bz, ARRAY_BITES);
	cudaMalloc((void**) &d_ey, ARRAY_BITES);
	cudaMalloc((void**) &d_ez, ARRAY_BITES);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_by, h_by, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BITES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BITES, cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	bfieldKernel<<<1, m>>>(d_by, d_bz, d_ey, d_ez, m, c);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: bfieldKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bfieldKernel!\n", cudaStatus);
		goto Error;
	}

	cudaMemcpy(h_by, d_by, ARRAY_BITES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_bz, d_bz, ARRAY_BITES, cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: cudaMemcpyDeviceToHost failed!");
		goto Error;
	}

Error:
	cudaFree(d_by);
	cudaFree(d_bz);
	cudaFree(d_ey);
	cudaFree(d_ez);

	return cudaStatus;
}
