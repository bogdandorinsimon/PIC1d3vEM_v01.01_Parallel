#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void bfieldKernel(double *by, double *bz, double *ey, double *ez, int m, double c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= m) {
		return;
	}

	if (index >= 3 && index <= m - 3) {
		by[index] = by[index] + 0.5 * c * (ez[index] - ez[index - 1]);
		bz[index] = bz[index] - 0.5 * c * (ey[index] - ey[index - 1]);
	}

	if (index == m - 1) {
		by[index] = by[4];
		bz[index] = bz[4];
	}

	if (index == m - 2) {
		by[index] = by[3];
		bz[index] = bz[3];
	}

	switch (index) {
	case 0:
		by[index] = by[m - 5];
		bz[index] = bz[m - 5];
		break;
	case 1:
		by[index] = by[m - 4];
		bz[index] = bz[m - 4];
		break;
	case 2:
		by[index] = by[m - 3];
		bz[index] = bz[m - 3];
		break;
	}
}

cudaError_t bfieldWithCuda(double *h_by, double *h_bz, double *h_ey, double *h_ez, int m, double c) {
	double *d_by, *d_bz, *d_ey, *d_ez;
	cudaError_t cudaStatus;

	const unsigned long ARRAY_BYTES = m * sizeof(double);
	const unsigned long BLOCK_SIZE = 256;
	const unsigned long NUM_OF_BLOCKS = (m - 1) / BLOCK_SIZE + 1;

	cudaMalloc((void**) &d_by, ARRAY_BYTES);
	cudaMalloc((void**) &d_bz, ARRAY_BYTES);
	cudaMalloc((void**) &d_ey, ARRAY_BYTES);
	cudaMalloc((void**) &d_ez, ARRAY_BYTES);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_by, h_by, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfieldWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	bfieldKernel<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(d_by, d_bz, d_ey, d_ez, m, c);
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

	cudaMemcpy(h_by, d_by, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_bz, d_bz, ARRAY_BYTES, cudaMemcpyDeviceToHost);
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
