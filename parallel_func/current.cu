#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void currentKernel(double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, double* x, double* vx, double* vy, \
	double* vz, double qse, double qsi, int np, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello world from index %d", index);
}

cudaError_t currentWithCuda(double* h_jxe_s, double* h_jye_s, double* h_jze_s, double* h_jxi_s, double* h_jyi_s, double* h_jzi_s, double* h_x, double* h_vx, double* h_vy, \
	double* h_vz, double qse, double qsi, int np, int m) {
	double* d_jxe_s, *d_jye_s, *d_jze_s, *d_jxi_s,* d_jyi_s,* d_jzi_s, *d_x, *d_vx,* d_vy, *d_vz;
	cudaError_t cudaStatus;

	const unsigned NUMBER_OF_PARTICLES = 2 * np;
	const unsigned ARRAY_BYTES_CELLS = m * sizeof(double);
	const unsigned ARRAY_BYTES_PARTICLES = NUMBER_OF_PARTICLES * sizeof(double);
	const unsigned BLOCK_SIZE = 256;
	const unsigned NUM_OF_BLOCKS = (NUMBER_OF_PARTICLES - 1) / BLOCK_SIZE;

	cudaMalloc((void**)&d_jxe_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jye_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jze_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jxi_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jyi_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jzi_s, ARRAY_BYTES_CELLS);

	cudaMalloc((void**)&d_x, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vx, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vy, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vz, ARRAY_BYTES_PARTICLES);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_jxe_s, h_jxe_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jye_s, h_jye_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jze_s, h_jze_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jxi_s, h_jxi_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jyi_s, h_jyi_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_jzi_s, h_jzi_s, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);

	cudaMemcpy(d_x, h_x, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, h_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vz, h_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	currentKernel<<<NUM_OF_BLOCKS, BLOCK_SIZE >> > (d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, d_x, d_vx, d_vy, d_vz, qse, qsi, np, m);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: currentKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching currentKernel!\n", cudaStatus);
		goto Error;
	}

	/* cudaMemcpy(h_x, d_x, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vx, d_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vy, d_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vz, d_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	*/

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: cudaMemcpyDeviceToHost failed!");
		goto Error;
	}

Error:
	cudaFree(d_jxe_s);
	cudaFree(d_jye_s);
	cudaFree(d_jze_s);
	cudaFree(d_jxi_s);
	cudaFree(d_jyi_s);
	cudaFree(d_jzi_s);

	cudaFree(d_x);
	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_vz);

	return cudaStatus;
}
