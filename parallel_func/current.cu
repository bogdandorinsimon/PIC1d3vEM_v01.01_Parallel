#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parallel.cuh"

__global__ void currentInitializationKernel(double *jxe, double *jye, double *jze, double *jxi, double *jyi, double *jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	jxe[index] = 0.;
	jye[index] = 0.;
	jze[index] = 0.;

	jxi[index] = 0.;
	jyi[index] = 0.;
	jzi[index] = 0.;

	jxe_s[index] = 0.;
	jye_s[index] = 0.;
	jze_s[index] = 0.;

	jxi_s[index] = 0.;
	jyi_s[index] = 0.;
	jzi_s[index] = 0.;
}

__global__ void currentSmoothingKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, int m) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= 2 && index <= m - 4) {
		// electrons
		jxe_s[index] = 0.25 * jxe[index - 1] + 0.5 * jxe[index] + 0.25 * jxe[index + 1];
		jye_s[index] = 0.25 * jye[index - 1] + 0.5 * jye[index] + 0.25 * jye[index  + 1];
		jze_s[index] = 0.25 * jze[index - 1] + 0.5 * jze[index] + 0.25 * jze[index + 1];

		// ions		
		jxi_s[index] = 0.25 * jxi[index - 1] + 0.5 * jxi[index] + 0.25 * jxi[index + 1];
		jyi_s[index] = 0.25 * jyi[index - 1] + 0.5 * jyi[index] + 0.25 * jyi[index + 1];
		jzi_s[index] = 0.25 * jzi[index - 1] + 0.5 * jzi[index] + 0.25 * jzi[index + 1];
	}
	else {
		if (index == m - 3) {
			jxe_s[index] = jxe_s[2];
			jye_s[index] = jye_s[2];
			jze_s[index] = jze_s[2];

			jxi_s[index] = jxi_s[2];
			jyi_s[index] = jyi_s[2];
			jzi_s[index] = jzi_s[2];
		}
		else {
			if (index == 1) {
				jxe_s[index] = jxe_s[m - 4];
				jye_s[index] = jye_s[m - 4];
				jze_s[index] = jze_s[m - 4];
				jxi_s[index] = jxi_s[m - 4];
				jyi_s[index] = jyi_s[m - 4];
				jzi_s[index] = jzi_s[m - 4];
			}
		}
	}
}


__global__ void currentKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, \
	double* jxi_s, double* jyi_s, double* jzi_s, double* x, double* vx, double* vy, double* vz, double qse, double qsi, int np, int m) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int i, j;
	double xp1, xp2, xp;

	// current density for electrons
	xp1 = x[index] - vx[index];
	xp2 = x[index];

	i = xp1;

	if (xp2 < i) {
		atomicAdd(&jxe[i - 1], qse * (xp2 - i));
		atomicAdd(&jxe[i], qse * (i - xp1));
	}
	else {
		if (xp2 > (i + 1)) {
			atomicAdd(&jxe[i], qse * (i + 1 - xp1));
			atomicAdd(&jxe[i + 1], qse * (xp2 - i - 1));
		}
		else {
			atomicAdd(&jxe[i], qse * (xp2 - xp1));
		}
	}

	xp = (xp1 + xp2) / 2;
	j = xp - 0.5;

	atomicAdd(&jye[j], (j + 1.5 - xp) * qse * vy[index]);
	atomicAdd(&jze[j], (j + 1.5 - xp) * qse * vz[index]);
	atomicAdd(&jye[j + 1], (xp - j - 0.5) * qse * vy[index]);
	atomicAdd(&jze[j + 1], (xp - j - 0.5) * qse * vz[index]);

	// current density for ions
	xp1 = x[index] - vx[index];
	xp2 = x[index];

	i = xp1;
	if (xp2 < i) {
		atomicAdd(&jxi[i - 1], qsi * (xp2 - i));
		atomicAdd(&jxi[i], qsi * (i - xp1));
	}
	else {
		if (xp2 > (i + 1)) {
			atomicAdd(&jxi[i], qsi * (i + 1 - xp1));
			atomicAdd(&jxi[i + 1], qsi * (xp2 - i - 1));
		}
		else {
			atomicAdd(&jxi[i], qsi * (xp2 - xp1));
		}
	}

	xp = (xp1 + xp2) / 2;
	j = xp - 0.5;

	atomicAdd(&jyi[j], (j + 1.5 - xp) * qsi * vy[index]);
	atomicAdd(&jzi[j], (j + 1.5 - xp) * qsi * vz[index]);
	atomicAdd(&jyi[j + 1], (xp - j - 0.5) * qsi * vy[index]);
	atomicAdd(&jzi[j + 1], (xp - j - 0.5) * qsi * vz[index]);
}


cudaError_t currentWithCuda(double* h_jxe_s, double* h_jye_s, double* h_jze_s, double* h_jxi_s, double* h_jyi_s, double* h_jzi_s, double* h_x, double* h_vx, double* h_vy, \
	double* h_vz, double qse, double qsi, int np, int m) {
	double* d_jxe, * d_jye, * d_jze, * d_jxi, * d_jyi, * d_jzi;
	double* d_jxe_s, *d_jye_s, *d_jze_s, *d_jxi_s,* d_jyi_s,* d_jzi_s, *d_x, *d_vx,* d_vy, *d_vz;
	cudaError_t cudaStatus;

	const unsigned NUMBER_OF_PARTICLES = 2 * np;
	const unsigned ARRAY_BYTES_CELLS = m * sizeof(double);
	const unsigned ARRAY_BYTES_PARTICLES = NUMBER_OF_PARTICLES * sizeof(double);
	const unsigned BLOCK_SIZE = 256;
	const unsigned NUM_OF_BLOCKS_PARTICLES = (NUMBER_OF_PARTICLES - 1) / BLOCK_SIZE;
	const unsigned NUM_OF_BLOCKS_CELLS = (m - 1) / BLOCK_SIZE;

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

	/* aux arrays */
	cudaMalloc((void**)&d_jxe, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jye, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jze, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jxi, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jyi, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jzi, ARRAY_BYTES_CELLS);

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
		getchar();
		goto Error;
	}

	// initialization
	currentInitializationKernel<<<NUM_OF_BLOCKS_CELLS, BLOCK_SIZE>>>(d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: currentInitializationKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching currentInitializationKernel!\n", cudaStatus);
		goto Error;
	}
	// end initialization
	
	
	// current
	currentKernel<<<NUM_OF_BLOCKS_PARTICLES, BLOCK_SIZE >> > (d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, d_x, d_vx, d_vy, d_vz, qse, qsi, np, m);
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
	// end current

	// smoothing
	currentSmoothingKernel<<<NUM_OF_BLOCKS_CELLS, BLOCK_SIZE >> > (d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, m);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "currentWithCuda: currentSmoothingKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching currentSmoothingKernel!\n", cudaStatus);
		goto Error;
	}
	// end smoothing

	cudaMemcpy(h_jxe_s, d_jxe_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jye_s, d_jye_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jze_s, d_jze_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jxi_s, d_jxi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jyi_s, d_jyi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jzi_s, d_jzi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_x, d_x, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vx, d_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vy, d_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vz, d_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);

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
