#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../utils/nr3.h"

__global__ void moverKernel(double *x, double *vx, double *vy, double *vz, double *ex, double *ey, double *ez, double *by, double*bz, \
		   double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m)
{
	double ex_p, ey_p, ez_p, by_p, bz_p, qm, gamma_p, eps_x, eps_y, eps_z, beta_x, beta_y, beta_z, ux1, uy1, uz1, wx, wy, wz, \
		   ux2, uy2, uz2, ux, uy, uz;
	int index = threadIdx.x;

	// particle 'index' is located between i and i+1 and between j+1/2 and j+3/2
	int i = x[index];
	int j = i - 0.5;

	// compute the electric and magnetic fields in the actual position of this particle
	// relocation from half-integer to full-integer grid points is needed for Ex: interpolarion from full-integer grid-points
	ex_p = (i + 1 - x[index]) * 0.5 * (ex[i - 1] + ex[i]) + (x[index] - i) * 0.5 * (ex[i] + ex[i + 1]);

	// no relocation is needed for Ey and Ez: interpolation from half-integer grid-points
	ey_p = (j + 1.5 - x[index]) * ey[j] + (x[index] - j - 0.5) * ey[j + 1];
	ez_p = (j + 1.5 - x[index]) * ez[j] + (x[index] - j - 0.5) * ez[j + 1];

	// relocation from full- to half-integer grid points is needed for By and Bz: interpolation from half-integer grid-points
	by_p = (j + 1.5 - x[index]) * 0.5 * (by[j] + by[j + 1]) + (x[index] - j - 0.5) * 0.5 * (by[j + 1] + by[j + 2]);
	bz_p = (j + 1.5 - x[index]) * 0.5 * (bz[j] + bz[j + 1]) + (x[index] - j - 0.5) * 0.5 * (bz[j + 1] + bz[j + 2]);

	// check if particle 'k' is electron or ion
	qm = index < np ? qme : qmi;

	// factor proportional to E-field
	// note that the external electric field is added to the internal one at this step
	eps_x = qm * 0.5 * (ex_p + ex0);
	eps_y = qm * 0.5 * (ey_p + ey0);
	eps_z = qm * 0.5 * (ez_p + ez0);

	// compute the relativistic factor gamma
	gamma_p = 1.0 / sqrt(1.0 - (pow(vx[index], 2.0) + pow(vy[index], 2.0) + pow(vz[index], 2.0)) / pow(c, 2.0));

	// compute u1 as shown in Notebook 5, page 26
	ux1 = gamma_p * vx[index] + eps_x;
	uy1 = gamma_p * vy[index] + eps_y;
	uz1 = gamma_p * vz[index] + eps_z;

	// compute the new relativistic factor
	gamma_p = sqrt(1.0 + (pow(ux1, 2.0) + pow(uy1, 2.0) + pow(uz1, 2.0)) / pow(c, 2.0));

	// factor proportional to B-field
	// note that the external magnetic field is added to the internal one at this step
	beta_x = (qm * 0.5 / gamma_p) * (bx0 / c);
	beta_y = (qm * 0.5 / gamma_p) * ((by_p + by0) / c);
	beta_z = (qm * 0.5 / gamma_p) * ((bz_p + bz0) / c);

	// intermediate quantity
	wx = ux1 + uy1 * beta_z - uz1 * beta_y;
	wy = uy1 + uz1 * beta_x - ux1 * beta_z;
	wz = uz1 + ux1 * beta_y - uy1 * beta_x;

	// compute u2 as shown in Notebook 5, page 26
	ux2 = ux1 + (2 / (1 + pow(beta_x, 2.0) + pow(beta_y, 2.0) + pow(beta_z, 2.0))) * (wy * beta_z - wz * beta_y);
	uy2 = uy1 + (2 / (1 + pow(beta_x, 2.0) + pow(beta_y, 2.0) + pow(beta_z, 2.0))) * (wz * beta_x - wx * beta_z);
	uz2 = uz1 + (2 / (1 + pow(beta_x, 2.0) + pow(beta_y, 2.0) + pow(beta_z, 2.0))) * (wx * beta_y - wy * beta_x);

	// compute u as shown in Notebook 5, page 26
	ux = ux2 + eps_x;
	uy = uy2 + eps_y;
	uz = uz2 + eps_z;

	// compute the new relativistic factor
	gamma_p = sqrt(1.0 + (pow(ux, 2.0) + pow(uy, 2.0) + pow(uz, 2.0)) / pow(c, 2.0));

	// compute the new velocity after one time-step: page 26, Notebook 5
	vx[index] = ux / gamma_p;
	vy[index] = uy / gamma_p;
	vz[index] = uz / gamma_p;

	// move the particle over one time-step: page 25 on Notebook 5
	x[index] = x[index] + vx[index];
}

cudaError_t moverWithCuda(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, \
double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m) {
	double* d_x, double* d_vx, double* d_vy, double* d_vz, double* d_ex, double* d_ey, double* d_ez, double* d_by, double* d_bz;
	cudaError_t cudaStatus;

	const unsigned NUMBER_OF_PARTICLES = 2 * np;
	const unsigned ARRAY_BYTES_CELLS = m * sizeof(double);
	const unsigned ARRAY_BYTES_PARTICLES = NUMBER_OF_PARTICLES * sizeof(double);

	cudaMalloc((void**)&d_x, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vx, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vy, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vz, ARRAY_BYTES_PARTICLES);

	cudaMalloc((void**)&d_ex, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_ey, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_ez, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_by, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_bz, ARRAY_BYTES_CELLS);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moverWithCuda: cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(d_x, h_x, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, h_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vz, h_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ex, h_ex, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_by, h_by, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moverWithCuda: cudaMemcpyHostToDevice failed!");
		goto Error;
	}

	moverKernel<<<1, NUMBER_OF_PARTICLES>>>(d_x, d_vx, d_vy, d_vz, d_ex, d_ey, d_ez, d_by, d_bz, ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, c, np, m);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moverWithCuda: moverKernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching moverKernel!\n", cudaStatus);
		goto Error;
	}

	cudaMemcpy(h_x, d_x, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vx, d_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vy, d_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vz, d_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moverWithCuda: cudaMemcpyDeviceToHost failed!");
		goto Error;
	}

Error:
	cudaFree(d_x);
	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_vz);

	cudaFree(d_ex);
	cudaFree(d_ey);
	cudaFree(d_ez);
	cudaFree(d_by);
	cudaFree(d_bz);

	return cudaStatus;
}
