#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parallel.cuh"

// particles
double* d_x;
double* d_vx;
double* d_vy;
double* d_vz;

// cells
double* d_ex;
double* d_ey;
double* d_ez;
double* d_by;
double* d_bz;

double* d_jxe;
double* d_jye;
double* d_jze;
double* d_jxi;
double* d_jyi;
double* d_jzi;

double* d_jxe_s;
double* d_jye_s;
double* d_jze_s;
double* d_jxi_s;
double* d_jyi_s;
double* d_jzi_s;

const unsigned BLOCK_SIZE = 256;
int  nrBlocksCells, nrBlocksParticles;

void initialize(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, double* h_jxe, double* h_jye, double* h_jze, double* h_jxi, double* h_jyi, double* h_jzi, int m, int np) {
	const unsigned long NUMBER_OF_PARTICLES = 2 * np;
	const unsigned long ARRAY_BYTES_CELLS = m * sizeof(double);
	const unsigned long ARRAY_BYTES_PARTICLES = NUMBER_OF_PARTICLES * sizeof(double);

	// particles
	cudaMalloc((void**)&d_x, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vx, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vy, ARRAY_BYTES_PARTICLES);
	cudaMalloc((void**)&d_vz, ARRAY_BYTES_PARTICLES);

	// cells
	cudaMalloc((void**)&d_ex, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_ey, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_ez, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_by, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_bz, ARRAY_BYTES_CELLS);

	cudaMalloc((void**)&d_jxe, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jye, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jze, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jxi, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jyi, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jzi, ARRAY_BYTES_CELLS);

	cudaMalloc((void**)&d_jxe_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jye_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jze_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jxi_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jyi_s, ARRAY_BYTES_CELLS);
	cudaMalloc((void**)&d_jzi_s, ARRAY_BYTES_CELLS);

	// transfer
	cudaMemcpy(d_x, h_x, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, h_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vz, h_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ex, h_ex, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, h_ey, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ez, h_ez, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_by, h_by, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bz, h_bz, ARRAY_BYTES_CELLS, cudaMemcpyHostToDevice);

	nrBlocksParticles = (NUMBER_OF_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
	nrBlocksCells = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

__global__ void bfieldStripeKernel(double* by, double* bz, double* ey, double* ez, int m, double c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < m; i += stride) {
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
}

__global__ void efieldStripeKernel(
	double* ex, double* ey, double* ez,
	double* by, double* bz,
	double* jxe, double* jye, double* jze,
	double* jxi, double* jyi, double* jzi,
	int m, double c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < m; i += stride) {
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
}

__global__ void moverStripeKernel(double* x, double* vx, double* vy, double* vz, double* ex, double* ey, double* ez, double* by, double* bz, \
	double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double c, int np, int m) {
	double ex_p, ey_p, ez_p, by_p, bz_p, qm, gamma_p, eps_x, eps_y, eps_z, beta_x, beta_y, beta_z, ux1, uy1, uz1, wx, wy, wz, \
		ux2, uy2, uz2, ux, uy, uz;

	const unsigned long NUMBER_OF_PARTICLES = 2 * np;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < NUMBER_OF_PARTICLES; k += stride) {
		// particle 'index' is located between i and i+1 and between j+1/2 and j+3/2
		int i = x[k];
		int j = x[k] - 0.5;

		// compute the electric and magnetic fields in the actual position of this particle
		// relocation from half-integer to full-integer grid points is needed for Ex: interpolarion from full-integer grid-points
		ex_p = (i + 1 - x[k]) * 0.5 * (ex[i - 1] + ex[i]) + (x[k] - i) * 0.5 * (ex[i] + ex[i + 1]);

		// no relocation is needed for Ey and Ez: interpolation from half-integer grid-points
		ey_p = (j + 1.5 - x[k]) * ey[j] + (x[k] - j - 0.5) * ey[j + 1];
		ez_p = (j + 1.5 - x[k]) * ez[j] + (x[k] - j - 0.5) * ez[j + 1];

		// relocation from full- to half-integer grid points is needed for By and Bz: interpolation from half-integer grid-points
		by_p = (j + 1.5 - x[k]) * 0.5 * (by[j] + by[j + 1]) + (x[k] - j - 0.5) * 0.5 * (by[j + 1] + by[j + 2]);
		bz_p = (j + 1.5 - x[k]) * 0.5 * (bz[j] + bz[j + 1]) + (x[k] - j - 0.5) * 0.5 * (bz[j + 1] + bz[j + 2]);

		// check if particle 'k' is electron or ion
		if (k < np) {
			//electron
			qm = qme;
		}
		else {
			//ion
			qm = qmi;
		}

		// factor proportional to E-field
		// note that the external electric field is added to the internal one at this step
		eps_x = qm * 0.5 * (ex_p + ex0);
		eps_y = qm * 0.5 * (ey_p + ey0);
		eps_z = qm * 0.5 * (ez_p + ez0);

		// compute the relativistic factor gamma
		gamma_p = 1.0 / sqrt(1.0 - (pow(vx[k], 2.0) + pow(vy[k], 2.0) + pow(vz[k], 2.0)) / pow(c, 2.0));

		// compute u1 as shown in Notebook 5, page 26
		ux1 = gamma_p * vx[k] + eps_x;
		uy1 = gamma_p * vy[k] + eps_y;
		uz1 = gamma_p * vz[k] + eps_z;

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
		vx[k] = ux / gamma_p;
		vy[k] = uy / gamma_p;
		vz[k] = uz / gamma_p;

		// move the particle over one time-step: page 25 on Notebook 5
		x[k] = x[k] + vx[k];
	}
}

__global__ void currentInitializationStripeKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < m; i += stride) {
		jxe[i] = 0.;
		jye[i] = 0.;
		jze[i] = 0.;

		jxi[i] = 0.;
		jyi[i] = 0.;
		jzi[i] = 0.;

		jxe_s[i] = 0.;
		jye_s[i] = 0.;
		jze_s[i] = 0.;

		jxi_s[i] = 0.;
		jyi_s[i] = 0.;
		jzi_s[i] = 0.;
	}
}

__global__ void currentSmoothingStripeKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, double* jxi_s, double* jyi_s, double* jzi_s, int m) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < m; i += stride) {
		if (i >= 2 && i <= m - 4) {
			// electrons
			jxe_s[i] = 0.25 * jxe[i - 1] + 0.5 * jxe[i] + 0.25 * jxe[i + 1];
			jye_s[i] = 0.25 * jye[i - 1] + 0.5 * jye[i] + 0.25 * jye[i + 1];
			jze_s[i] = 0.25 * jze[i - 1] + 0.5 * jze[i] + 0.25 * jze[i + 1];

			// ions		
			jxi_s[i] = 0.25 * jxi[i - 1] + 0.5 * jxi[i] + 0.25 * jxi[i + 1];
			jyi_s[i] = 0.25 * jyi[i - 1] + 0.5 * jyi[i] + 0.25 * jyi[i + 1];
			jzi_s[i] = 0.25 * jzi[i - 1] + 0.5 * jzi[i] + 0.25 * jzi[i + 1];
		}
		else {
			if (i == m - 3) {
				jxe_s[i] = jxe_s[2];
				jye_s[i] = jye_s[2];
				jze_s[i] = jze_s[2];

				jxi_s[i] = jxi_s[2];
				jyi_s[i] = jyi_s[2];
				jzi_s[i] = jzi_s[2];
			}
			else {
				if (i == 1) {
					jxe_s[i] = jxe_s[m - 4];
					jye_s[i] = jye_s[m - 4];
					jze_s[i] = jze_s[m - 4];
					jxi_s[i] = jxi_s[m - 4];
					jyi_s[i] = jyi_s[m - 4];
					jzi_s[i] = jzi_s[m - 4];
				}
			}
		}
	}
}


__global__ void currentStripeKernel(double* jxe, double* jye, double* jze, double* jxi, double* jyi, double* jzi, double* jxe_s, double* jye_s, double* jze_s, \
	double* jxi_s, double* jyi_s, double* jzi_s, double* x, double* vx, double* vy, double* vz, double qse, double qsi, int np, int m) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int k = index; k < m; k += stride) {
		int i, j;
		double xp1, xp2, xp;

		// current density for electrons
		xp1 = x[k] - vx[k];
		xp2 = x[k];

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

		atomicAdd(&jye[j], (j + 1.5 - xp) * qse * vy[k]);
		atomicAdd(&jze[j], (j + 1.5 - xp) * qse * vz[k]);
		atomicAdd(&jye[j + 1], (xp - j - 0.5) * qse * vy[k]);
		atomicAdd(&jze[j + 1], (xp - j - 0.5) * qse * vz[k]);

		// current density for ions
		xp1 = x[k] - vx[k];
		xp2 = x[k];

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

		atomicAdd(&jyi[j], (j + 1.5 - xp) * qsi * vy[k]);
		atomicAdd(&jzi[j], (j + 1.5 - xp) * qsi * vz[k]);
		atomicAdd(&jyi[j + 1], (xp - j - 0.5) * qsi * vy[k]);
		atomicAdd(&jzi[j + 1], (xp - j - 0.5) * qsi * vz[k]);
	}
}

void freeMemory() {
	cudaFree(d_x);
	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_vz);

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

	cudaFree(d_jxe_s);
	cudaFree(d_jye_s);
	cudaFree(d_jze_s);
	cudaFree(d_jxi_s);
	cudaFree(d_jyi_s);
	cudaFree(d_jzi_s);
}


void parallelFunctions(double* h_x, double* h_vx, double* h_vy, double* h_vz, double* h_ex, double* h_ey, double* h_ez, double* h_by, double* h_bz, double* h_jxe, double* h_jye, double* h_jze, double* h_jxi, double* h_jyi, double* h_jzi, \
	double ex0, double ey0, double ez0, double bx0, double by0, double bz0, double qme, double qmi, double qse, double qsi, double c, int m, int np, bool copyToHost) {
	const unsigned long NUMBER_OF_PARTICLES = 2 * np;
	const unsigned long ARRAY_BYTES_CELLS = m * sizeof(double);
	const unsigned long ARRAY_BYTES_PARTICLES = NUMBER_OF_PARTICLES * sizeof(double);

	
	bfieldStripeKernel << <nrBlocksCells, BLOCK_SIZE >> > (d_by, d_bz, d_ey, d_ez, m, c);
	cudaDeviceSynchronize();
	moverStripeKernel<<<nrBlocksParticles, BLOCK_SIZE>>>(d_x, d_vx, d_vy, d_vz, d_ex, d_ey, d_ez, d_by, d_bz, ex0, ey0, ez0, bx0, by0, bz0, qme, qmi, c, np, m);
	cudaDeviceSynchronize();
	bfieldStripeKernel<<<nrBlocksCells, BLOCK_SIZE>>>(d_by, d_bz, d_ey, d_ez, m, c);
	cudaDeviceSynchronize();
	currentInitializationStripeKernel<<<nrBlocksCells, BLOCK_SIZE>>>(d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, m);
	cudaDeviceSynchronize();
	currentStripeKernel<<<nrBlocksParticles, BLOCK_SIZE>>>(d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, d_x, d_vx, d_vy, d_vz, qse, qsi, np, m);
	cudaDeviceSynchronize();
	currentSmoothingStripeKernel<<<nrBlocksCells, BLOCK_SIZE>>>(d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, d_jxe_s, d_jye_s, d_jze_s, d_jxi_s, d_jyi_s, d_jzi_s, m);
	cudaDeviceSynchronize();
	efieldStripeKernel<<<nrBlocksCells, BLOCK_SIZE>>>(d_ex, d_ey, d_ez, d_by, d_bz, d_jxe, d_jye, d_jze, d_jxi, d_jyi, d_jzi, m, c);
	cudaDeviceSynchronize(); 

	if (copyToHost) {
		// copy to host
		cudaMemcpy(h_x, d_x, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vx, d_vx, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vy, d_vy, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_vz, d_vz, ARRAY_BYTES_PARTICLES, cudaMemcpyDeviceToHost);

		cudaMemcpy(h_ex, d_ex, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ey, d_ey, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ez, d_ez, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_by, d_by, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_bz, d_bz, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);

		cudaMemcpy(h_jxi, d_jxi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_jyi, d_jyi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_jzi, d_jzi_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_jxe, d_jxe_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_jye, d_jye_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_jze, d_jze_s, ARRAY_BYTES_CELLS, cudaMemcpyDeviceToHost);
	}
}
