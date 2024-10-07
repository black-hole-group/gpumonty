#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "constants.h"
#include "config.h"
#include "gpu_header.h"


/*GLOBAL VARIABLES*/
/*We need to be carefull with global variables that are modified by multiple threads at a time. We can use global variables, but just
do not edit with multiple threads, unless we know what we are doing*/

#ifndef GPUGLOBALS
#define GPUGLOBALS
	__device__ double d_table[NW + 1][NT + 1];
	__device__ double d_maximum_w = 0;

	__device__ unsigned long long photon_count = 0;
	__device__ unsigned long long generated_sphotons, d_N_superph_recorded;
	__device__ int d_N1, d_N2, d_N3, d_Ns, d_N_scatt;
	__device__ double d_a, d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1], d_bias_norm, d_stopx[NDIM], d_Rh, d_max_tau_scatt;
		

	__device__ unsigned long long scattering_counter = 0;
	__device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];
	__device__ unsigned long long tracking_counter = 0;
	__device__ double d_nint[NINT + 1];
	__device__ double d_dndlnu_max[NINT + 1];
	__device__ double d_hslope = 0;
	__device__ double d_R0 = 0;
	__device__ int total_sca = 0;



	struct of_scattering{
		int bound_flag;
		double dtau_scatt, dtau_abs, dtau;
		double bi, bf;
		double alpha_scatti, alpha_scattf;
		double alpha_absi, alpha_absf;
		double dl, x1;
		double nu, Thetae, Ne, B, theta;
		double dtauK, frac;
		double bias;
		double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
		double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
		int nstep;
	};
#endif