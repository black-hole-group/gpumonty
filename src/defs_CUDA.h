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
	extern __device__ double d_table[NW + 1][NT + 1];
	extern __device__ double d_maximum_w;

	extern __device__ unsigned long long photon_count;
	extern __device__ unsigned long long generated_sphotons, d_N_superph_recorded;
	extern __device__ int d_N1, d_N2, d_N3, d_Ns, d_N_scatt;
	extern __device__ double d_a, d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1], d_bias_norm, d_stopx[NDIM], d_Rh, d_max_tau_scatt;
		

	extern __device__ unsigned long long scattering_counter;
	extern __device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];
	extern __device__ unsigned long long tracking_counter;
	extern __device__ double d_nint[NINT + 1];
	extern __device__ double d_dndlnu_max[NINT + 1];
	extern __device__ double d_hslope;
	extern __device__ double d_R0;
	extern __device__ int total_sca;

#endif

#ifndef CPUGLOBALS
#define CPUGLOBALS
	extern double * p;
	extern double hslope;
	extern double nint[NINT + 1];
	extern double dndlnu_max[NINT + 1];
#endif