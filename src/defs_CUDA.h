#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <omp.h>
#include "constants.h"
#include <cuda_runtime.h>
#include "config.h"

/*Definitions of other globals*/
#define KFAC	(9*M_PI*ME*CL/EE)
#define KMIN (0.002)
#define KMAX (1.e7)
#define CST 1.88774862536	/* 2^{11/12} */
#define SMALL_VECTOR	1.e-30
#define EPSABS 0.
#define EPSREL 1.e-6
#define KMIN (0.002)
#define KMAX (1.e7)
#define TMIN (THETAE_MIN)
#define TMAX (1.e2)
#define BTHSQMIN	(1.e-8)//Pedro edit from 1e-4 //1e-8 works for riaf
#define BTHSQMAX	(1.e14) //1.e8 Pedro edit  // 1e12 works for riaf
#define	NINT		(20000) //20000

#define MINW	1.e-12
#define MAXW	1.e6
#define MINT	0.0001
#define MAXT	1.e4
#define NW	220
#define NT	80


/*GLOBAL VARIABLES*/
__device__ int d_N1, d_N2, d_N3, d_Ns;
__device__ double d_a, d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1];;
__device__ double * d_p;
__device__ double d_nint[NINT + 1];
__device__ double d_dndlnu_max[NINT + 1];
__device__ 	int zone_flag;

#define d_lT_min    (log(TMIN))
#define d_dlw       (log10(MAXW / MINW) / NW)
#define d_dlT       (log10(MAXT / MINT) / NT)


#define REF_2 (1)
#define TILT_ANGLE (0.0)
#define LEFT (0)
#define RIGHT (1)
#define FACE1	(0)	
#define FACE2	(1)
#define CORN	(2)
#define CENT	(3)
#define FACE3	(4)

#define DEVICE_NPRIM_INDEX3D(i,j,k,l) (i * (d_N1 * d_N2 * d_N3) + ((l) + d_N3 * (k + d_N2 * j))) /*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
#define DEVICE_SPATIAL_INDEX2D(i,j) ((j + d_N2 * i))/*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
