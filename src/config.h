
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define N_BLOCKS 176//176//30
#define N_THREADS 256//256//256


#define NDIM	4
#define NPRIM	8

/* Range of initial superphoton frequencies */
#define NUMIN 1.e9
#define NUMAX 1.e18

#define THETAE_MAX	1000.	/* Only used for harm3d models */
#define THETAE_MIN	0.3
#define TP_OVER_TE	(3.)

#define WEIGHT_MIN	(1.e31)

#define HAMR (0)
#define HAMR3D (0) /* Leave it equal 0 to do HAMR2D */


/*Setting units for the problem*/
#if(HAMR)
#define MBH (10)/*In solar UNITs*/
#else
#define MBH (4e6)/*In solar UNITs*/
#endif

#if(HAMR)
#define M_UNIT (4e7)
#else
#define M_UNIT (4.e19) /*Try to find rho_scale as this parameter*/
#endif

#define L_UNIT (GNEWT * MBH * MSUN/(CL * CL)) /* UNIT of length*/
#define T_UNIT (L_UNIT/CL) /*UNIT of time*/
#define RHO_UNIT (M_UNIT / pow(L_UNIT, 3)) /* UNIT of density*/
#define U_UNIT (RHO_UNIT * CL * CL) /*UNITy of energy density*/
#define B_UNIT (CL * sqrt(4. * M_PI * RHO_UNIT))
#define NE_UNIT (RHO_UNIT/(MP + ME))

/* mnemonics for primitive vars; conserved vars */
#define KRHO     0
#define UU      1
#define U1      2
#define U2      3
#define U3      4
#define B1      5
#define B2      6
#define B3      7

/* numerical convenience */
#define SMALL	1.e-40

/* physical parameters */
#define MMW	0.5		/* mean molecular weight, in units of mp */
#define NPRIM_INDEX(i,j) (i * (N1 * N2 * N3) + j) /*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/

#define NPRIM_INDEX3D(i,j,k,l) (i * (N1 * N2 * N3) + ((l) + N3 * (k + N2 * j))) /*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
/* some useful macros */
#define SLOOP_DEVICE for(int i=0;i<d_N1;i++)for(int j = 0; j< d_N2; j++)for(int k=0; k < d_N3; k++)
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
#define SPATIAL_INDEX2D(i,j) (((j) + N2 * (i)))
#define SPATIAL_INDEX3D(i,j,k) ((k) + N3 * ((j) + N2 * (i)))
#define SPATIAL_INDEX4D(i,j,k,l) ((l) + NPRIM*((k) + N3 * ((j) + N2 * (i))))
#define INDEX(i,j,k)	(NPRIM*( (k) + N3*((j) + N2*(i))))
#define MYSIN(x,sx) 	{							\
			double _xp = (x)-M_PI; 					\
			double _yp = _xp*(FOUR_PI - FOUR_PISQ*fabs(_xp)); 	\
			sx = -_yp*(0.225*fabs(_yp)+0.775);			\
			}
#define MYCOS(x,cx) 	{							\
			double _xp = (x)-THREEPI_TWO; 					\
			_xp += (_xp<-M_PI)*2.*M_PI; 				\
			double _yp = _xp*(FOUR_PI - FOUR_PISQ*fabs(_xp));		\
			cx = _yp*(0.225*fabs(_yp)+0.775);			\
			}

#define N_ESAMP		200
#define N_EBINS		200
#define N_THBINS	6
#define NW	220
#define NT	80

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
#define DEVICE_SPATIAL_INDEX3D(i,j,k) (k+ d_N3*(j + d_N2 * i))

#define d_lmint     (log10(MINT))
#define d_lminw     (log10(MINW))
#define d_lT_min    (log(TMIN))
#define d_dlw       (log10(MAXW / MINW) / NW)
#define d_dlT       (1/(log(TMAX / TMIN) / (N_ESAMP)))

#define	NINT (20000) //20000

/* spectral bin parameters */
#define	dlE (0.25)
#define lE0	(log(1.e-12))
/*Definitions of other globals*/
#define KFAC	(9*M_PI*ME*CL/EE)
#define KMIN (0.002)
/*Right now, KMAX and KMIN should be changed with careful regarding the function GPU_Inverse_F_eval*/
#define KMAX (1.e7)
#define KMIN (0.002)
#define CST 1.88774862536	/* 2^{11/12} */
#define SMALL_VECTOR	1.e-30
#define EPSABS 0.
#define EPSREL 1.e-6
#define TMIN (THETAE_MIN)
#define TMAX (1.e2)
#define BTHSQMIN	(1.e-8)//Pedro edit from 1e-4 //1e-8 works for riaf
#define BTHSQMAX	(1.e40) //1.e8 Pedro edit  // 1e12 works for riaf

#define MINW	1.e-12
#define MAXW	1.e10
#define MINT	1e-10
#define MAXT	1.e10
#define NW	220
#define NT	80

#define MAX_LAYER_SCA 8
/*for stop criterium*/
#define RMAX	100.
#define ROULETTE	1.e4

/*for stepsize*/
#define EPS   0.04

/*Push photon routine*/
#define FAST_CPY(in,out) {out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];}
#define ETOL 1.e-3
#define MAX_ITER 2

/*Hot cross routines*/
#define MAXGAMMA	12.
#define DMUE		0.05
#define DGAMMAE		0.05
/*track super photon*/
#define MAXNSTEP 1280000