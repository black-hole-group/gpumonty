
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

#define NDIM	4
#define NPRIM	8

/* Range of initial superphoton frequencies */
#define NUMIN 1.e9
#define NUMAX 1.e16

#define THETAE_MAX	1000.	/* Only used for harm3d models */
#define THETAE_MIN	0.3
#define TP_OVER_TE	(3.)

#define WEIGHT_MIN	(1e31)

#define HAMR (0)
#define HAMR3D (0) /* Leave it equal 0 to do HAMR2D */


/*Setting units for the problem*/
#if(HAMR)
#define MBH (10)/*In solar UNITs*/
#else
#define MBH (4e6)/*In solar UNITs*/
#endif

#if(HAMR)
#define M_UNIT (3.2e10)
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
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
#define SPATIAL_INDEX2D(i,j) ((j + N2 * i))
#define SPATIAL_INDEX3D(i,j,k) ((k) + N3 * (j + N2 * i))
#define SPATIAL_INDEX4D(i,j,k,l) (l + NPRIM*((k) + N3 * (j + N2 * i)))
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
