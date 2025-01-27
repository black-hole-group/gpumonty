
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
#include <curand_kernel.h>


/*GPU Number of blocks and threads*/
#define N_BLOCKS 1792//176
#define N_THREADS 256 //256



/* spectral bin parameters */
#define	dlE (0.1) //Size of the energy bin
//#define lE0	(log(HPL * NUMIN/(ME * CL * CL))) //Minimum energy of the energy bin
#define lE0	(log(1.e-13)) //Minimum energy of the energy bin


/*Number of energy bins (I don't quite know the difference between the two)*/
//Calculate it by doing (int((log(NUMAX/NUMIN)/dlE)))
#define N_ESAMP 400
#define N_EBINS 400

/*Number of theta bins, (90/6) or (180/6) in case of not folding*/
#define N_THBINS	6

/*Compton cross section calculation */
#define MINW      1.e-12       // Minimum weight in the table
#define MAXW      1.e6        // Maximum weight in the table
#define MINT      1.e-4      // Minimum temperature
#define MAXT      1.e4        // Maximum temperature 
#define NW        220          // Number of weight steps for table
#define NT        80           // Number of temperature steps
#define HOTCROSS  "./table/hotcross.dat" // Name of the table file

/*Hot cross routines*/
#define MAXGAMMA	12. //MAX gamma
#define DMUE		0.05 //Stepsize for mu_e
#define DGAMMAE		0.05 //Stepsize for Gamma_e


/*Setting units for the problem*/
/* physical parameters */
#define MMW	0.5		/* mean molecular weight, in units of mp */


/*Mass of the black hole and the unit of M in order to transform to natural code units*/
#define MBH (4.2e2)/*In solar UNITs*/
#define M_UNIT (1.) /*Find this based on the rho_scale parameter for HAMR sims*/



/*Units based off the mass of the blackhole and the Unit of mass*/
#define L_UNIT (GNEWT * MBH * MSUN/(CL * CL)) /* UNIT of length*/
#define T_UNIT (L_UNIT/CL) /*UNIT of time*/
#define RHO_UNIT (M_UNIT / pow(L_UNIT, 3)) /* UNIT of density*/
#define U_UNIT (RHO_UNIT * CL * CL) /*UNITy of energy density*/
#define B_UNIT (CL * sqrt(4. * M_PI * RHO_UNIT)) /*Unit of magnetig field*/
#define NE_UNIT (RHO_UNIT/(MP + ME)) /*Unit of electron density*/


/*Number of spacetime dimensions*/
#define NDIM	4
/*Number of primitive variables*/
#define NPRIM	8
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



/* some useful macros */

#ifdef __CUDA_ARCH__
#define DIM1 d_N1
#define DIM2 d_N2
#define DIM3 d_N3
#else
#define DIM1 N1
#define DIM2 N2
#define DIM3 N3
#endif


#define NPRIM_INDEX3D(i,j,k,l) (i * (DIM1 * DIM2 * DIM3) + ((l) + DIM3 * (k + DIM2 * j))) /*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
#define SPATIAL_INDEX2D(i,j) ((j + DIM2 * i))/*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
#define SPATIAL_INDEX3D(i,j,k) (k+ DIM3*(j + DIM2 * i))
#define NPRIM_INDEX(i,j) (i * (DIM1 * DIM2 * DIM3) + j) /*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
#define SLOOP_DEVICE for(int i=0;i<DIM1;i++)for(int j = 0; j< DIM2; j++)for(int k=0; k < DIM3; k++)

#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
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
#define MAX(a,b) (((a)>(b))?(a):(b))


/*Mmnemonics for hamr*/
#define REF_2 (1)
#define TILT_ANGLE (0.0)
#define LEFT (0)
#define RIGHT (1)
#define FACE1	(0)	
#define FACE2	(1)
#define CORN	(2)
#define CENT	(3)
#define FACE3	(4)


/*Some device global variables definition*/
#define d_lmint     (log10(MINT))
#define d_lminw     (log10(MINW))
#define d_lT_min    (log(TMIN))
#define d_dlw       (log10(MAXW / MINW) / NW)
#define d_dlT       (1/(log(TMAX / TMIN) / (N_ESAMP)))




/*Used for synchrotron emissivity calculation of the table*/ 
#define KMAX (1.e7)
#define KMIN (0.002)
#define SMALL_VECTOR (1.e-30)
#define EPSABS (0.)
#define EPSREL (1.e-6)
#define TMIN (THETAE_MIN)
#define TMAX (1.e2)

/*This define the numbers of scatterins per photon. This is just an approximate to allocate memory, if you don't know, just leave it equal 1.*/
/*If this number is very large and you are still getting invalid memory access, it means that something is wrong with bias, prob*/
#define SCATTERINGS_PER_PHOTON (5) 

/*Making of Nint table*/
#define	NINT (20000) //Number of table data
#define BTHSQMIN	(1.e-8) //Minimum of log(B *thetae^2)
#define BTHSQMAX	(1.e14) //Maximum of log(B *thetae^2)

/*Max number of scatterings*/
#define MAX_LAYER_SCA (4)

/*for stepsize*/
#define EPS (0.04)

/*Push photon routine*/
#define FAST_CPY(in,out) {out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];}
#define ETOL (1.e-3)
#define MAX_ITER (2)

/*track super photon max number of steps*/
#define MAXNSTEP (1280000)
