
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define N_BLOCKS 3456//176
#define N_THREADS 256 //256



/* spectral bin parameters */
#ifdef SPHERE_TEST
    #define	dlE (0.06) //Size of the energy bin
    #define lE0	(log(1.e-12)) //Minimum energy of the energy bin
    #define N_ESAMP 800
    #define N_EBINS 800
#else
    // #define	dlE (0.25) //Size of the energy bin
    // #define lE0	(log(1.e-12)) //Minimum energy of the energy bin
    // #define N_ESAMP 200
    // #define N_EBINS 200
    #define	dlE (0.12) //Size of the energy bin
    #define lE0	(log(1.e-12)) //Minimum energy of the energy bin
    #define N_ESAMP 800
    #define N_EBINS 800
#endif


/*Number of theta bins, (90/6) or (180/6) in case of not folding*/
#define N_THBINS	6

/*Compton cross section calculation */
#define MINW      1.e-12       // Minimum weight in the table
#define MAXW      1.e15        // Maximum weight in the table
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

/*Units based off the mass of the blackhole and the Unit of mass*/
#define T_UNIT (L_UNIT/CL) /*UNIT of time*/
#define U_UNIT (RHO_UNIT * CL * CL) /*UNITy of energy density*/
#define B_UNIT (CL * sqrt(4. * M_PI * RHO_UNIT)) /*Unit of magnetig field*/
#define NE_UNIT (RHO_UNIT/(MP + ME)) /*Unit of electron density*/


/*Number of spacetime dimensions*/
#define NDIM	4
/* mnemonics for primitive vars; conserved vars */
#define KRHO    0
#define UU      1
#define U1      2
#define U2      3
#define U3      4
#define B1      5
#define B2      6
#define B3      7
#define KEL     8 
#define KTOT    9
/* numerical convenience */
#define SMALL	1.e-40


//#define NPRIM_INDEX3D(v,i,j,k) (((v) * N1 * N2 * N3) + (k + N3 * (j + N2 * i)))


// #define SPATIAL_INDEX2D(i,j) ((j + N2 * i))/*i should be mmenemonics for memory, j, k, l should be 3D spatial index for dimensions with N1, N2 and N3*/
// #define SPATIAL_INDEX3D(i,j,k) (k+ N3*(j + N2 * i))


#define NPRIM_INDEX(i, j) (j * NPRIM + i)
// #define SLOOP_DEVICE for(int i=0;i<N1;i++)for(int j = 0; j< N2; j++)for(int k=0; k < N3; k++)

#ifdef __CUDA_ARCH__
#define SLOOP_DEVICE(i,j,k) \
    for (int i = 0; i < d_N1; i++) \
        for (int j = 0; j < d_N2; j++) \
            for (int k = 0; k < d_N3; k++)
#else
#define SLOOP_DEVICE(i,j,k) \
    for (int i = 0; i < N1; i++) \
        for (int j = 0; j < N2; j++) \
            for (int k = 0; k < N3; k++)
#endif

#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)

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
#define d_dlT1       (1/(log(TMAX / TMIN) / (N_ESAMP)))
#define d_dlT2       (log10(MAXT/MINT) / NT)




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
#define SCATTERINGS_PER_PHOTON (1) 

/*Making of Nint table*/
#define	NINT (40000) //Number of table data
#define BTHSQMIN	(1.e-8) //Minimum of log(B *thetae^2)
#define BTHSQMAX	(1.e9) //Maximum of log(B *thetae^2)

/*Max number of scatterings*/
#define MAX_LAYER_SCA (3)

/*for stepsize*/
#define EPS (0.04)

/*Push photon routine*/
#define FAST_CPY(in,out) {out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];}
#define ETOL (1.e-3)
#define MAX_ITER (2)

/*track super photon max number of steps*/
#define MAXNSTEP (1280000)
