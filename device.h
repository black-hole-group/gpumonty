#ifndef DEVICE_H 
#define DEVICE_H

/*
  DEFINITIONS NEEDED FOR CUDA FUNCTIONS
  ======================================
*/



/*
  device global variables
  ========================
*/
// harm dimensions
 __constant__ int N1, N2, N3; 
 __constant__ int n_within_horizon;

// some coordinate parameters 
 __constant__ double a;
 __constant__ double R0, Rin, Rh, Rout, Rms;
 __constant__ double hslope;
 __constant__ double startx[NDIM], stopx[NDIM], dx[NDIM];
 __constant__ double dlE, lE0;
 __constant__ double gam;
 __constant__ double dMsim;

// units
 __constant__ double M_unit;
 __constant__ double L_unit;
 __constant__ double T_unit;
 __constant__ double RHO_unit;
 __constant__ double U_unit;
 __constant__ double B_unit;
 __constant__ double Ne_unit;
 __constant__ double Thetae_unit;

// misc
__constant__ double max_tau_scatt;
__constant__ double F[N_ESAMP + 1], K2[N_ESAMP + 1]; 
__constant__ double lK_min, dlK;
__constant__ double lT_min, dlT;



/* 
  Device functions
  =================
*/
// replace GSL routines
#include "../cusl/src/cusl_math.h" 
#include "../cusl/src/cheb_eval.cuh"
#include "../cusl/src/poly.cuh"
#include "../cusl/src/psi.cuh"
#include "../cusl/src/sphere.cuh" 
#include "../cusl/src/gamma.cuh" 
#include "../cusl/src/chisq.cuh"   
#include "../cusl/src/bessel.cuh"


#include "tetrads.cuh"
#include "compton.cuh"
#include "harm.cuh" 
#include "jnu_mixed.cuh"
#include "hotcross.cuh"
#include "radiation.cuh"





void launchKernel(double *p, simvars sim, allunits units, settings setup, double *pharr, int nph);

#endif