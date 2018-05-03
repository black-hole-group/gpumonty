#ifndef KERNEL_H
#define KERNEL_H

/*
  Definitions needed for CUDA functions
  ======================================

/* mnemonics for dimensional indices */
#define TT      0
#define RR      1
#define TH      2
#define PH      3



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






#endif