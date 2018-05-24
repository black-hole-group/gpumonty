#ifndef KERNEL_H
#define KERNEL_H

/*
  Definitions needed for CUDA functions
  ======================================

  Some of them are repeated from decs.h.
*/
#define NDIM	4
#define NPRIM	8

/* number of variables necessary for each photon.
   change this for 3D GRMHD simulations
*/
#define NPHVARS 25 

/* Range of initial superphoton frequencies */
#define NUMIN 1.e9
#define NUMAX 1.e16

#define THETAE_MAX	1000.	/* Only used for harm3d models */
#define THETAE_MIN	0.3
#define TP_OVER_TE	3.

#define WEIGHT_MIN 1.e31

/* numerical convenience */
#define SMALL	1.e-40

/* physical parameters */
#define MMW	0.5		/* mean molecular weight, in units of mp */


/* mnemonics for primitive HARM vars; conserved vars */
#define KRHO     0
#define UU      1
#define U1      2
#define U2      3
#define U3      4
#define B1      5
#define B2      6
#define B3      7

/* 
  Mnemonics for the different photon variables
*/
#define X0      0
#define X1      1
#define X2      2
#define X3      3
#define K0_      4
#define K1_      5
#define K2_      6
#define K3_      7
#define D0      8
#define D1      9
#define D2      10
#define D3      11
#define W       12
#define E_       13
#define L_       14
#define X1I     15
#define X2I     16
#define TAUA    17
#define TAUS    18
#define NE0     19
#define TH0     20
#define B0      21
#define E0_      22
#define E0S     23
#define NS      24 // this guy was an int before, now it will be a double or float

/* mnemonics for dimensional indices */
#define TT      0
#define RR      1
#define TH      2
#define PH      3


/* EPS really ought to be related to the number of
   zones in the simulation. Used in harm_mode/stepsize.
*/
#define EPS	0.04

/* some useful macros */
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
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







/* Data structures
   ===============

   Required for passing variables to device
*/

// Data structure carrying all GRMHD variables
struct simvars { 
	// harm dimensions
	int N1, N2, N3; 
	int n_within_horizon;

	/* some coordinate parameters */
	double a;
	double R0, Rin, Rh, Rout, Rms;
	double hslope;
	double startx[NDIM], stopx[NDIM], dx[NDIM];
	double dlE, lE0;
	double gam;
	double dMsim;
};


struct allunits {
	double M_unit;
	double L_unit;
	double T_unit;
	double RHO_unit;
	double U_unit;
	double B_unit;
	double Ne_unit;
	double Thetae_unit;
};



/* Data structure needed for reusing many functions previously
   written for the host, in the device.

   THIS IS PROBABLY AN UNNECESSARY REPRETITION OF PHOTON STRUCT,
   REMOVE IN THE FUTURE
*/
struct d_photon { 
	double X[NDIM];
	double K[NDIM];
	double dKdlam[NDIM];
	double w;
	double E;
	double L;
	double X1i;
	double X2i;
	double tau_abs;
	double tau_scatt;
	double ne0;
	double thetae0;
	double b0;
	double E0;
	double E0s;
	int nscatt;
};



void launchKernel(double *p, simvars sim, allunits units, double max_tau_scatt, double *pharr, int nph);

#endif