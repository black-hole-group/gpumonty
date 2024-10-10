
#include "config.h"
#include "functions.h"
#include "model.h"



/** data structures **/
struct of_photon {
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


struct of_geom {
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

struct of_spectrum {
	double dNdlE;
	double dEdlE;
	double nph;
	double nscatt;
	double X1iav;
	double X2isq;
	double X3fsq;
	double tau_abs;
	double tau_scatt;
	double ne0;
	double thetae0;
	double b0;
	double E0;
};

struct of_grid {
	struct of_spectrum spec[N_EBINS];
	double th, phi;
	int nlist;
	int *in;
};


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
	extern double K2[N_ESAMP + 1];
	/*Global Variable Section*/
	/* defining declarations for global variables */
	extern struct of_spectrum spect[N_THBINS][N_EBINS];

	extern struct of_geom *geom;
	extern int N1, N2, N3, n_within_horizon;
	extern double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
	extern double table[NW + 1][NT + 1];

	extern int Ns, N_scatt;
	extern unsigned long long N_superph_recorded;

	/* some coordinate parameters */
	extern double a;
	extern double R0, Rin, Rh, Rout, Rms;
	extern double hslope;
	extern double startx[NDIM], stopx[NDIM], dx[NDIM];

	//extern double dlE, lE0;
	extern double gam;
	extern double dMsim;
	extern double Thetae_unit;
	extern double max_tau_scatt, Ladv, dMact, bias_norm;


	extern gsl_rng *r;

#endif