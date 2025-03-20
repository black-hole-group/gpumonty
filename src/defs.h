#include "config.h"
struct of_geom *geom;
gsl_rng *r;
gsl_integration_workspace *w;
#pragma omp threadprivate(r)
/*Global Variable Section*/
/* defining declarations for global variables */
int N1, N2, N3, n_within_horizon;
double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
int Ns, N_scatt;
unsigned long long N_superph_recorded;

/* some coordinate parameters */
double a;
double R0, Rin, Rh, Rout, Rms;
double hslope;
double startx[NDIM], stopx[NDIM], dx[NDIM];
//double dlE, lE0;
double gam;
double dMsim;
double Thetae_unit;
double max_tau_scatt, Ladv, dMact, bias_norm;
/*Model Dependent*/
double *****econ;
double *****ecov;
double ****bcon;
double ****bcov;
double ****ucon;
double ****ucov;
double * p;
double ***ne;
double ***thetae;
double ***b;


double table[NW + 1][NT + 1];
double dlw, dlT, lminw, lmint; 
double nint[NINT + 1];
double K2[N_ESAMP + 1];
double dndlnu_max[NINT + 1];

__device__ curandState my_curand_state[N_BLOCKS * N_THREADS]; // Array of curandState structures

__device__ double d_table[NW + 1][NT + 1];

__device__ unsigned long long photon_count = 0;
__device__ unsigned long long generated_sphotons, d_N_superph_recorded;
__device__ int d_N1, d_N2, d_N3, d_Ns, d_N_scatt;
__device__ double d_a, d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1], d_bias_norm, d_stopx[NDIM], d_Rh, d_max_tau_scatt;
	
__device__ unsigned long long scattering_counter = 0;
__device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];
__device__ unsigned long long tracking_counter = 0;
__device__ double d_nint[NINT + 1];
__device__ double d_dndlnu_max[NINT + 1];
__device__ double d_hslope = 0;
__device__ double d_R0 = 0;
__device__ int total_sca = 0;
__device__ unsigned long long tracking_counter_sampling = 0;
__device__ unsigned long long d_number_of_geodesics = 0;
__device__ unsigned long long sc_ph_next = 0;
__device__ unsigned long long sc_ph_current= 0;

/** data structures **/




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
	double E0;
};




