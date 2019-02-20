#ifndef _DECS_H
#define _DECS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <gsl/gsl_rng.h>
// #include <gsl/gsl_randist.h>
// #include <gsl/gsl_math.h>
// #include <gsl/gsl_sf_bessel.h>
// #include <gsl/gsl_matrix.h>
// #include <gsl/gsl_vector.h>
// #include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <openacc.h>
#include "config.h"
#include "constants.h"
#include "rng.h"

#define NDIM	4
#define NPRIM	8

/* Range of initial superphoton frequencies */
#define NUMIN 1.e9
#define NUMAX 1.e20

#define THETAE_MAX	1000.
#define THETAE_MIN	0.3
#define TP_OVER_TE	(3.)
#define WEIGHT_MIN	(1.e28)

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

#define N_ESAMP		200
#define N_EBINS		200
#define N_THBINS	6

#define dlE (0.25) /* bin width */
#define lE0 (-27.631021115928547) /* location of first bin, in electron
									rest-mass units. log(1.e-12) */


/* some useful macros */
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
#define MIN(elem1,elem2) ((elem1) < (elem2) ? (elem1) : (elem2))
// #define INDEX(i,j,k)	(NPRIM*( (k) + N3*((j) + N2*(i))))

// phothon's tracking status
#define TRACKING_STATUS_INCOMPLETE 0
#define TRACKING_STATUS_COMPLETE 1
#define TRACKING_STATUS_POSTPONED 2

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
	char tracking_status;
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


/** global variables **/
/** model independent */
extern struct of_geom **geom;
extern double Ladv, dMact, bias_norm;
extern double gam;
extern double dMsim;
extern double M_unit;
extern double T_unit;
extern double RHO_unit;
extern double U_unit;
// unsigned long long N_scatt = 0;
// int n_within_horizon;

// Gpu used variables
extern double startx[NDIM], stopx[NDIM], dx[NDIM];
extern double B_unit;
extern double max_tau_scatt;
extern double Ne_unit;
extern double Thetae_unit;
extern int N1, N2;
// double L_unit;
extern cudaStream_t max_tau_scatt_stream;

extern __device__ double d_startx[NDIM], d_stopx[NDIM], d_dx[NDIM];
extern __device__ double d_B_unit;
extern __device__ double d_max_tau_scatt;
extern __device__ double d_Ne_unit;
extern __device__ double d_Thetae_unit;
extern __device__ int d_N1, d_N2;

/** model-independent subroutines **/
/* core monte carlo/radiative transport routines */
__global__
void track_super_photon_batch(struct of_photon *phs, unsigned int N);
__host__ __device__
void track_super_photon(struct of_photon *ph);


void record_super_photon(struct of_photon *ph);
void report_spectrum(unsigned long long N_superph_made);
void init_spectrum();
__host__ __device__
void scatter_super_photon(struct of_photon *ph, struct of_photon *php,
			  double Ne, double Thetae, double B,
			  double Ucon[NDIM], double Bcon[NDIM],
			  double Gcov[NDIM][NDIM]);

/* geodesic integration */
__host__ __device__
void init_dKdlam(double X[], double Kcon[], double dK[]);
__host__ __device__
void push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],  double dl,
	double *E0);
__host__ __device__
double stepsize(double X[NDIM], double K[NDIM]);

/* basic coordinate functions supplied by grmonty */
__host__ __device__
void boost(double v[4], double u[4], double vp[4]);
__host__ __device__
void lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov);
double gdet_func(double gcov[][NDIM]);  /* calculated numerically */
__host__ __device__
void coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM],
			  double K_tetrad[NDIM]);
__host__ __device__
void tetrad_to_coordinate(double Ecov[NDIM][NDIM], double K_tetrad[NDIM],
			  double K[NDIM]);
__host__ __device__
double delta(int i, int j);
__host__ __device__
void make_tetrad(double Ucon[NDIM], double Bhatcon[NDIM],
		 double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
		 double Ecov[NDIM][NDIM]);

/* functions related to basic radiation functions & physics */
	/* physics-independent */
__host__ __device__
double get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);
__host__ __device__
double get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
		    double Bcov[NDIM], double B);
__host__ __device__
double alpha_inv_scatt(double nu, double Thetae, double Ne);
__host__ __device__
double alpha_inv_abs(double nu, double Thetae, double Ne, double B,
		     double theta);
__host__ __device__
double Bnu_inv(double nu, double thetae);
__host__ __device__
double jnu_inv(double nu, double thetae, double ne, double B,
	       double theta);

	/* thermal synchrotron */
__host__ __device__
double jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta);
double int_jnu(double Ne, double Thetae, double Bmag, double nu);
void init_emiss_tables(void);
double F_eval(double Thetae, double Bmag, double nu);
__host__ __device__
double K2_eval(double Thetae);

	/* compton scattering */
void init_hotcross(void);
__host__ __device__
double total_compton_cross_lkup(double nu, double theta);
__host__ __device__
double klein_nishina(double a, double ap);
__host__ __device__
double kappa_es(double nu, double theta);
__host__ __device__
void sample_electron_distr_p(double k[4], double l_p[4], double Thetae);
__host__ __device__
void sample_beta_distr(double theta, double *gamma_e, double *beta_e);
__host__ __device__
double sample_klein_nishina(double k0);
__host__ __device__
double sample_thomson();
__host__ __device__
double sample_mu_distr(double beta_e);
__host__ __device__
double sample_y_distr(double theta);
__host__ __device__
void sample_scattered_photon(double k[4],
							 double l_p[4], double kp[4]);

/** model dependent functions required by code: these
   basic interfaces define the model **/

/* physics related */
void init_model(char *args[]);
void init_zone(int i, int j, unsigned long long*nz, double *dnmax, unsigned long long Ns);
__host__ __device__
double bias_func(double Te, double w);
__host__ __device__
void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM]);
__host__ __device__
int stop_criterion(struct of_photon *ph);
int record_criterion(struct of_photon *ph);

/* coordinate related */
__host__ __device__
void get_connection(double X[4], double lconn[4][4][4]);
__host__ __device__
void gcov_func(double *X, double gcov[][NDIM]);
__host__ __device__
void gcon_func(double *X, double gcon[][NDIM]);

#endif
