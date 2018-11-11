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
#include "constants.h"
#include "gpu_rng.h"

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

#define N_ESAMP		200
#define N_EBINS		200
#define N_THBINS	6

struct of_grid {
	struct of_spectrum spec[N_EBINS];
	double th, phi;
	int nlist;
	int *in;
};

/** global variables **/
/** model independent */

// unsigned long long N_scatt;
/* HARM model globals */
struct of_geom **geom;
int N1, N2, N3;
int n_within_horizon;

double max_tau_scatt, Ladv, dMact, bias_norm;

/* some coordinate parameters */
double startx[NDIM], stopx[NDIM], dx[NDIM];
double dlE, lE0;
double gam;
double dMsim;
double M_unit;
double L_unit;
double T_unit;
double RHO_unit;
double U_unit;
double B_unit;
double Ne_unit;
double Thetae_unit;

#pragma acc declare create(startx, stopx, dx, B_unit, L_unit, max_tau_scatt, Ne_unit,\
	Thetae_unit, lE0, dlE, N1, N2, N3, n_within_horizon)

//From hotcross.c
#define NW	220
#define NT	80


/* some useful macros */
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
#define INDEX(i,j,k)	(NPRIM*( (k) + N3*((j) + N2*(i))))
#define MIN(elem1,elem2) ((elem1) < (elem2) ? (elem1) : (elem2))


/** model-independent subroutines **/
/* core monte carlo/radiative transport routines */
void track_super_photon(curandState_t *curandstate, struct of_photon *ph, unsigned long long *N_superph_recorded, struct of_spectrum **spect);
void record_super_photon(struct of_photon *ph, unsigned long long *N_superph_recorded, struct of_spectrum **spect);
void report_spectrum(unsigned long long N_superph_made, unsigned long long N_superph_recorded, struct of_spectrum **spect);
void scatter_super_photon(curandState_t *curandstate, struct of_photon *ph, struct of_photon *php,
			  double Ne, double Thetae, double B,
			  double Ucon[NDIM], double Bcon[NDIM],
			  double Gcov[NDIM][NDIM]);

/* geodesic integration */
void init_dKdlam(double X[], double Kcon[], double dK[]);
void push_photon_ham(double X[NDIM], double Kcon[][NDIM], double dl[]);
void push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], double dl,
	double *E0);
void push_photon4(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
		  double dl);
void push_photon_cart(double X[NDIM], double Kcon[NDIM],
		      double dKcon[NDIM], double dl);
double stepsize(double X[NDIM], double K[NDIM]);
void push_photon_gsl(double X[NDIM], double Kcon[NDIM], double dl);
int geodesic_deriv(double t, const double y[], double dy[], void *params);
void interpolate_geodesic(double Xi[], double X[], double Ki[], double K[],
			  double frac, double del_l);

/* basic coordinate functions supplied by grmonty */
void boost(double k[NDIM], double p[NDIM], double ke[NDIM]);
void lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov);
double gdet_func(double gcov[][NDIM]);  /* calculated numerically */
void coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM],
			  double K_tetrad[NDIM]);
void tetrad_to_coordinate(double Ecov[NDIM][NDIM], double K_tetrad[NDIM],
			  double K[NDIM]);
double delta(int i, int j);
void make_tetrad(double Ucon[NDIM], double Bhatcon[NDIM],
		 double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
		 double Ecov[NDIM][NDIM]);

/* functions related to basic radiation functions & physics */
	/* physics-independent */
double get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);
double get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
		    double Bcov[NDIM], double B);
double alpha_inv_scatt(double nu, double thetae, double Ne);
double alpha_inv_abs(double nu, double thetae, double Ne, double B,
		     double theta);
double Bnu_inv(double nu, double thetae);
double jnu_inv(double nu, double thetae, double ne, double B,
	       double theta);

	/* thermal synchrotron */
double jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta);
double int_jnu(double Ne, double Thetae, double Bmag, double nu);
void init_emiss_tables(void);
double F_eval(double Thetae, double Bmag, double nu);
double K2_eval(double Thetae);

	/* compton scattering */
void init_hotcross(void);
double total_compton_cross_lkup(double nu, double theta);
double klein_nishina(double a, double ap);
double kappa_es(double nu, double theta);
void sample_electron_distr_p(curandState_t *curandstate, double k[NDIM], double p[NDIM], double theta);
void sample_beta_distr(curandState_t *curandstate, double theta, double *gamma_e, double *beta_e);
double sample_klein_nishina(curandState_t *curandstate, double k0);
double sample_thomson(curandState_t *curandstate);
double sample_mu_distr(curandState_t *curandstate, double beta_e);
double sample_y_distr(curandState_t *curandstate, double theta);
void sample_scattered_photon(curandState_t *curandstate, double k[NDIM], double p[NDIM],
			     double kp[NDIM]);

/** model dependent functions required by code: these
   basic interfaces define the model **/

/* physics related */
void init_model(char *args[]);
void init_zone(int i, int j, unsigned long long*nz, double *dnmax, unsigned long long Ns);
double bias_func(double Te, double w);
void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM]);
int stop_criterion(curandState_t *curandstate, struct of_photon *ph);
int record_criterion(struct of_photon *ph);

/* coordinate related */
void get_connection(double *X, double lconn[][NDIM][NDIM]);
void gcov_func(double *X, double gcov[][NDIM]);
void gcon_func(double *X, double gcon[][NDIM]);

// hotcross.c gpu functions
double gpu_dNdgammae(double thetae, double gammae);
double gpu_total_compton_cross_num(double w, double thetae);


/* openacc device routines pragmas */

// GPU-Only Routines
#pragma acc routine(track_super_photon) nohost
#pragma acc routine(get_fluid_params) nohost
#pragma acc routine(Xtoij) nohost
#pragma acc routine(interp_scalar) nohost
#pragma acc routine(get_bk_angle) nohost
#pragma acc routine(get_fluid_nu) nohost
#pragma acc routine(alpha_inv_scatt) nohost
#pragma acc routine(kappa_es) nohost
#pragma acc routine(total_compton_cross_lkup) nohost
#pragma acc routine(alpha_inv_abs) nohost
#pragma acc routine(jnu_inv) nohost
#pragma acc routine(Bnu_inv) nohost
#pragma acc routine(bias_func) nohost
#pragma acc routine(stop_criterion) nohost
#pragma acc routine(stepsize) nohost
#pragma acc routine(push_photon) nohost
#pragma acc routine(scatter_super_photon) nohost
#pragma acc routine(coordinate_to_tetrad) nohost
#pragma acc routine(sample_electron_distr_p) nohost
#pragma acc routine(sample_beta_distr) nohost
#pragma acc routine(sample_y_distr) nohost
#pragma acc routine(sample_mu_distr) nohost
#pragma acc routine(sample_scattered_photon) nohost
#pragma acc routine(boost) nohost
#pragma acc routine(sample_klein_nishina) nohost
#pragma acc routine(klein_nishina) nohost
#pragma acc routine(sample_thomson) nohost
#pragma acc routine(record_criterion) nohost
#pragma acc routine(record_super_photon) nohost
// Also: isinf, isnan, and gpu_rng_* and gpu_sf_bessel_Kn and hotcross.c static functions and
// acc_print*

// Routines called in CPU and GPU
#pragma acc routine(get_connection)
#pragma acc routine (boostcross)
#pragma acc routine (normalize)
#pragma acc routine (tetrad_to_coordinate)
#pragma acc routine (delta)
#pragma acc routine (make_tetrad)
#pragma acc routine (project_out)
#pragma acc routine (jnu_synch)
#pragma acc routine (K2_eval)
#pragma acc routine (gcon_func)
#pragma acc routine (gcov_func)
#pragma acc routine (bl_coord)
#pragma acc routine (lower)
#pragma acc routine (hc_klein_nishina)
#pragma acc routine(init_dKdlam)

#endif
