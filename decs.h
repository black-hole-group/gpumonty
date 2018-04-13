
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

#define NDIM	4
#define NPRIM	8

/* Range of initial superphoton frequencies */
#define NUMIN 1.e9
#define NUMAX 1.e20

#define THETAE_MAX	1000.	
#define THETAE_MIN	0.3
#define TP_OVER_TE_JET	(1.)
#define TP_OVER_TE_DISK	(15.)
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

/* define size of random string in photon filenames */
#define SIZE_STR 10

/** data structures **/
struct of_photon {        // defined in grmonty.c, used elsewhere
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

struct of_geom {        // defined in grmonty.c
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

struct of_spectrum {        // defined in harm_model.c
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
	struct of_spectrum spec[N_EBINS];    // see harm_model.c
	double th, phi;
	int nlist;
	int *in;
};

/** global variables **/
/** model independent */
extern gsl_rng *r;

extern double F[N_ESAMP + 1], wgt[N_ESAMP + 1];

extern int Ns;
extern int N_superph_recorded, N_scatt;

/* HARM model globals */
extern struct of_geom **geom;
extern int N1, N2, N3;
extern int n_within_horizon;

/* some coordinate parameters */
extern double a;
extern double R0, Rin, Rh, Rout, Rms;
extern double hslope;
extern double startx[NDIM], stopx[NDIM], dx[NDIM];
extern double dlE, lE0;
extern double gam;
extern double dMsim;
extern double x1br, cpow2, npow2, rbr;

extern double M_unit;
extern double L_unit;
extern double T_unit;
extern double RHO_unit;
extern double U_unit;
extern double B_unit;
extern double Ne_unit;
extern double Thetae_unit;
extern double tpte;

extern double max_tau_scatt, Ladv, dMact, bias_norm;

/* some useful macros */
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)
#define INDEX(i,j,k)	(NPRIM*( (k) + N3*((j) + N2*(i))))

/** model-independent subroutines **/
/* core monte carlo/radiative transport routines */
void track_super_photon(struct of_photon *ph);            // defined in track_super_photon.c
void record_super_photon(struct of_photon *ph);           // defined in harm_model.c
void report_spectrum(int N_superph_made);                 // defined in harm_model.c
void scatter_super_photon(struct of_photon *ph, struct of_photon *php,
			  double Ne, double Thetae, double B,
			  double Ucon[NDIM], double Bcon[NDIM],
			  double Gcov[NDIM][NDIM]);       // defined in scatter_super_photon.c

/* OpenMP specific functions */
void omp_reduce_spect(void);       // defined in harm_model.c

/* MC/RT utilities */
void init_monty_rand(int seed);    // defined in compton.c
double monty_rand(void);           // defined in compton.c

/* geodesic integration */
void init_dKdlam(double X[], double Kcon[], double dK[]);                   // defined in geodesics.c
void push_photon_ham(double X[NDIM], double Kcon[][NDIM], double dl[]);     // NOT DEFINED
void push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
		 double dl, double *E0, int n);                             // defined in geodesics.c
void push_photon4(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
		  double dl);                                               // defined in geodesics.c
void push_photon_cart(double X[NDIM], double Kcon[NDIM],
		      double dKcon[NDIM], double dl);                       // NOT DEFINED
double stepsize(double X[NDIM], double K[NDIM]);                            // defined in harm_model.c
void push_photon_gsl(double X[NDIM], double Kcon[NDIM], double dl);         // NOT DEFINED
int geodesic_deriv(double t, const double y[], double dy[], void *params);  // NOT DEFINED
void interpolate_geodesic(double Xi[], double X[], double Ki[], double K[],
			  double frac, double del_l);                       // NOT DEFINED

/* basic coordinate functions supplied by grmonty */
void boost(double k[NDIM], double p[NDIM], double ke[NDIM]);        // defined in compton.c
void lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov);    // defined in tetrads.c
double gdet_func(double gcov[][NDIM]);  /* calculated numerically */// defined in geometry.c
void coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM],
			  double K_tetrad[NDIM]);                   // defined in tetrads.c
void tetrad_to_coordinate(double Ecov[NDIM][NDIM], double K_tetrad[NDIM],
			  double K[NDIM]);                          // defined in tetrads.c
double delta(int i, int j);                                         // defined in tetrads.c
void normalize(double Ucon[NDIM], double Gcov[NDIM][NDIM]);         // defined in tetrads.c
void normalize_null(double Gcov[NDIM][NDIM], double K[NDIM]);       // defined in tetrads.c
void make_tetrad(double Ucon[NDIM], double Bhatcon[NDIM],
		 double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
		 double Ecov[NDIM][NDIM]);                          // defined in tetrads.c

/* functions related to basic radiation functions & physics */
	/* physics-independent */
double get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);    // defined in radiation.c
double get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
		    double Bcov[NDIM], double B);                    // defined in radiation.c
double alpha_inv_scatt(double nu, double thetae, double Ne);         // defined in radiation.c
double alpha_inv_abs(double nu, double thetae, double Ne, double B,
		     double theta);                                  // defined in radiation.c
double Bnu_inv(double nu, double thetae);                            // defined in radiation.c
double jnu_inv(double nu, double thetae, double ne, double B,
	       double theta);                                        // defined in radiation.c

	/* thermal synchrotron */
double jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta);                                      // defined in jnu_mixed.c
double int_jnu(double Ne, double Thetae, double Bmag, double nu);    // defined in jnu_mixed.c
void init_emiss_tables(void);                                        // defined in jnu_mixed.c
double F_eval(double Thetae, double Bmag, double nu);                // defined in jnu_mixed.c
double K2_eval(double Thetae);                                       // defined in jnu_mixed.c

	/* compton scattering */
void init_hotcross(void);                                    // defined in hotcross.c
double total_compton_cross_lkup(double nu, double theta);    // defined in hotcross.c
double klein_nishina(double a, double ap);                   // defined in compton.c
double kappa_es(double nu, double theta);                    // defined in radiation.c
void sample_electron_distr_p(double k[NDIM], double p[NDIM], double theta);    // defined in compton.c
void sample_beta_distr(double theta, double *gamma_e, double *beta_e);         // defined in compton.c
double sample_klein_nishina(double k0);                      // defined in compton.c
double sample_thomson(void);                                 // defined in compton.c
double sample_mu_distr(double beta_e);                       // defined in compton.c
double sample_y_distr(double theta);                         // defined in compton.c
void sample_scattered_photon(double k[NDIM], double p[NDIM],
			     double kp[NDIM]);               // defined in compton.c

/** model dependent functions required by code: these 
   basic interfaces define the model **/

/* physics related */
void init_model(char *args[]);                                   // defined in harm_model.c
void make_super_photon(struct of_photon *ph, int *quit_flag);    // defined in harm_model.c
double bias_func(double Te, double w);                           // defined in harm_model.c
void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM]);                        // defined in harm_model.c
int stop_criterion(struct of_photon *ph);                        // defined in harm_model.c
int record_criterion(struct of_photon *ph);                      // defined in harm_model.c

/* coordinate related */
void get_connection(double *X, double conn[][NDIM][NDIM]);    // defined in harm_model.c
void gcov_func(double *X, double gcov[][NDIM]);                // defined in harm_model.c
//void gcon_func(double *X, double gcon[][NDIM]);                // defined in harm_model.c
void gcon_func(double gcov[][NDIM], double gcon[][NDIM]);
void get_dxdX(double X[NDIM], double dxdX[NDIM][NDIM]);
/* misc. routines */
char *rand_string(char *str, size_t size);
