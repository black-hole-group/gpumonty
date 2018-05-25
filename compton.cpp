
#include "decs.h"
#pragma omp threadprivate(r)

/*

Routines for treating Compton scattering via Monte Carlo.

Uses a Gnu Scientific Library (GSL) random number generator.
The choice of generator can be changed in init_monty_rand;
now set to Mersenne twister.

Sampling procedures for electron distribution is based on
Canfield, Howard, and Liang, 1987, ApJ 323, 565.

*/

void init_monty_rand(int seed)
{
	r = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
	gsl_rng_set(r, seed);
}

/* return pseudo-random value between 0 and 1 */
double monty_rand()
{
	return (gsl_rng_uniform(r));
}



//void sample_scattered_photon(double k[4], double p[4], double kp[4])

//void boost(double v[4], double u[4], double vp[4])

//double sample_thomson()

//double sample_klein_nishina(double k0)

//double klein_nishina(double a, double ap)

//void sample_electron_distr_p(double k[4], double p[4], double Thetae)

//void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e)

//double sample_y_distr(double Thetae)

//double sample_mu_distr(double beta_e)
