#include <gsl/gsl_rng.h>

static gsl_rng *rng;

/*

Routines for random number generation at CPU..

Uses a Gnu Scientific Library (GSL) random number generator.
The choice of generator can be changed in cpu_rng_init;
now set to Mersenne twister.
*/

void cpu_rng_init(unsigned long int seed)
{
	rng = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
	gsl_rng_set(rng, seed);
}

/* return pseudo-random value between 0 and 1 */
double cpu_rng_uniforme()
{
	return (gsl_rng_uniform(rng));
}
