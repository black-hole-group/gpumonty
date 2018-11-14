#include "gpu_rng.h"
#include "gpu_utils.h"
// #include <math.h>


__global__
void gpu_rng_init (curandState_t *curandstates, long int seed) {
    int id = gpu_thread_id();
    curand_init (seed, id, 0, &curandstates[id]);
}


/*******************************************************************************
* Device-only Functions
*
*******************************************************************************/


/* Taken from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/sphere.c
   Line, 65-91
*/
__device__
void gpu_rng_ran_dir_3d(curandState_t *curandst, double *x, double *y, double *z) {
  double s, a;

  /* This is a variant of the algorithm for computing a random point
   * on the unit sphere; the algorithm is suggested in Knuth, v2,
   * 3rd ed, p136; and attributed to Robert E Knop, CACM, 13 (1970),
   * 326.
   */

  /* Begin with the polar method for getting x,y inside a unit circle
   */
  do
    {
      *x = -1 + 2 * curand_uniform_double(curandst);
      *y = -1 + 2 * curand_uniform_double(curandst);
      s = (*x) * (*x) + (*y) * (*y);
    }
  while (s > 1.0);

  *z = -1 + 2 * s;              /* z uniformly distributed from -1 to 1 */
  a = 2 * sqrt (1 - s);         /* factor to adjust x,y so that x^2+y^2
                                 * is equal to 1-z^2 */
  *x *= a;
  *y *= a;
}


/* The Gamma distribution of order a>0 is defined by:
   p(x) dx = {1 / \Gamma(a) b^a } x^{a-1} e^{-x/b} dx
   for x>0.  If X and Y are independent gamma-distributed random
   variables of order a1 and a2 with the same scale parameter b, then
   X+Y has gamma distribution of order a1+a2.
   The algorithms below are from Knuth, vol 2, 2nd ed, p. 129.
   Code adapted from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/gamma.c
*/
__device__
double gpu_rng_ran_gamma(curandState_t *curandst, double a, const double b){
    /* assume a > 0 */
    double pow_multiplier = 1.0;
    while (a < 1) {
        pow_multiplier *= pow (curand_uniform_double(curandst), 1.0 / a);
        a += 1.0;
    }
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1){
        do{
            x = curand_normal_double(curandst);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(curandst);

        if (u < 1 - 0.0331 * x * x * x * x)
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    return b * d * v * pow_multiplier;
}

/* The chisq distribution has the form
   p(x) dx = (1/(2*Gamma(nu/2))) (x/2)^(nu/2 - 1) exp(-x/2) dx
   for x = 0 ... +infty
   Code taken from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/chisq.c
*/
__device__
double gpu_rng_ran_chisq(curandState_t *curandst, const double nu) {
  double chisq = 2 * gpu_rng_ran_gamma(curandst, nu / 2, 1.0);
  return chisq;
}
