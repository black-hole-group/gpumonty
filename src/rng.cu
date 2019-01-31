#include "rng.h"
#include "config.h"
#include "gpu_utils.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <curand_kernel.h>
#include <omp.h>

static gsl_rng **gsl_rngs;
__device__ static curandState_t *d_curandstates;

__global__
static void gpu_rng_init(curandState_t *d_tmp, unsigned long seed) {
    int id = gpu_thread_id();
    curand_init (seed, id, 0, &d_tmp[id]);
    if (id == 0) d_curandstates = d_tmp;
}

void rng_init(unsigned long seed) {
    gsl_rngs = (gsl_rng **) malloc(N_CPU_THS * sizeof(gsl_rng *));
    for (int i = 0; i < N_CPU_THS; ++i) {
        gsl_rngs[i] = gsl_rng_alloc(gsl_rng_mt19937); /* use Mersenne twister */
        gsl_rng_set(gsl_rngs[i], seed);
    }

    curandState_t *d_tmp;
	CUDASAFE(cudaMalloc(&d_tmp, N_GPU_THS * sizeof(curandState_t)));
    gpu_rng_init<<<BLOCK_SIZE, NUM_BLOCKS>>>(d_tmp, seed);
    CUDAERRCHECK();
}

void rng_destroy() {
    for (int i = 0; i < N_CPU_THS; ++i) {
        gsl_rng_free(gsl_rngs[i]);
    }
    free(gsl_rngs);
    // TODO: Currently it's not possible (or too much time-expensive) to free
    // d_curandstates as it is a device variable (?)
}

__device__ __host__
double rng_uniform_double() {
#ifdef __CUDA_ARCH__
    return curand_uniform_double(&d_curandstates[gpu_thread_id()]);
#else
    return gsl_rng_uniform(gsl_rngs[omp_get_thread_num()]);
#endif
}

__device__ __host__
double rng_gaussian_double() {
#ifdef __CUDA_ARCH__
    return curand_normal_double(&d_curandstates[gpu_thread_id()]);
#else
    return gsl_ran_gaussian(gsl_rngs[omp_get_thread_num()], 1.0);
#endif
}

/* Taken from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/sphere.c
   Line, 65-91
*/
__host__ __device__
void rng_ran_dir_3d(double *x, double *y, double *z) {
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
      *x = -1 + 2 * rng_uniform_double();
      *y = -1 + 2 * rng_uniform_double();
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
__host__ __device__
double rng_ran_gamma(double a, const double b){
    /* assume a > 0 */
    double pow_multiplier = 1.0;
    while (a < 1) {
        pow_multiplier *= pow (rng_uniform_double(), 1.0 / a);
        a += 1.0;
    }
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1){
        do{
            x = rng_gaussian_double();
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = rng_uniform_double();

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
__host__ __device__
double rng_ran_chisq(const double nu) {
  double chisq = 2 * rng_ran_gamma(nu / 2, 1.0);
  return chisq;
}
