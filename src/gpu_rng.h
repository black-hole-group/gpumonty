#ifndef _GPU_RNG_H
#define _GPU_RNG_H

// #include <cuda.h>
// #include <curand.h>
#include <curand_kernel.h>

__global__
void gpu_rng_init (curandState_t *curandstates, long int seed);
__device__
void gpu_rng_ran_dir_3d(curandState_t *curandst, double *x, double *y, double *z);
__device__
double gpu_rng_ran_gamma(curandState_t *curandst, double a, const double b);
__device__
double gpu_rng_ran_chisq(curandState_t *curandst, const double nu);

#endif
