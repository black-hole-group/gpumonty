#ifndef _GPU_RNG_H
#define _GPU_RNG_H

// #include <cuda.h>
// #include <curand.h>
#include <openacc_curand.h>

#pragma acc routine(gpu_rng_init) nohost
#pragma acc routine(gpu_rng_uniform) nohost
#pragma acc routine(gpu_rng_ran_dir_3d) nohost
#pragma acc routine(gpu_rng_ran_gamma) nohost
#pragma acc routine(gpu_rng_ran_chisq) nohost

void gpu_rng_init (curandState_t *curandst, long int seed);
double gpu_rng_uniform (curandState_t *curandst);
void gpu_rng_ran_dir_3d(curandState_t *curandst, double *x, double *y, double *z);
double gpu_rng_ran_gamma(curandState_t *curandst, const double a, const double b);
double gpu_rng_ran_chisq(curandState_t *curandst, const double nu);

#endif
