#ifndef _RNG_H
#define _RNG_H


void rng_init(unsigned long seed);
void rng_destroy();
__device__ __host__
double rng_uniform_double();
__device__ __host__
double rng_gaussian_double();
__host__ __device__
void rng_ran_dir_3d(double *x, double *y, double *z);
__host__ __device__
double rng_ran_gamma(double a, const double b);
__host__ __device__
double rng_ran_chisq(const double nu);


#endif
