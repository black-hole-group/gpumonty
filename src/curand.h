/*
Declarations of the functions in curand.cu file
*/

#ifndef CURAND_H
#define CURAND_H
__device__ void GPU_init_monty_rand(int seed);
__device__ void generate_random_direction(double *x, double *y, double *z, curandState * localState);
__device__ double chi_square(int df, curandState * localState);
__device__ double legacy_standard_exponential(curandState * localState);
__device__ void legacy_gauss(double* out1, double* out2, curandState * localState);
__device__ double legacy_standard_gamma(double shape, curandState * localState);
#endif