/*
Declarations of the functions in the metrics.cu file
*/

#ifndef METRICS_H
#define METRICS_H
double gdet_func(double gcov[][NDIM]);
__host__  __device__ int LU_decompose( double A[][NDIM], int permute[] );
__host__ __device__ void LU_substitution( double A[][NDIM], double B[], int permute[] );
__host__ __device__ int invert_matrix( double Am[][NDIM], double Aminv[][NDIM] )  ;
__host__ __device__ void gcon_func(const double X[4], double gcov[][NDIM], double gcon[][NDIM]);
__device__ void GPU_get_connection(const double X[4], double lconn[4][4][4]);
__host__ __device__ void lower(double *ucon, const double Gcov[NDIM][NDIM], double *ucov);
#endif
