/*
 Declaration of the functions used in tetrads.cu file
*/

#ifndef TETRADS_H
#define TETRADS_H
__device__ void GPU_make_tetrad(double Ucon[NDIM], double trial[NDIM], const double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM], double Ecov[NDIM][NDIM]);
__device__ void GPU_tetrad_to_coordinate(const double Econ[NDIM][NDIM], const double K_tetrad[NDIM], double K[NDIM]);
__device__ void GPU_coordinate_to_tetrad(const double Ecov[NDIM][NDIM], const double K[NDIM], double K_tetrad[NDIM]);
__device__ double GPU_delta(int i, int j);
__device__ void GPU_normalize(double *vcon, const double Gcov[NDIM][NDIM]);
__device__ void GPU_project_out(double *vcona, double *vconb, const double Gcov[NDIM][NDIM]);
#endif