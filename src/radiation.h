/*
Declaration of the functions in radiation.cu file
*/

#ifndef RADIATION_H
#define RADIATION_H
__device__ double GPU_jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta, cudaTextureObject_t besselTexObj);
__device__ double GPU_Bnu_inv(const double nu, const double Thetae);
__device__ double GPU_kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr);
__device__ double GPU_get_fluid_nu(const double X[NDIM] , const double K[NDIM] , const double Ucov[NDIM]);
__device__ double GPU_alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double * __restrict__ d_table_ptr);
__device__ double GPU_alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B, const double theta, cudaTextureObject_t besselTexObj);
__device__ double GPU_get_bk_angle(const double X[NDIM], const double K[NDIM] , const double Ucov[NDIM] , const double Bcov[NDIM], const double B);
#endif