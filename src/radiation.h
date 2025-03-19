/*
Declaration of the functions in radiation.cu file
*/

#ifndef RADIATION_H
#define RADIATION_H
__device__ double GPU_jnu_inv(double nu, double Thetae, double Ne, double B, double theta);
__device__ double GPU_Bnu_inv(double nu, double Thetae);
__device__ double GPU_kappa_es(double nu, double Thetae, double * d_table_ptr);
__device__ double GPU_get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);
__device__ double GPU_alpha_inv_scatt(double nu, double Thetae, double Ne, double * d_table_ptr);
__device__ double GPU_alpha_inv_abs(double nu, double Thetae, double Ne, double B,
                                    double theta);
__device__ double GPU_get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM], double Bcov[NDIM], double B);
#endif