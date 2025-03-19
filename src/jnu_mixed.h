/*
Declaration of the functions in the jnu_mixed.cu file
*/

#ifndef JNU_MIXED_H
#define JNU_MIXED_H
__host__ __device__ double jnu_synch(double nu, double Ne, double Thetae, double B,double theta);
__host__ double int_jnu(double Ne, double Thetae, double Bmag, double nu);
double jnu_integrand(double th, void *params);
__host__ void init_emiss_tables(void);
__host__ __device__ double K2_eval(double Thetae);
__host__ __device__ double F_eval(double Thetae, double Bmag, double nu);
__host__ __device__ double linear_interp_F(double K);
__host__ __device__ double linear_interp_K2(double Thetae);
#endif
