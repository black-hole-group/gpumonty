/*
Declaration of the functions in the jnu_mixed.cu file
*/

#ifndef JNU_MIXED_H
#define JNU_MIXED_H
__host__ __device__ double jnu_synch(const double nu, const double Ne, const double Thetae, const double B,
    const double theta
   #ifdef __CUDA_ARCH__
   , cudaTextureObject_t besselTexObj
   #endif
   );
__host__ double int_jnu(double Ne, double Thetae, double Bmag, double nu);
double jnu_integrand(double th, void *params);
__host__ void init_emiss_tables(void);
__host__ __device__ double K2_eval(const double Thetae
    #ifdef __CUDA_ARCH__
        , cudaTextureObject_t besselTexObj
    #endif
        );
__host__ __device__ double F_eval(const double Thetae, const double Bmag, const double nu);
__host__ __device__ double linear_interp_F(const double K);
//__host__ __device__ double linear_interp_K2(const double Thetae);
__host__ __device__ double linear_interp_K2(const double Thetae
    #ifdef __CUDA_ARCH__
        , cudaTextureObject_t besselTexObj
    #endif
        );
#endif
