/*
Declaration of the functions in the hotcross.cu file
*/


#ifndef HOTCROSS_H
#define HOTCROSS_H
__host__ void init_hotcross(void);
__device__ float total_compton_cross_lkup(double w, double thetae, const double * __restrict__ d_table_ptr);
__host__ __device__ double total_compton_cross_num(double w, double thetae);
__host__ __device__ double dNdgammae(double thetae, double gammae);
__host__ __device__ double boostcross(double w, double mue, double gammae);
__host__ __device__ double hc_klein_nishina(double we);
__host__ __device__ double bessi0(double xbess);
__host__ __device__ double bessi1(double xbess);
__host__ __device__ double bessk0(double xbess);
__host__ __device__ double bessk1(double xbess);
__host__ __device__ double bessk2(double xbess);
#endif
