/*
Declaration of the functions in compton.cu file
*/

#ifndef COMPTON_H
#define COMPTON_H
__device__ void GPU_sample_scattered_photon(double k[4], double p[4], double kp[4], curandState localState);
__device__ void GPU_boost(double v[4], double u[4], double vp[4]);
__device__ double GPU_sample_thomson(curandState localState);
__device__ double GPU_sample_klein_nishina(double k0, curandState localState);
__device__ double GPU_klein_nishina(double a, double ap);
__device__ void GPU_sample_electron_distr_p(double k[4], double p[4], double Thetae, curandState localState);
__device__ void GPU_sample_beta_distr(double Thetae, double *gamma_e, double *beta_e, curandState localState);
__device__ double GPU_sample_y_distr(double Thetae, curandState localState);
__device__ double GPU_sample_mu_distr(double beta_e, curandState localState);
__device__ void GPU_scatter_super_photon(struct of_photonSOA ph, struct of_photonSOA php,double Ne, double Thetae, double B, double Ucon[NDIM], double Bcon[NDIM], double Gcov[NDIM][NDIM], curandState localState, unsigned long long photon_index);
#endif