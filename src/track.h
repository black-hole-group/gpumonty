/*
Declarations of the functions in the track.cu file
*/

#ifndef TRACK_H
#define TRACK_H
__device__ void GPU_track_super_photon(struct of_photonSOA ph , cudaTextureObject_t d_p, const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int round_scat, const unsigned long long photon_index, curandState * localState, cudaTextureObject_t besselTexObj);
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[]);
__device__ double GPU_stepsize(const double X[NDIM], const double K[NDIM]);
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], const double dl, double *E0);
#endif