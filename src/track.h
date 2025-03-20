/*
Declarations of the functions in the track.cu file
*/

#ifndef TRACK_H
#define TRACK_H
__device__ void GPU_track_super_photon(struct of_photon *ph, double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, int round_scat, int photon_index, int instant_partition, curandState localState);
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[]);
__device__ double GPU_stepsize(double X[NDIM], double K[NDIM]);
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], double dl, double *E0, int n);
#endif