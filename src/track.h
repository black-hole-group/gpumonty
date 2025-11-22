/*
Declarations of the functions in the track.cu file
*/

#ifndef TRACK_H
#define TRACK_H
__device__ void GPU_track_super_photon(struct of_photonSOA ph , 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
    	double * __restrict__ d_p,
    #else
    	cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long starting_scattering_index, const int round_scat, const unsigned long long photon_index, curandState * localState, cudaTextureObject_t besselTexObj);
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[]);
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], const double dl, double *E0);
#endif