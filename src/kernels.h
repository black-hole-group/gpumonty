/*
Declarations of the kernels.cu file functions
*/

#ifndef KERNELS_H
#define KERNELS_H
__global__ void GPU_generate_photons(const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const time_t time, unsigned long long * __restrict__  generated_photons_arr, double * __restrict__ dnmax_arr, cudaTextureObject_t besselTexObj);    
__device__ void GPU_init_zone(const int i, const int j, const int k, unsigned long long * __restrict__  n2gen, double * __restrict__ dnmax, const struct of_geom * __restrict__  d_geom, const double * __restrict__ d_p, const int d_Ns_par, curandState  * localState, cudaTextureObject_t besselTexObj);
__global__ void GPU_sample_photons_batch(struct of_photonSOA ph_init, const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const unsigned long long * __restrict__  generated_photons_arr, const double * __restrict__ dnmax_arr, const int max_partition_ph,  const unsigned long long photons_processed_sofar, const unsigned long long * __restrict__  index_to_ijk, cudaTextureObject_t besselTexObj);
__device__ void GPU_sample_zone_photon(const int i, const int j, const int k, const double dnmax, struct of_photonSOA ph, const struct of_geom * d_geom, const double * d_p, const int zone_flag, const unsigned long long ph_arr_index, double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState * localState, cudaTextureObject_t besselTexObj);
__global__ void GPU_track(struct of_photonSOA ph, 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
        double * __restrict__  d_p,
    #else
        cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long max_partition_ph, const int nblocks, cudaTextureObject_t besselTexObj);  

__global__ void GPU_track_scat(struct of_photonSOA ph, 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
        double * __restrict__ d_p,
    #else
        cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int n, const int number_of_threads, cudaTextureObject_t besselTexObj, unsigned long long round_num_scat_init, unsigned long long round_num_scat_end);

__global__ void GPU_record(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks);
__global__ void GPU_record_scattering(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks, const int n);

__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum spect[N_THBINS][N_EBINS], const char * filename);
__host__ void mainFlowControl(time_t time, double * p);
#endif
