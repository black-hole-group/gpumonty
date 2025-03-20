/*
Declarations of the kernels.cu file functions
*/

#ifndef KERNELS_H
#define KERNELS_H
__global__ void GPU_generate_photons(struct of_geom * d_geom, double * d_p, time_t time, unsigned long long * generated_photons_arr, double * dnmax_arr);
__global__ void GPU_sample_photons_batch(struct of_photon *ph_init, struct of_geom * d_geom, double * d_p, unsigned long long * generated_photons_arr, double * dnmax_arr, int max_partition_ph, unsigned long long photons_processed_sofar, unsigned long long * index_to_ijk);
__global__ void GPU_track(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, unsigned long long max_partition_ph, int nblocks);
__global__ void GPU_track_scat(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, int n, int number_of_threads);
__global__ void GPU_record(struct of_photon * ph, struct of_spectrum * d_spect, unsigned long long  max_partition_ph, int nblocks);
__global__ void GPU_record_scattering(struct of_photon * ph, struct of_spectrum * d_spect, unsigned long long  max_partition_ph, int nblocks, int n);
__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * n2gen);
__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * zone_flag);
__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p, int zone_flag, unsigned long long ph_arr_index, double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState localState);
__device__  void GPU_init_zone(int i, int j, int k, unsigned long long * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par, curandState localState);
__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum spect[N_THBINS][N_EBINS], const char * filename);
__host__ void mainFlowControl(time_t time, double * p, const char * filename);
#endif


/*Cuda error function*/
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Macro to simplify cudaMemcpy calls with error checking
#define cudaMemcpyErrorCheck(dst, src, count, kind) \
    cudaMemcpyCheck((dst), (src), (count), (kind), __FILE__, __LINE__)

// Function to handle CUDA memory copies and check for errors
inline void cudaMemcpyCheck(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                            const char *file, int line)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in file %s at line %d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}