


/*Testing functions*/
#ifndef GPU_FUNCTIONS
#define GPU_FUNCTIONS

__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * n2gen);
__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * zone_flag);
__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p, int zone_flag, unsigned long long ph_arr_index, double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState localState);

__device__  void GPU_init_zone(int i, int j, int k, unsigned long long * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par, curandState localState);

/*track super photon and its dependencies*/

__device__ void GPU_init_blackbody_photons(int i, int j, int k, unsigned long long *n2gen, double *dnmax, 
                                          struct of_geom *d_geom,  
                                          double *d_dx, int d_Ns_par);

__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum spect[N_THBINS][N_EBINS], const char * filename);
__host__ void mainFlowControl(time_t time, double * p, const char * filename);
#endif