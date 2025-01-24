
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


/*Testing functions*/
#ifndef GPU_FUNCTIONS
#define GPU_FUNCTIONS
__device__ double GPU_monty_rand();
__global__ void GPU_mainloop(struct of_photon ph, time_t time, struct of_geom *d_geom, double *d_p, double * d_table_ptr, int * super_photon_made, struct of_spectrum* d_spect);
__global__ void GPU_generate_photons(struct of_geom * d_geom, double * d_p, time_t time, unsigned long long * generated_photons_arr, double * dnmax_arr);
__global__ void GPU_sample_photons_batch(struct of_photon *ph_init, struct of_geom * d_geom, double * d_p, unsigned long long * generated_photons_arr, double * dnmax_arr, int max_partition_ph, unsigned long long photons_processed_sofar, unsigned long long * index_to_ijk);
__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * n2gen);
__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * zone_flag);
__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p, int zone_flag, unsigned long long ph_arr_index, double (*Econ)[NDIM], double (*Ecov)[NDIM]);

__device__ void GPU_init_monty_rand(int seed);
__host__ __device__ void get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
                                   double Ucon[NDIM], double Bcon[NDIM], struct of_geom *d_geom, double *d_p);
__device__ static double GPU_linear_interp_weight(double nu);
__host__ __device__ double F_eval(double Thetae, double Bmag, double nu);
__host__ __device__ double jnu_synch(double nu, double Ne, double Thetae, double B,
                                double theta);

__device__ void GPU_make_tetrad(double Ucon[NDIM], double trial[NDIM],
                                double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
                                double Ecov[NDIM][NDIM]);
__device__ double GPU_delta(int i, int j);
__device__ void GPU_tetrad_to_coordinate(double Econ[NDIM][NDIM], double K_tetrad[NDIM],
                                         double K[NDIM]);
__host__ __device__ void lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov);
__host__ __device__ double linear_interp_F(double K);
__host__ __device__ double linear_interp_K2(double Thetae);
__host__ __device__ double K2_eval(double Thetae);
__device__ void GPU_project_out(double *vcona, double *vconb, double Gcov[NDIM][NDIM]);
__device__ void GPU_normalize(double *vcon, double Gcov[NDIM][NDIM]);
__device__  void GPU_init_zone(int i, int j, int k, int * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par);

/*track super photon and its dependencies*/
__device__ void GPU_copy_survivor(struct of_scattering * survivor, int bound_flag, double dtau_scatt, double d_tau_abs, double dtau, double bi, double bf, double alpha_scatti, double alpha_scattf, double alpha_absi, double alpha_absf, double dl, double x1, double nu, double Thetae, double Ne, double B, double theta, double dtauK, double frac, double bias, double Xi[], double Ki[], double dKi[], double E0, double Gcov[][NDIM], double Ucon[], double Ucov[], double Bcon[], double Bcov[], int nstep, struct of_photon * ph);
__global__ void GPU_track(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, int max_partition_ph, int instant_partition);
__device__ void GPU_track_super_photon(struct of_photon *ph, struct of_spectrum * d_spect, double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, int round_scat, int photon_index, int instant_partition);

__global__ void GPU_track_scat(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, int n, int number_of_threads);
//__device__ void GPU_track_super_photon(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum* d_spect, struct of_scattering * survivor_photon_properties, struct of_photon * survivor_photon, int * local_recursive_index, int  * is_recursive, struct of_photon * scattered_photon);
__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne, double *Thetae, double *B, double Ucon[NDIM], double Ucov[NDIM], double Bcon[NDIM], double Bcov[NDIM], double *d_p);
__device__ double GPU_get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM], double Bcov[NDIM], double B);
__device__ void GPU_vofx_matthewcoords(double *X, double *V);
__host__ __device__ int LU_decompose(double A[][NDIM], int permute[]);
__host__ __device__ void LU_substitution(double A[][NDIM], double B[], int permute[]);
__host__ __device__ int invert_matrix(double Am[][NDIM], double Aminv[][NDIM]);
__device__ double GPU_get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);
__device__ double GPU_alpha_inv_scatt(double nu, double Thetae, double Ne, double * d_table_ptr);
__device__ double GPU_alpha_inv_abs(double nu, double Thetae, double Ne, double B,
                                    double theta);

__device__ void GPU_init_blackbody_photons(int i, int j, int k, int *n2gen, double *dnmax, 
                                          struct of_geom *d_geom,  
                                          double *d_dx, int d_Ns_par);

__device__ double GPU_bias_func(double Te, double w);
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[]);
__device__ double GPU_stepsize(double X[NDIM], double K[NDIM]);
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
                                double dl, double *E0, int n);
__device__ void GPU_scatter_super_photon(struct of_photon *ph, struct of_photon *php, double Ne, double Thetae, double B, double Ucon[NDIM], double Bcon[NDIM], double Gcov[NDIM][NDIM]);
__device__ void GPU_coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM],
                                         double K_tetrad[NDIM]);
__device__ void GPU_sample_electron_distr_p(double k[4], double p[4], double Thetae);
__device__ double GPU_sample_y_distr(double Thetae);
__device__ void GPU_sample_beta_distr(double Thetae, double *gamma_e, double *beta_e);
__device__ double GPU_sample_mu_distr(double beta_e);
__device__ void GPU_sample_scattered_photon(double k[4], double p[4], double kp[4]);
__device__ double generate_chi_square(int df);
__device__ double chi_square(int df);
__device__ void GPU_boost(double v[4], double u[4], double vp[4]);
__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index);
__device__ double GPU_sample_thomson();
__device__ double GPU_sample_klein_nishina(double k0);
__device__ double GPU_klein_nishina(double a, double ap);
__device__ void generate_random_direction(double * x, double *y, double *z);
__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]);
__device__ void GPU_get_connection(double X[NDIM], double conn[NDIM][NDIM][NDIM]);
__device__ void GPU_record_super_photon(struct of_photon *ph, struct of_spectrum* d_spect);
__device__ void omp_reduce_spect_kernel(struct of_spectrum *spect, struct of_spectrum *shared_spect);
__device__ double GPU_kappa_es(double nu, double Thetae, double * d_table_ptr);
__host__ __device__ double total_compton_cross_num(double w, double thetae);
__host__ __device__ double dNdgammae(double thetae, double gammae);
__host__ __device__ double boostcross(double w, double mue, double gammae);
__host__ __device__ double hc_klein_nishina(double we);
__host__ __device__ double bessi0(double xbess);
__host__ __device__ double bessi1(double xbess);
__host__ __device__ double bessk0(double xbess);
__host__ __device__ double bessk1(double xbess);
__host__ __device__ double bessk2(double xbess);
__host__ double int_jnu(double Ne, double Thetae, double Bmag, double nu);

__device__ double GPU_jnu_inv(double nu, double Thetae, double Ne, double B, double theta);
__device__ double GPU_Bnu_inv(double nu, double Thetae);
__host__ __device__ double total_compton_cross_lkup(double w, double thetae, double * d_table_ptr);

/*GPU  variables*/
/*These variables should be passed only to initialize GPU, then they should become function parameters*/
__host__ __device__ void gcon_func(double X[4], double gcov[][NDIM], double gcon[][NDIM]);
__device__ __forceinline__ double atomicMaxdouble(double *address, double val);


__host__ void init_geometry();
__host__ double gdet_func(double gcov[][NDIM]);
__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum spect[N_THBINS][N_EBINS], const char * filename);
__host__ void init_hotcross(void);
__host__ void init_weight_table(void);
__host__ void init_weight_table_blackbody(void);
__host__ void init_emiss_tables(void);
__host__ void init_nint_table(void);
__host__ void launch_loop(struct of_photon ph, int quit_flag, time_t time, double * p, const char * filename);
__host__ void init_model(char *args[]);
#endif