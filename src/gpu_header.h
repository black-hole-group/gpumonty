#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <omp.h>
#include "constants.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>
#include "config.h"

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

// extern __device__ int d_N1, d_N2, d_N3;
// extern __device__ double d_a;

#define N_BLOCKS 2
#define N_THREADS 64
/*Testing functions*/
__global__ void GPU_mainloop(struct of_photon ph, time_t time, struct of_geom *d_geom, double *d_p, double * d_table_ptr, struct local_track_var * local_track_vars, int * super_photon_made, struct of_spectrum* d_spect);

__global__ void GPU_generate_photons(struct of_photon *ph_init, struct of_geom * d_geom, double * d_p, time_t time);
__global__ void GPU_track(struct of_photon *ph_init);
__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * n2gen);
__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom *d_geom, double *d_p, int * zi, int d_Ns_par, int * zone_flag);
__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom *d_geom, double *d_p, int zone_flag);
__device__ void GPU_init_monty_rand(int seed);
__device__ double GPU_monty_rand();
__device__ void GPU_coord_hamr(int i, int j, int z, int loc, double *X);
__device__ void GPU_get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
                                   double Ucon[NDIM], double Bcon[NDIM], struct of_geom *d_geom, double *d_p);
__device__ static double GPU_linear_interp_weight(double nu);
__device__ void GPU_coord(int i, int j, double *X);
__device__ double GPU_F_eval(double Thetae, double Bmag, double nu);
__device__ double GPU_jnu_synch(double nu, double Ne, double Thetae, double B,
                                double theta);
__device__ void GPU_make_tetrad(double Ucon[NDIM], double trial[NDIM],
                                double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
                                double Ecov[NDIM][NDIM]);
__device__ double GPU_delta(int i, int j);
__device__ void GPU_tetrad_to_coordinate(double Econ[NDIM][NDIM], double K_tetrad[NDIM],
                                         double K[NDIM]);
__device__ void GPU_lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov);
__device__ double GPU_linear_interp_F(double K);
__device__ double GPU_K2_eval(double Thetae);
__device__ void GPU_project_out(double *vcona, double *vconb, double Gcov[NDIM][NDIM]);
__device__ void GPU_normalize(double *vcon, double Gcov[NDIM][NDIM]);
//__device__ static void GPU_init_zone(int i, int j, int k, double *nz, double *dnmax, struct of_geom *d_geom, double *d_p, int d_Ns_par);
__device__ static void GPU_init_zone(int i, int j, int k, int * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par);

/*track super photon and its dependencies*/
__device__ void GPU_track_super_photon(struct of_photon *ph, double *d_p, struct local_track_var * local_track_vars, int recursive_index, double * d_table_ptr, struct of_spectrum* d_spect);
__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
                                     double *Thetae, double *B, double Ucon[NDIM],
                                     double Ucov[NDIM], double Bcon[NDIM],
                                     double Bcov[NDIM], double *d_p);
__device__ double GPU_get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
                                   double Bcov[NDIM], double B);

__device__ void GPU_gcov_func_hamr(double *X, double gcovp[][NDIM]);
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);
__device__ void GPU_vofx_matthewcoords(double *X, double *V);
__device__ void GPU_bl_coord_hamr(double *X, double *r, double *th, double *phi);
__device__ int GPU_LU_decompose(double A[][NDIM], int permute[]);
__device__ void GPU_LU_substitution(double A[][NDIM], double B[], int permute[]);
__device__ int GPU_invert_matrix(double Am[][NDIM], double Aminv[][NDIM]);
__device__ void GPU_gcon_func_hamr(double gcov[][NDIM], double gcon[][NDIM]);
__device__ double GPU_get_fluid_nu(double X[4], double K[4], double Ucov[NDIM]);
__device__ double GPU_alpha_inv_scatt(double nu, double Thetae, double Ne, double * d_table_ptr);
__device__ double GPU_alpha_inv_abs(double nu, double Thetae, double Ne, double B,
                                    double theta);
__device__ double GPU_bias_func(double Te, double w);
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[]);
__device__ int GPU_stop_criterion(struct of_photon *ph);
__device__ double GPU_stepsize(double X[NDIM], double K[NDIM]);
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
                                double dl, double *E0, int n);
__device__ void GPU_scatter_super_photon(struct of_photon *ph, struct of_photon *php,
                                         double Ne, double Thetae, double B,
                                         double Ucon[NDIM], double Bcon[NDIM],
                                         double Gcov[NDIM][NDIM]);
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
__device__ double GPU_sample_thomson();
__device__ double GPU_sample_klein_nishina(double k0);
__device__ double GPU_klein_nishina(double a, double ap);
__device__ void generate_random_direction(double * x, double *y, double *z);
__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]);
__device__ void GPU_get_connection(double X[4], double lconn[4][4][4]);
__device__ int GPU_record_criterion(struct of_photon *ph);
__device__ void GPU_record_super_photon(struct of_photon *ph, struct of_spectrum* d_spect);
__device__ void omp_reduce_spect_kernel(struct of_spectrum *spect, struct of_spectrum *shared_spect);
__device__ double atomicMax_double(double* address, double val);
__device__ double GPU_kappa_es(double nu, double Thetae, double * d_table_ptr);
__device__ double GPU_total_compton_cross_num(double w, double thetae);
__device__ double GPU_dNdgammae(double thetae, double gammae);
__device__ double GPU_boostcross(double w, double mue, double gammae);
__device__ double GPU_hc_klein_nishina(double we);
__device__ double bessi0(double xbess);
__device__ double bessi1(double xbess);
__device__ double bessk0(double xbess);
__device__ double bessk1(double xbess);
__device__ double bessk2(double xbess);
__device__ double GPU_jnu_inv(double nu, double Thetae, double Ne, double B, double theta);
__device__ double GPU_Bnu_inv(double nu, double Thetae);
__device__ double GPU_total_compton_cross_lkup(double w, double thetae, double * d_table_ptr);
/*GPU  variables*/
/*These variables should be passed only to initialize GPU, then they should become function parameters*/
__device__ void GPU_gcon_func(double *X, double gcon[][NDIM]);
__device__ void GPU_gcov_func(double *X, double gcov[][NDIM]);
__device__ void GPU_bl_coord(double *X, double *r, double *th);