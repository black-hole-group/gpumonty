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
#include "config.h"

/*Cuda error function*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Macro to simplify cudaMemcpy calls with error checking
#define cudaMemcpyErrorCheck(dst, src, count, kind) \
    cudaMemcpyCheck((dst), (src), (count), (kind), __FILE__, __LINE__)

// Function to handle CUDA memory copies and check for errors
inline void cudaMemcpyCheck(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                            const char *file, int line) {
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in file %s at line %d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// extern __device__ int d_N1, d_N2, d_N3;
// extern __device__ double d_a;



/*Testing functions*/
__global__ void GPU_mainloop(struct of_photon ph, time_t time, struct of_geom * d_geom, double * d_p);
__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom * d_geom, double * d_p);
__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom * d_geom, double * d_p);
__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p);
__device__ void GPU_init_monty_rand(int seed);
__device__ double GPU_monty_rand();
__device__ void GPU_coord_hamr(int i, int j, int z, int loc, double * X);
__device__ void GPU_get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM], struct of_geom * d_geom, double * d_p);
__device__ static double GPU_linear_interp_weight(double nu);
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
__device__ static void GPU_init_zone(int i, int j, int k, double *nz, double *dnmax, struct of_geom * d_geom, double * d_p);


/*GPU  variables*/
/*These variables should be passed only to initialize GPU, then they should become function parameters*/



