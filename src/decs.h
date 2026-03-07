/*
 * GPUmonty - decs.h
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */
#include "config.h"


typedef struct params_t {
  int seed;

  double Ns;
  double MBH_par;
  double M_unit;
  char dump[256];
  char spectrum[256];

  // bias
  int scattering;
  double biasTuning;
  int    fitBias;
  double fitBiasNs;
  double targetRatio;

  // electron temperature models
  double tp_over_te;
  double beta_crit;
  double trat_small;
  double trat_large;
  double Thetae_max;

  char loaded;
} Params;

#include "model.h"



/** data structures **/
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


#include <execinfo.h> // For backtrace() and backtrace_symbols_fd()
#include <unistd.h>   // For STDERR_FILENO
/*Cuda error function*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        
        // --- PRINT BACKTRACE ---
        fprintf(stderr, "\n--- Host Call Stack (Backtrace) ---\n");
        void* callstack[128];
        int frames = backtrace(callstack, 128);
        backtrace_symbols_fd(callstack, frames, STDERR_FILENO);
        fprintf(stderr, "-----------------------------------\n\n");
        // -----------------------

        fflush(stderr);
        if (abort)
            exit(code);
    }
}

#define Flag(message) flag(message, __FILE__, __LINE__)

inline void flag(const char *message, const char *file, int line) {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        // 3. Print the passed-in 'file' and 'line' instead of the macros
        fprintf(stderr, "Error in %s at line %d in: %s: ERROR: %s\n", file, line, message, cudaGetErrorString(cudaStatus));
        exit(1);
    }
}

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

struct of_photon {
	double X[NDIM];
	double K[NDIM];
	double dKdlam[NDIM];
	double w, E, X1i, X2i;
	double tau_abs, tau_scatt;
	double E0, E0s;
	int nscatt;
};

struct of_photonSOA {
    double *X0, *X1, *X2, *X3;
    double *K0, *K1, *K2, *K3;
    double *dKdlam0, *dKdlam1, *dKdlam2, *dKdlam3;
    double *w, *E, *X1i, *X2i;
    double *tau_abs, *tau_scatt;
    double *E0, *E0s;
    int *nscatt;

};


struct of_geom {
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

struct of_spectrum {
	double dNdlE;
	double dEdlE;
	double nph;
	double nscatt;
	double X1iav;
	double X2isq;
	double X3fsq;
	double tau_abs;
	double tau_scatt;
	double E0;
};



#ifndef GPUGLOBALS
#define GPUGLOBALS
	extern __device__ double d_table[NW + 1][NT + 1];

	extern __device__ unsigned long long photon_count;
	extern __device__ unsigned long long d_N_superph_recorded;
	extern __device__ int d_Ns;
	extern __device__ double d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1], d_bias_norm, d_stopx[NDIM], d_Rh, d_max_tau_scatt;


	extern __device__ unsigned long long scattering_counter;
	extern __device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];
	extern __device__ unsigned long long tracking_counter;
	extern __device__ double d_nint[NINT + 1];
	extern __device__ double d_dndlnu_max[NINT + 1];
	extern __device__ double d_hslope;
	extern __device__ double d_R0;
	extern __device__ unsigned long long tracking_counter_sampling;
	extern __device__ curandState my_curand_state[N_BLOCKS * N_THREADS]; // Array of curandState structures
	extern __device__ int d_N1, d_N2, d_N3;
	extern __device__ int d_scattering;
	extern __device__ double d_biastuning[MAX_LAYER_SCA];
	extern __device__ double d_trat_small, d_trat_large, d_beta_crit, d_thetae_max, d_tp_over_te;
	extern __device__ double d_MBH, d_L_unit, d_B_unit, d_Ne_unit;
	extern __device__ double d_bhspin;

#endif

#ifndef CPUGLOBALS
#define CPUGLOBALS
	extern double * p;
	extern double hslope;
	extern double nint[NINT + 1];
	extern double dndlnu_max[NINT + 1];
	extern double K2[N_ESAMP + 1];
	extern double bhspin;
	/*Global Variable Section*/
	/* defining declarations for global variables */
	extern struct of_spectrum spect[N_TYPEBINS][N_THBINS][N_EBINS];

	extern struct of_geom *geom;
	extern double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
	extern double table[NW + 1][NT + 1];

	extern unsigned long long N_scatt;
	extern unsigned long long N_superph_recorded;

	/* some coordinate parameters */
	extern double R0, Rin, Rh, Rout;
	extern double hslope;
	extern double startx[NDIM], stopx[NDIM], dx[NDIM];

	//extern double dlE, lE0;
	extern double Thetae_unit;
	extern double max_tau_scatt, Ladv, dMact, bias_norm;

	extern int N1, N2, N3;
	extern Params params;
	extern double L_unit, B_unit, Ne_unit, Rho_unit, U_unit, M_unit, T_unit;

#endif