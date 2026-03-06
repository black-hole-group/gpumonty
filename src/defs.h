/*
 * GPUmonty - defs.h
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
#include "model.h"

/**
 * @brief Global host data that holds the metric components for the simulation grid \f$ g_{μν}, g^{μν},\ \rm{and}\ \sqrt{- g}  \f$ .
 */
struct of_geom *geom;

/** 
 * @brief Global host reusable workspace for tracking sub-intervals and error estimates 
 * during adaptive numerical integration used by GSL. 
 */
gsl_integration_workspace *w;


double F[N_ESAMP + 1], /**< Global host data array of precomputed emissivity values binned by frequency. */
       wgt[N_ESAMP + 1]; /**< Global host data array containing precomputed superphotons weights binned by frequency. */


/**
* Global host data counter to keep track of the number of scattered superphotons per scattering layer.
*/
unsigned long long N_scatt;

/** 
 * @brief Global host data that holds the number of superphotons recorded into the spectrum.
 */
unsigned long long N_superph_recorded;

/* some coordinate parameters */

/**
 * Global host variable representing the physical inner boundary of the simulation grid, e.g, the starting point of the 
 * physical domain.
 */
double Rin;

/**
 * Global host variable representing the physical outer boundary of the simulation grid. It defines the edge of the grid for the simulation.
 */
double Rout;

/**
 * Global host variable representing the event horizon of the black hole in Kerr-Schild coordinates
 */
double Rh;

/**
 * Global host variable representing a shift in the logarithmic radial coordinate.
 */
double R0;

/**
 * Global host variable for the latitude slope for Modified Kerr-Schild (MKS) coordinates
 */
double hslope;

/**
 * Global host array defining the start coordinate of the simulation grid in each dimension in native coordinates.
 */
double startx[NDIM];

/**
 * Global host array defining the stop coordinate of the simulation grid in each dimension in native coordinates.
 */
double stopx[NDIM];

/**
 * Global host array defining the grid spacing in each dimension in native coordinates.
 */
double dx[NDIM];


/**
 * Global host variable for the electron temperature unit to translate from cgs to code units.
 */
double Thetae_unit;

/**
 * Global host variable for the maximum optical depth found so far, used generally to calculate the biasing factor.
 */
double max_tau_scatt;

/**
 * Global host variable calculating the total amount of mass falling into the black hole per unit time.
 */
double dMact;

/**
 * Global host variable for the advection luminosity in CGS units.
 */
double Ladv;

/**
 * Global host variable for the bias normalization factor.
 */
double bias_norm;

/**
 * Global host pointer for accessing plasma properties.
 */
double * p;

/**
 * Global host variable for the number of grid cells in the radial direction.
 */
int N1;
/**
 * Global host variable for the number of grid cells in the poloidal direction.
 */
int N2;

/**
 * Global host variable for the number of grid cells in the azimuthal direction.
 */
int N3;

/**
*  Global host variable for the length unit conversion factor. Scales code units to centimeters (cm). 
*/
double L_unit;
/**
 * Global host variable for the magnetic field unit conversion factor. Scales code units to Gauss (G). 
 */
double B_unit;

/**
 * Global host variable for the electron number density unit conversion factor. Scales code units to \f$ \rm{cm}^-3\f$. 
 */
double Ne_unit;

/** 
 * Global host variable for the mass density unit conversion factor. Scales code units to \f$\rm{g/cm}^3\f$. 
 */
double Rho_unit;

/** 
 * Global host variable for the internal energy density unit conversion factor. Scales code units to \f$\rm{erg/cm}^3\f$. 
 */
double U_unit;

/** 
 * Global host variable for the mass unit conversion factor. Scales code units to grams (g). 
 */
double M_unit;

/**  
 * Global host variable for the time unit conversion factor. Scales code units to seconds (s). 
 */
double T_unit;

/**
 * Global host array for the precomputed hot cross section lookup table.
 * It stores values indexed by photon energy (\f$\rm{N}_{\omega}\f$) and electron temperature (\f$\rm{N}_\rm{T}\f$).
 */
double table[NW + 1][NT + 1];

/** 
 * Global host variable for the logarithmic step size in photon frequency (\f$\Delta \ln \omega\f$). 
 * It determines the resolution of the frequency axis in hotcross section precomputed lookup table.
 */
double dlw;

/** 
 * Global host variable for the logarithmic step size in electron temperature (\f$\Delta \ln \Theta_\rm{e}\f$). 
 * It determines the resolution of the temperature axis in hotcross section precomputed lookup table.
 */
double dlT;

/** 
 * Global host variable for the minimum logarithmic photon frequency (\f$\ln \omega_{min}\f$). 
 * Defines the lower boundary of the frequency grid for cross-section calculations.
 */
double lminw;

/** 
 * Global host variable for the minimum logarithmic electron temperature (\f$\ln \Theta_{e,min}\f$). 
 * Defines the lower boundary of the temperature grid for cross-section calculations.
 */
double lmint;

/** 
 * Global host array for the superphoton density distribution function. 
*/
double nint[NINT + 1];


/**
 * Global host array for the precomputed values of the modified Bessel function of the second kind (\f$K_2\f$). 
 */
double K2[N_ESAMP + 1];

/** 
 * Global host array for the maximum values of the differential photon number distribution (\f$\frac{dN}{d\ln \nu}\f$). 
 * Used for normalization and sampling in the scattering process.
 */
double dndlnu_max[NINT + 1];


/**
 * Global device array of curandState structures for random number generation on the GPU. It's sized to accommodate all threads across all blocks.
 */
__device__ curandState my_curand_state[N_BLOCKS * N_THREADS]; // Array of curandState structures

/**
 * Global device array for the precomputed hot cross section lookup table.
 * It stores values indexed by photon energy (\f$\rm{N}_{\omega}\f$) and electron temperature (\f$\rm{N}_\rm{T}\f$).
 * It's the same as the host array table but accessible on the GPU.
 */
__device__ double d_table[NW + 1][NT + 1];

/**
 * Global device variable counter to keep track of the total number of superphotons generated from the grid during the simulation.
 */
__device__ unsigned long long photon_count = 0;


/**
 * Global device variable to keep track of the number of superphotons recorded into the spectrum.
 */
__device__ unsigned long long  d_N_superph_recorded;

/**
 * Global device variable for the target input number of superphotons to be generated in the simulation.
 * It's the same as the host variable Ns but accessible on the GPU.
 */
__device__ int d_Ns;

/**
 * Global device variable counter to keep track of the total number of scatterings that have occurred during the simulation.
 */
__device__ unsigned long long d_N_scatt;


/**
 * Global device variable for the electron temperature unit to translate from cgs to code units.
 * It's the same as the host variable Thetae_unit but accessible on the GPU.
 */
__device__ double d_thetae_unit;

/**
 * Global device array defining the start coordinate of the simulation grid in each dimension in native coordinates.
 * It's the same as the host array startx but accessible on the GPU.
 */
__device__ double d_startx[NDIM];

/**
 * Global device array defining the grid spacing in each dimension in native coordinates.
 * It's the same as the host array dx but accessible on the GPU.
 */
__device__ double d_dx[NDIM];

/**
 * Global device array containing precomputed superphotons weights binned by frequency.
 * It's the same as the host array wgt but accessible on the GPU.
 */
__device__ double d_wgt[N_ESAMP + 1];

/**
 * Global device array of precomputed emissivity values binned by frequency.
 * It's the same as the host array F but accessible on the GPU.
 */
__device__ double d_F[N_ESAMP + 1];

/**
 * Global device array for the precomputed modified Bessel function values binned by frequency.
 * It's the same as the host array K2 but accessible on the GPU.
 */
__device__ double d_K2[N_ESAMP + 1];

/**
 * Global device variable for the bias normalization factor.
 * It's the same as the host variable bias_norm but accessible on the GPU.
 */
__device__ double d_bias_norm;

/**
 * Global device array defining the stop coordinate of the simulation grid in each dimension in native coordinates.
 * It's the same as the host array stopx but accessible on the GPU.
 */
__device__ double d_stopx[NDIM];

/**
 * Global device variable representing the event horizon of the black hole in Kerr-Schild coordinates.
 * It's the same as the host variable Rh but accessible on the GPU.
 */
__device__ double d_Rh;

/**
 * Global device variable for the maximum optical depth found so far, used generally to calculate the biasing factor.
 * It's the same as the host variable max_tau_scatt but accessible on the GPU.
 */
__device__ double d_max_tau_scatt;

	
/**
 * Global device variable counter to keep track of the number of scattered superphotons in the current scattering layer.
 * @note This is different than d_N_scatt which counts the total number of scattering events in the simulation.
 */
__device__ unsigned long long scattering_counter = 0;

/**
 * Global device array to keep track of the number of scattered superphotons per scattering layer.
 * @note This is scattering_counter because scattered photons are stored layer by layer.
 */
__device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];

/**
 * Global device variable counter to keep track of the number of photons being tracked in the current scattering batch.
 * Used to dynamically allocate threads for photon tracking.
 */
__device__ unsigned long long tracking_counter = 0;

/**
 * Global device array for the superphoton density distribution function.
 * It mirrors the host array nint but is accessible on the GPU.
 */
__device__ double d_nint[NINT + 1];


/**
 * Global device array for the maximum values of the differential photon number distribution (\f$\frac{dN}{d\ln \nu}\f$). 
 * It mirrors the host array dndlnu_max but is accessible on the GPU.
 */
__device__ double d_dndlnu_max[NINT + 1];

/**
 * Global device variable for the latitude slope for Modified Kerr-Schild (MKS) coordinates
 * It mirrors the host variable hslope but is accessible on the GPU.
 */
__device__ double d_hslope = 0;

/**
 * Global device variable representing a shift in the logarithmic radial coordinate.
 * It mirrors the host variable Rin but is accessible on the GPU.
 */
__device__ double d_R0 = 0;

/**
 * Global device variable counter to keep track of the number of photons being sampled in the current tracking batch.
 * Used to dynamically allocate threads for photon sampling.
 */
__device__ unsigned long long tracking_counter_sampling = 0;

/**
 * Global device variable for the number of grid cells in the radial direction.
 * It mirrors the host variable N1 but is accessible on the GPU.
 */
__device__ int d_N1;

/**
 * Global device variable for the number of grid cells in the poloidal direction.
 * It mirrors the host variable N2 but is accessible on the GPU.
 */
__device__ int d_N2;

/**
 * Global device variable for the number of grid cells in the azimuthal direction.
 * It mirrors the host variable N3 but is accessible on the GPU.
 */
__device__ int d_N3;    


/*iharm variables*/

/**
 * Global device variable for the ion-to-electron temperature ratio in the low-beta limit (\f$R_{low}\f$).
 * This defines the temperature ratio in highly magnetized regions (where \f$\beta \ll \beta_{\rm crit}\f$).
 */
__device__ double d_trat_small;

/**
 * Global device variable for the ion-to-electron temperature ratio in the high-beta limit (\f$R_{high}\f$).
 * This defines the temperature ratio in weakly magnetized regions (where \f$\beta \gg \beta_{\rm crit}\f$).
 */
__device__ double d_trat_large;

/**
 * Global device variable for the critical plasma beta value (\f$\beta_{\rm crit}\f$) that delineates the transition between low-beta and high-beta temperature ratios.
 */
__device__ double d_beta_crit;

/**
 * Global device variable for the maximum allowed dimensionless electron temperature (\f$\Theta_{e, max}\f$).
 * This sets an upper limit on the electron temperature to prevent unphysical values during the simulation.
 */
__device__ double d_thetae_max;

/**
 * Global device variable for the ion-to-electron temperature ratio (\f$T_p/T_e\f$).
 */
__device__ double d_tp_over_te;

/**
 * Global device variable that controls whether scattering is enabled (1) or disabled (0) in the simulation.
 */
__device__ int d_scattering;


/**
 * Global device variable that scales bias to match desired ratio.
 */
__device__ double d_biastuning[MAX_LAYER_SCA];

/**
 * Global device variable for the black hole mass in grams (g).
 */
__device__ double d_MBH;

/**
 * Global device variable for the length unit conversion factor. Scales code units to centimeters (cm). 
 */
__device__ double d_L_unit;

/**
 * Global device variable for the magnetic field unit conversion factor. Scales code units to Gauss (G). 
 */
__device__ double d_B_unit;

/**
 * Global device variable for the electron number density unit conversion factor. Scales code units to \f$ \rm{cm}^-3\f$. 
 */
__device__ double d_Ne_unit;

/**
 * Global host variable for the black hole spin parameter (\f$a_*\f$).
 */
double bhspin;

/**
 * Global device variable for the black hole spin parameter (\f$a_*\f$).
 * It's the same as the host variable bhspin but accessible on the GPU.
 */
__device__ double d_bhspin;

/** data structures **/

/**
 * @brief Structure to hold the metric components and determinant for a given grid point.
 */
struct of_geom {
	double gcon[NDIM][NDIM];
	double gcov[NDIM][NDIM];
	double g;
};

/**
 * @brief Structure to hold the spectrum data for a given grid point.
 */
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

/**
 * @brief Structure to hold simulation parameters parsed by the parameter file.
 */
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

/**
 * @brief Global host variable to hold simulation parameters parsed by the parameter file. 
 */
Params params = { 0 };


