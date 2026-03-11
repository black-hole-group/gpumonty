/*
 * GPUmonty - /iharm_model/model.h
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

#ifndef MODEL_H
#define MODEL_H
    /**
     * Radiation flags 
     * TODO: implement said flags (currently used for io purposes). maybe use runtime parameters?
     * TODO: implement Bremss
     */
    #define SYNCHROTRON (1)
    #define BREMSSTRAHLUNG (0)
    #define COMPTON (1)

    /* Range of superphoton frequencies */

    /**
     * Lowest sampled frequency for superphotons in the simulation in Hz.
     */
    #define NUMIN 1.e8

    /**
     * Highest sampled frequency for superphotons in the simulation in Hz.
     * 
     * @note Superphotons with frequencies above NUMAX can be generated through scattering events. 
     */
    #define NUMAX 1.e24

    /**
     * @brief Natural logarithm of the lower sampling bound.
     *
     * Derived from `NUMIN` as `log(NUMIN)`, this value represents the
     * log-space minimum used for energy/frequency grid construction.
     */
    #define LNUMIN log(NUMIN)

    /**
     * @brief Natural logarithm of the upper sampling bound.
     *
     * Derived from `NUMAX` as `log(NUMAX)`, this value represents the
     * log-space maximum used for energy/frequency grid construction.
     */
    #define LNUMAX log(NUMAX)

    /**
     * @brief Log-space step size between samples.
     *
     * Computed from the log-range `[LNUMIN, LNUMAX]` divided by `N_ESAMP`,
     * giving the increment used to traverse the sampling domain uniformly
     * in natural-log space.
     */
    #define DLNU ((LNUMAX-LNUMIN)/N_ESAMP)

    /**
     * @brief Macro to identify the iharm GRMHD model.
     */
    #define IHARM (1)

    /**
     * Macro not to use texture memory for fluid variables table in device code. 
     */
    #define DO_NOT_USE_TEXTURE_MEMORY 1 


    /**
     * Minimum dimensionless electron temperature (\f$\Theta_{e, min}\f$) for physical validity.
     * 
     * This is used to characterize the lower bound for superphoton generation.
     */
    #define THETAE_MIN	0.001

    /**
     * Maximum dimensionless electron temperature (\f$\Theta_{e, max}\f$) for physical validity.
     */
    #define THETAE_MAX 1000.


    /**
     * Size of the energy bin in logarithmic scale for the spectral output binning.
     */
    #define	dlE (0.25) //Size of the energy bin

    /**
     * Minimum energy of the energy bin in logarithmic scale.
     */
    #define lE0	(log(1.e-12)) //Minimum energy of the energy bin

    /**
     * Number of energy samples for the emissivity and weight tables.
     */
    #define N_ESAMP 800

    /**
     * Number of energy bins for the spectral output.
     */
    #define N_EBINS 800


    /**
     * Minimum superphoton weight to be considered to the spectrum. Superphotons that fall below this weight will be terminated.
     */
    #define WEIGHT_MIN	(1.e28)

    /**
     * Maximum radius to track photons in code units. If photons exceed this radius, their tracking will stop and they will be recorded.
     */
    #define RMAX	1000. //Define the maximum radius up to track the photon

    /**
     * Roulette factor for photon termination based on weight.
     */
    #define ROULETTE	1.e4 //Roulette to randomly increase superphoton weight

    /** 
     * Number of primitive variables in the iharm model. This should match the value read from the HDF5 file.
    */
    #define NPRIM	10

    #ifndef MODEL_FUNCTIONS
    #define MODEL_FUNCTIONS


    /**
     * @brief Allocates memory on the host for fluid primitives and grid geometry.
     * 
     * This function reserves space in the system RAM for the 4D primitive variables array  
     * and the 2D spacetime metric (geometry) array. This must be called before 
     * reading any simulation snapshots or calculating the metric.
     * 
     * @return void
     */
    __host__ void init_storage(void);

    /**
     * @brief Main initialization routine for physical data and simulation grid.
     * 
     * This function performs the following critical tasks:
     * 1. Opens and parses the HDF5 dump file.
     * 2. Sets up the computational grid dimensions (N1, N2, N3).
     * 3. Configures the electron temperature model (Fixed, R-beta, or Custom).
     * 4. Loads the spacetime geometry (MKS/KS) and calculates the event horizon.
     * 5. Allocates memory and populates the primitive variable arrays.
     * 6. Calculates initial diagnostics (accretion rate, bias normalization).
     * 
     * @return void
     */
    __host__ void init_data();

    /**
     * @brief Determines if a photon has exited the simulation domain and should be recorded.
     *
     * This function checks if the photon has crossed a specific outer boundary.
     *
     * @param X1 The current internal radial coordinate (\f$X^1\f$) of the photon.
     * @return Returns 1 if the photon is outside the recording radius, 0 otherwise.
     */
    __device__ int record_criterion(double X1);

    /**
     * @brief Evaluates whether to terminate the integration of a photon's path.
     * This function checks for two types of termination:
     * 1. Physical: Falling into the Black Hole or leaving the domain.
     * 2. Statistical: Using a "Roulette" to terminate photons with negligible weights.
     *
     * @param X1 Current radial coordinate in code units.
     * @param w Pointer to the photon's weight (energy/intensity).
     * @param localState Pointer to the GPU's random number generator state.
     * 
     * @return Returns 1 if the photon should be stopped/deleted, 0 if tracking should continue.
     */
    __device__ int stop_criterion(double X1, double * w, curandState * localState);


    /**
     * @brief Maps continuous coordinates to discrete grid indices and interpolation weights.
     * 
     * Bridge between the geodesic integrator and the fluid grid.
     * It identifies the nearest cell and calculates the fractional offset (del) 
     * required for trilinear interpolation of fluid properties.
     *
     * @param X Input continuous internal coordinates \f$ X^\mu \f$.
     * @param i, j, k Output integer indices for the grid cell.
     * @param del Output array [1..3] containing the fractional distance within the cell [0, 1].
     * 
     * @return void
     */
    __device__ void Xtoijk(const double X[NDIM], int *i, int *j, int *k, double del[NDIM]);


    /**
     * @brief Maps discrete grid indices to continuous internal coordinates.
     * 
     * Calculates the coordinate of a cell center.
     *
     * @param i, j, k Integer grid indices.
     * @param X Output array to store the resulting coordinates \f$ X^\mu \f$.
     * 
     * @return void
     */
    __host__ __device__ void coord(const int i, const int j, const int k, double *X);

    /**
     * @brief Computes the covariant metric tensor \f$ g_μν \f$ in Modified Kerr-Schild (MKS) coordinates.
     *
     * It transforms the standard Kerr-Schild metric into the simulation's specific 
     * Modified Kerr-Schild (MKS) coordinate system using Jacobian factors (tfac, rfac, etc.).
     *
     * @param X Input internal coordinates \f$ X^\mu \f$.
     * @param gcov Output 4x4 array where the metric components will be stored.
     */
    __host__ __device__ void gcov_func(const double *X , double gcov[][NDIM]);

    /**
     * @brief Calculates the integrated solid angle between two polar grid boundaries.
     * This function computes the solid angle by evaluating the integral:
     * \f[
     * \Delta\Omega = \int_{0}^{2\pi} d\phi \int_{\theta_i}^{\theta_f} \sin\theta \, d\theta = 2\pi (\cos\theta_i - \cos\theta_f)
     * \f]
     * where \f$ \theta \f$ is the physical polar angle derived from the internal 
     * coordinate \f$ X^2 \f$.
     *
     * @param x2i Starting internal polar coordinate (X2).
     * @param x2f Ending internal polar coordinate (X2).
     * 
     * @return The solid angle in steradians.
     */
    __host__ double dOmega_func(double x2i, double x2f);

    /**
     * @brief Transforms internal simulation coordinates to physical Boyer-Lindquist coordinates.
     * This function maps the logarithmic radial coordinate and the stretched polar 
     * coordinate to their physical counterparts:
     * * \f[
     * r = e^{X^1}
     * \f]
     * \f[
     * \theta = \pi X^2 + \frac{1 - h}{2} \sin(2\pi X^2)
     * \f]
     * where \f$ h \f$ is the hslope parameter controlling equatorial refinement.
     * 
     * @note \f$ X^3 \f$ (azimuthal angle) remains unchanged in this transformation.
     *
     * @param X Input array of internal coordinates \f$ X^\mu \f$.
     * @param r Output pointer for the Boyer-Lindquist radius.
     * @param th Output pointer for the Boyer-Lindquist polar angle (theta).
     */
    __host__ __device__ void bl_coord(const double *X, double *r, double *th);

    /**
     * @brief Recovers physical fluid and magnetic field properties from primitive variables.
     * 
     * This function performs the transformation from the GRMHD code's primitive variables 
     * (relative to a normal observer) to the physical quantities in the fluid's local frame.
     * 
     * It calculates the 4-velocity \f$ u^\mu \f$ ensuring the normalization condition:
     * \f[ u_\mu u^\mu = -1 \f]
     * 
     * And the magnetic 4-vector \f$ b^\mu \f$ satisfying the ideal MHD condition:
     * \f[ b^\mu u_\mu = 0 \f]
     *
     * @param i, j, k Grid indices.
     * @param Ne [out] Physical electron number density \f$ n_e \f$.
     * @param Thetae [out] Dimensionless electron temperature \f$ \Theta_e \f$.
     * @param B [out] Fluid-frame magnetic field strength.
     * @param Ucon [out] Contravariant 4-velocity \f$ u^\mu \f$.
     * @param Bcon [out] Contravariant magnetic 4-vector \f$ b^\mu \f$.
     * @param d_geom Pointer to the geometry metric structure.
     * @param d_p Pointer to the primitive variable array.
     * 
     * @return void
     */
    __host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,
        double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *  d_geom, const double *  d_p);



        /**
     * @brief Samples fluid and magnetic properties at an arbitrary spatial point using trilinear interpolation.
     * * This function is called during geodesic integration to determine local plasma 
     * conditions. It performs the following sequence:
     * 1. Validates the photon's position against the grid boundaries.
     * 2. Computes weights for trilinear interpolation between the 8 neighboring cells.
     * 3. Interpolates primitive variables (density, energy, velocity, magnetic field).
     * 4. Reconstructs the 4-velocity and magnetic 4-vector using the local metric.
     * 5. Applies safety cuts.
     * 
     * @param X Continuous internal coordinates.
     * @param gcov [out] Local covariant metric tensor.
     * @param Ne [out] Interpolated electron number density.
     * @param Thetae [out] Interpolated dimensionless electron temperature.
     * @param B [out] Physical magnetic field magnitude.
     * @param Ucon, Ucov [out] Reconstructed 4-velocity vectors.
     * @param Bcon, Bcov [out] Reconstructed magnetic 4-vectors.
     * @param d_p Pointer to the global primitive variable array in GPU memory.
     * 
     * @return void
     */
    __device__ void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne, double *Thetae, double *B, double Ucon[NDIM], double Ucov[NDIM], double Bcon[NDIM], double Bcov[NDIM], double * d_p);

    /**
     * @brief Calculates the scattering bias parameter based on local temperature.
     * This function determines the "bias factor" used to adjust scattering probabilities. 
     * It prioritizes high-temperature regions (where \f$ \Theta_e \f$ is large) to improve the signal-to-noise ratio of the final spectrum.
     *
     * @param Te Dimensionless electron temperature (\f$ \Theta_e \f$).
     * @param w Current statistical weight of the photon.
     * @param round_scatt The current scattering iteration count.
     * 
     * @return The calculated bias factor used to rescale Monte Carlo probabilities.
     */
    __device__ double bias_func(double Te, double w, int round_scatt);

    /**
     * @brief Records a photon's final properties into the global spectrum array.
     * This function maps a photon's energy and polar exit angle to a specific energy and angular
     * bin in the spectrum. Because many threads may try to update the same 
     * bin simultaneously, it uses CUDA atomic operations to ensure thread safety.
     * 
     * @note Energy is binned logarithmically: \f$ i_E \approx \frac{\ln(E) - \ln(E_0)}{\Delta \ln E} \f$.
     * @note Polar angle \f$ \theta \f$ is binned into angular zones, mirrored across the equator.
     *
     * @param ph The Structure of Arrays (SOA) containing all photon data.
     * @param d_spect Pointer to the global spectrum structure array.
     * @param photon_index The specific index of the photon being recorded.
     * 
     * @return void
     */
    __device__ void record_super_photon(struct of_photonSOA ph, struct of_spectrum* d_spect, unsigned long long photon_index);

    /**
     * @brief Performs the calculation of the stepsize \f$ d\lambda \f$ for the geodesic integration.
     * The stepsize follows this formula:
     * 
     * \f$ d\lambda = \left[ \frac{|K^1| + \epsilon_{s}}{\epsilon_{e}} + \frac{|K^2| + \epsilon_{s}}{\epsilon_{e} \cdot \min(X^2, 1-X^2)} + \frac{|K^3| + \epsilon_{s}}{\epsilon_{e}} \right]^{-1} \f$
     * 
     * Where:
     * - \f$ \epsilon_{e} \f$ is the tolerance parameter (EPS).
     * - \f$ \epsilon_{s} \f$ is the safety floor (SMALL) to prevent division by zero.
     * - \f$ \min(X^2, 1-X^2) \f$ is the proximity factor that reduces the step size 
     * as the photon approaches the polar coordinate singularities at 0 or 1.
     */
    __device__ double stepsize(const double X[NDIM], const double K[NDIM]);

    /**
     * @brief Calculates the dimensionless electron temperature \f$ \Theta_e \f$.
     * This function provides several models for the electron-to-proton temperature ratio.
     * It is primarily used to map the single-fluid internal energy from GRMHD 
     * onto the electron populations that actually produce the radiation.
     *
     * @param uu Local internal energy density.
     * @param rho Local mass density.
     * @param B Local magnetic field strength.
     * @param kel Local entropy/electron constant (used for mode 1).
     * 
     * @return The effective dimensionless electron temperature, soft-clamped by Thetae_max.
     */
    __host__ __device__ double thetae_func(double uu, double rho, double B, double kel);


    /**
     * @brief Processes energy and angled binned simulation data to generate and save the final spectrum in hdf5 format.
     * 
     * This function converts the raw energy and photon counts accumulated in the 
     * spectral grid into physical units. It calculates the SED across different inclination angles, determines 
     * the average optical depths, and computes global simulation diagnostics such 
     * as total luminosity and accretion efficiency. Contains the fluid_header, params and output groups.
     * 
     * @param N_superph_made Total number of superphotons generated during the run.
     * @param spect 3D array containing the accumulated spectral data (Photon physical origin,Energy,Theta bins).
     * @param filename Name of the output file (saved in the `./output/` directory).
     */
    __host__ void report_spectrum_h5(unsigned long long N_superph_made, struct of_spectrum ***spect, const char * filename);

    #endif


    //Parameters that probably should be read from file or job submission

    /**
     * @brief Activates the constant temperature ratio model.
     * When enabled (1), the code assumes a globally uniform ratio between 
     * proton and electron temperatures (\f$ T_p/T_e \f$), defined by the 
     * `params.tp_over_te` variable.
     * - Sets `with_electrons = 0`.
     * - Calculates `Thetae_unit` based on a fixed partitioning 
     * of the internal energy, often assuming a non-relativistic monoatomic gas 
     * (\f$ \gamma = 5/3 \f$).
     */
    #define USE_FIXED_TPTE (0)

    /**
     * @brief Activates the \f$ R-\beta \f$ (Mixed) temperature ratio model.
     * When enabled (1), the code uses the plasma \f$ \beta \f$ to interpolate 
     * between two temperature ratios: `trat_small` (for highly magnetized regions 
     * like the jet) and `trat_large` (for the accretion disk).
     * - Sets `with_electrons = 2`.
     */
    #define USE_MIXED_TPTE (1)

    /**
     * @brief Selects the sub-grid physics model for electron thermodynamics.
     * 
     * * Available Models:
     * - **0: Fixed Ratio** * Assumes a constant ratio between proton and electron temperatures throughout 
     * the entire simulation domain. 
     * \f$ \Theta_e \propto \frac{u}{\rho} \f$.
     * - **1: Howes/Kawazura Model** * A kinetic-based prescription that tracks electron heating based on 
     * local plasma turbulence and entropy (\f$ \kappa_{el} \f$).
     * - **2: R-high / R-low Model (Default)** * The standard \f$\beta\f$-dependent model for Black Hole imaging. It varies the 
     * temperature ratio \f$ R = T_p/T_e \f$ based on the local plasma \f$ \beta \f$ 
     * (the ratio of gas to magnetic pressure):
     * - In the **Disk** (high \f$\beta\f$): Protons are much hotter than electrons (\f$ R \to R_{high} \f$).
     * - In the **Jet** (low \f$\beta\f$): Electrons and protons are closer in temperature (\f$ R \to R_{low} \f$).
     */
    #define WITH_ELECTRONS (2)


    /**
     * Macro for looping over all grid zones in 3D.
     */
    #define ZLOOP for (int i = 0; i < N1; i++) for (int j = 0; j < N2; j++) for (int k = 0; k < N3; k++)

    extern double mks3R0, mks3H0, mks3MY1, mks3MY2, mks3MP0;
    extern __device__ double d_mks3R0, d_mks3H0, d_mks3MY1, d_mks3MY2, d_mks3MP0;
    extern __device__ double d_poly_norm, d_poly_xt, d_poly_alpha, d_mks_smooth;
    extern __device__ int d_METRIC;


    #define METRIC_eKS 0
    #define METRIC_MKS 1
    #define METRIC_MKS3 2
    #define METRIC_FMKS 3
#endif