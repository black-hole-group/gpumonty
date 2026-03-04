
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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



/**
 * The number of thread blocks dispatched per kernel call. 
 * This defines the grid size and should ideally be a multiple of the number of Streaming Multiprocessors (SMs) on your GPU to maximize occupancy.
 * This is set through automatic GPU tuning in the Makefile unless ``BLOCK_TUNING`` is set to 0 in the Makefile.
 */
#define N_BLOCKS 3456

/**
 * The number of threads per block. 
 * This defines the block size and should be chosen as a multiple of the warp size (32 for NVIDIA GPUs) 
 * to ensure efficient execution up to 2048 depending on the GPU architecture.
 */
#define N_THREADS 256

/**
 * Number of compton scatterings tracked in the spectrum. e.g., once, twice, >twice
 */
#define N_COMPTBINS (3)
/**
 * Number of total types of photons (synch and brems) * N_COMPTBINS
 */
#define N_TYPEBINS (2*(N_COMPTBINS+1))

/**
 * Number of theta bins for angular distribution for the spectral output binning.
 */
#define N_THBINS	6


/*Compton cross section calculation */

/**
 * Minimum dimensionless photon frequency (or energy) used to construct the Compton cross-section lookup table
 */
#define MINW      1.e-12   

/**
 * Maximum dimensionless photon frequency (or energy) used to construct the Compton cross-section lookup table
 */
#define MAXW      1.e15

/**
 * Minimum dimensionless temperature (\f$ \Theta_{\rm e} \f$) used to construct the Compton cross-section lookup table
*/
#define MINT      1.e-4     

/**
 * Maximum dimensionless temperature (\f$ \Theta_{\rm e} \f$) used to construct the Compton cross-section lookup table
 */
#define MAXT      1.e4

/**
 * Number of energy bins for the Compton cross-section lookup table
 */
#define NW        220

/**
 * Number of dimensionless temperature (\f$ \Theta_{\rm e} \f$) bins for the Compton cross-section lookup table
 */
#define NT        80     

/**
 * Standard path to the Compton cross-section table file
 */
#define HOTCROSS  "./table/hotcross.dat" 

/*Hot cross routines*/

/**
 * The integration cutoff multiplier for the electron Lorentz factor.
 * MAXGAMMA tells the code how far into the high-energy "tail" of the electron distribution 
 * it needs to integrate to get an accurate result. The integration goes up to \f$\gamma_\rm{e} = 1 + \text{MAXGAMMA} \Theta_{\rm e}\f$.
 */
#define MAXGAMMA	12. 

/**
 * The step size for the angular integration variable (\f$ \mu_\rm{e} \f$), that represents 
 * the cosine of the angle between the photon's direction and the electron's velocity
 * in the Compton cross-section calculation.
 */
#define DMUE		0.05 

/**
 * The step size resolution for the electron Lorentz factor (\f$ \gamma_\rm{e} \f$) integration
 * in the Compton cross-section calculation.
 */
#define DGAMMAE		0.05 //Stepsize for Gamma_e


/**
 * Mean molecular weight, in units of proton mass (\f$ m_p \f$).
 */
#define MMW	0.5		


/**
 * Number of dimensions in spacetime, time + 3 spatial dimensions.
 */
#define NDIM	4



/**
 * Mmnemonics for the density primitive variable (\f$ \rho \f$) array index.
 */
#define KRHO    0

/**
 * Mmnemonics for the internal energy primitive variable (\f$ u \f$) array index.
 */
#define UU      1

/**
 * Mmnemonics for the 1st-spatial velocity primitive variable (\f$ u^1 \f$) array index.
 */
#define U1      2

/**
 * Mmnemonics for the 2nd-spatial velocity primitive variable (\f$ u^2 \f$) array index.
 */
#define U2      3

/**
 * Mmnemonics for the 3rd-spatial velocity primitive variable (\f$ u^3 \f$) array index.
 */
#define U3      4

/**
 * Mmnemonics for the 1st-spatial magnetic field primitive variable (\f$ B^1 \f$) array index.
 */
#define B1      5

/**
 * Mmnemonics for the 2nd-spatial magnetic field primitive variable (\f$ B^2 \f$) array index.
 */
#define B2      6

/**
 * Mmnemonics for the 3rd-spatial magnetic field primitive variable (\f$ B^3 \f$) array index.
 */
#define B3      7

/**
 * Mmnemonics for the electron temperature primitive variable (\f$ T_e \f$) array index.
 */
#define KEL     8 

/**
 * Mmnemonics for the Ktot 
 */
#define KTOT    9


/**
 * Small number to avoid division by zero or logarithm of zero.
 */
#define SMALL	1.e-40



/**
 * Calculates the 1D index for a 2D array stored in a 1D format.
 */
#define NPRIM_INDEX(i, j) (j * NPRIM + i)


/**
 * Macro to loop over all spatial grid points in device or host code, depending on the compilation context.
 */
#ifdef __CUDA_ARCH__
#define SLOOP_DEVICE(i,j,k) \
    for (int i = 0; i < d_N1; i++) \ for (int j = 0; j < d_N2; j++) \ for (int k = 0; k < d_N3; k++)
#else
#define SLOOP_DEVICE(i,j,k) \
    for (int i = 0; i < N1; i++) \ for (int j = 0; j < N2; j++) \ for (int k = 0; k < N3; k++)
#endif

/**
 * Macro to loop over all dimensions in spacetime twice (used for tensors).
 */
#define DLOOP  for(k=0;k<NDIM;k++)for(l=0;l<NDIM;l++)

/**
 * Macro to compute the maximum of two values.
 */
#define MAX(a,b) (((a)>(b))?(a):(b))


/*Some device global variables definition*/

/**
 * Logarithm of the minimum dimensionless temperature (\f$ \Theta_{\rm e} \f$) for the Compton cross-section table.
 */
#define d_lmint     (log10(MINT))

/**
 * Logarithm of the minimum dimensionless photon frequency for the Compton cross-section table.
 */
#define d_lminw     (log10(MINW))

/**
 * TMIN represents the minimum dimensionless electron temperature (\f$ \Theta_{\rm e} \f$) used in
 * both \f$ K_2(1/\Theta_{\rm e}) \f$ calculations and synchrotron emissivity table generation.
 */
#define d_lT_min    (log(TMIN))

/**
 * Logarithmic step size for the Compton cross-section table in the photon frequency dimension.
 */
#define d_dlw       (log10(MAXW / MINW) / NW)

/**
 * Precomputed inverse of the logarithmic temperature step size for emissivity and \f$ K_2(1/\Theta_{\rm e}) \f$ calculations.
 */
#define d_dlT1       (1/(log(TMAX / TMIN) / (N_ESAMP)))

/**
 * Logarithmic step size for the Compton cross-section table in the dimensionless temperature (\f$ \Theta_{\rm e} \f$) dimension.
 */
#define d_dlT2       (log10(MAXT/MINT) / NT)




/*Used for synchrotron emissivity calculation of the table*/ 

/**
 * Maximum boundery for the dimensionless frequency grid used to precompute the synchrotron emissivity table.
 */
#define KMAX (1.e7)

/**
 * Minimum boundery for the dimensionless frequency grid used to precompute the synchrotron emissivity table.
 */
#define KMIN (0.002)

/**
 * Precomputed definitin of small vector value to avoid numerical issues in tetrad calculations.
 */
#define SMALL_VECTOR (1.e-30)

/**
 * The maximum allowable absolute error. This provides a "floor" for the error. 
 * If the result of the integral is very close to zero, this prevents the integrator from working forever to find an infinitely small
 * relative error.
 */
#define EPSABS (0.)

/**
 * The maximum allowable relative error (as a fraction). 
 * For example, a value of \f$10^{-6}\f$ means you want the result to be accurate to within \f$0.0001\%\f$.
 */
#define EPSREL (1.e-6)

/**
 * Minimum dimensionless electron temperature (\f$ \Theta_{\rm e} \f$) for the bessel function (\f$ K_2 \f$) table calculation.
 */
#define TMIN (THETAE_MIN)

/**
 * Maximum dimensionless electron temperature (\f$ \Theta_{\rm e} \f$) for the bessel function (\f$ K_2 \f$) table calculation.
 */
#define TMAX (1.e2)

/**
 * Number of scatterings each superphoton is allowed to undergo.
 * @note This serves as a estimation for memory allocation and should be adjusted based on the specific simulation requirements. If
 * the medium is optically thin, a value of 1 is sufficient. However, for optically thick scenarios, consider increasing this value.
 * A high number of scatterings per photon may lead to increased memory usage and increase in the number of serialized batches.
 */
#define SCATTERINGS_PER_PHOTON (1) 

/*Making of Nint table*/

/**
 * Number of data points in the Nint table used for solid angle averaged synchrotron emissivity calculations.
 */
#define	NINT (40000) //Number of table data

/**
 * Minimum value of the product \f$ B \theta_e^2 \f$ used in the Nint table.
 */
#define BTHSQMIN	(1.e-8)

/**
 * Maximum value of the product \f$ B \theta_e^2 \f$ used in the Nint table.
 */
#define BTHSQMAX	(1.e9)


/**
 * Maximum number of scattering layers allowed.
 */
#define MAX_LAYER_SCA (3)

/**
 * Epsilon parameter used in the photon geodesic integration to scale the stepsize.
 * Decreasing this value increases accuracy but also computational cost.
 */
#define EPS (0.04)

/*Push photon routine*/
/**
 * Macro for fast copying of 4-element arrays.
 */
#define FAST_CPY(in,out) {out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];}

/**
 * Tolerance for the geodesic integration error.
 */
#define ETOL (1.e-3)

/**
 * Maximum number of iterations for the geodesic integration push photon function.
 */
#define MAX_ITER (2)

/**
 * Maximum number of integration steps for photon geodesic integration.
 */
#define MAXNSTEP (1280000)

/**
 * Two-level macro stringification: expands `s` first, then converts the result
 * to a string literal (useful when `s` is itself a macro like VERSION).
 */
#define xstr(s) str(s)
#define str(s) #s