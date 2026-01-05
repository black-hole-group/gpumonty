
/* Range of superphoton frequencies */
#define NUMIN 1.e7
#define NUMAX 1.e16

/*This indicates the minimum of thetae = kTe/(mec^2)*/
#define THETAE_MIN	0.3
#define THETAE_MAX 1000.

/*Define the minimum weight of the superphoton to be considered*/
#define WEIGHT_MIN	(1.e31)

/*for stop criterium*/
#define RMAX	100. //Define the maximum radius up to track the photon
#define ROULETTE	1.e4 //Roulette to randomly increase superphoton weight
#define NPRIM	8

/**
 * Size of the energy bin in logarithmic scale for the spectral output binning.
 */
#define	dlE (0.01) //Size of the energy bin

/**
 * Minimum energy of the energy bin in logarithmic scale.
 */
#define lE0	(log(1.e-12)) //Minimum energy of the energy bin

/**
 * Number of energy samples for the emissivity and weight tables.
 */
#define N_ESAMP 2500

/**
 * Number of energy bins for the spectral output.
 */
#define N_EBINS 2500


#ifndef MODEL_FUNCTIONS
#define MODEL_FUNCTIONS
__host__ void init_storage(void);
__host__ void init_data();
__device__ int GPU_record_criterion(double X1);
__device__ int GPU_stop_criterion(double X1, double * w, curandState * localState);
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);
__host__ __device__ void coord(int i, int j, int k, double *X);
__host__ __device__ void gcov_func(const double *X , double gcov[][NDIM]);
__host__ double dOmega_func(double x2i, double x2f);
__host__ __device__ void bl_coord(const double *X, double *r, double *th);
__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,  double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *   d_geom, const double *  d_p);
__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne, double *Thetae, double *B, double Ucon[NDIM], double Ucov[NDIM], double Bcon[NDIM], double Bcov[NDIM], cudaTextureObject_t d_p);
__device__ double GPU_bias_func(double Te, double w, int round_scatt);
__device__ void GPU_record_super_photon(struct of_photonSOA ph, struct of_spectrum* d_spect, unsigned long long photon_index);
__device__ double GPU_stepsize(const double X[NDIM], const double K[NDIM]);
#endif
