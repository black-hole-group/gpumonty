
/* Range of superphoton frequencies */
#define NUMIN 1.e7
#define NUMAX 1.e16

/*This indicates the minimum of thetae = kTe/(mec^2)*/
#define THETAE_MIN	0.3
#define THETAE_MAX 1000.

/*Ratio of proton temperature to electron temperature*/
#define TP_OVER_TE	(3.)

/*Define the minimum weight of the superphoton to be considered*/
#define WEIGHT_MIN	(0)

/*for stop criterium*/
#define RMAX	100 //Define the maximum radius up to track the photon
#define ROULETTE	1.e4 //Roulette to randomly increase superphoton weight




#ifndef MODEL_FUNCTIONS
#define MODEL_FUNCTIONS
__host__ void init_storage(void);
__host__ void init_data(char *fname);
__device__ int GPU_record_criterion(struct of_photon *ph);
__device__ int GPU_stop_criterion(struct of_photon *ph);
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);
__host__ __device__ void coord(int i, int j, double *X);
__host__ __device__ void gcov_func(double *X, double gcov[][NDIM]);
__host__ double dOmega_func(double x2i, double x2f);
__host__ __device__ void bl_coord(double *X, double *r, double *th);
#endif