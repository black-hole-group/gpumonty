

#define exponential_coordinates (1)
/* Range of superphoton frequencies */
#define NUMIN 1.e7
#define NUMAX 1.e14

/*This indicates the minimum of thetae = kTe/(mec^2)*/
#define THETAE_MIN	0.3
#define THETAE_MAX 1000.

/*Ratio of proton temperature to electron temperature*/
#define TP_OVER_TE	(3.)

/*Define the minimum weight of the superphoton to be considered*/
#define WEIGHT_MIN	(0)

/*for stop criterium*/
#define RMAX	(1.e5/L_UNIT) //Define the maximum radius up to track the photon
#define ROULETTE	1.e4 //Roulette to randomly increase superphoton weight

//RMIN for sphere model only
#define RMIN (0.2/L_UNIT)

/*Some basic functions had to be changed to do the sphere_test, therefore, I had to create this switch.*/
#define SPHERE_TEST (1)
#define SCATTERING_TEST (1)

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
__host__ void check_scan_error(int scan_output, int number_of_arguments );
__host__ __device__ void bl_coord(double *X, double *r, double *th);
#endif