
/* Range of superphoton frequencies */
#define NUMIN 1.e8
#define NUMAX 1.e16
#define IHARM (1)
#define DO_NOT_USE_TEXTURE_MEMORY 1 /*Define this macro to not use texture memory for fluid variables*/
/*This indicates the minimum of thetae = kTe/(mec^2)*/
#define THETAE_MIN	0.3
#define THETAE_MAX 1000.
/*Ratio of proton temperature to electron temperature*/
#define TP_OVER_TE	(3.)

/*Define the minimum weight of the superphoton to be considered*/
#define WEIGHT_MIN	(1.e28)

/*for stop criterium*/
#define RMAX	1000. //Define the maximum radius up to track the photon
#define ROULETTE	1.e4 //Roulette to randomly increase superphoton weight

#define N1 288
#define N2 128
#define N3 128
#define BHSPIN 0.9375

/*Mass of the black hole and the unit of M in order to transform to natural code units*/
#define MBH (4.14e6)/*In solar UNITs*/
#define M_UNIT (1.e16)
#define L_UNIT (GNEWT * MBH * MSUN/(CL * CL)) /* UNIT of length*/
#define RHO_UNIT (M_UNIT/pow(L_UNIT,3)) /*UNIT of density*/
#define NPRIM	8

#ifndef MODEL_FUNCTIONS
#define MODEL_FUNCTIONS
__host__ void init_storage(void);
__host__ void init_data(char *fname);
__device__ int GPU_record_criterion(double X1);
__device__ int GPU_stop_criterion(double X1, double * w, curandState * localState);
__device__ void GPU_Xtoijk(const double X[NDIM], int *i, int *j, int *k, double del[NDIM]);
__host__ __device__ void coord(int i, int j, double *X);
__host__ __device__ void gcov_func(const double *X , double gcov[][NDIM]);
__host__ double dOmega_func(double x2i, double x2f);
__host__ __device__ void bl_coord(const double *X, double *r, double *th);
__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,  double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *   d_geom, const double *  d_p);
__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne, double *Thetae, double *B, double Ucon[NDIM], double Ucov[NDIM], double Bcon[NDIM], double Bcov[NDIM], double * d_p);
__device__ double GPU_bias_func(double Te, double w, int round_scatt);
__device__ void GPU_record_super_photon(struct of_photonSOA ph, struct of_spectrum* d_spect, unsigned long long photon_index);
__device__ double GPU_stepsize(const double X[NDIM], const double K[NDIM]);
__host__ __device__ double thetae_func(double uu, double rho, double B, double kel);
__host__ void init_geometry();

#endif


//Parameters that probably should be read from file or job submission
#define BETA_CRIT 1.
#define TRAT_SMALL 1.
#define TRAT_LARGE 10.
#define GAME (4./3.)
#define GAMP (5./3.)
#define GAM (1.4444440)
#define USE_FIXED_TPTE (0)
#define USE_MIXED_TPTE (1)
#define WITH_ELECTRONS (2)
#define Thetae_MAX2 (1.e100)

#define ZLOOP \
    for (int i = 0; i < N1; i++) \
        for (int j = 0; j < N2; j++) \
            for (int k = 0; k < N3; k++)
