

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
__host__ __device__ void vofx_matthewcoords(double *X, double *V);
__host__ void check_scan_error(int scan_output, int number_of_arguments );
__host__ __device__ void bl_coord(double *X, double *r, double *th);
#endif