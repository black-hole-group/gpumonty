

#ifndef MODEL_FUNCTIONS
#define MODEL_FUNCTIONS
__host__ void init_storage(void);
__host__ void init_data(char *fname);
__device__ int GPU_record_criterion(struct of_photon *ph);
__device__ int GPU_stop_criterion(struct of_photon *ph);
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);
__host__ __device__ void coord(int i, int j, double *X);
__host__ __device__ void gcov_func(double *X, double gcov[][NDIM]);
#endif