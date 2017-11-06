__host__ void cuda_get_connection(double X[4], double lconn[4][4][4]);
__host__ void has_error_happend(cudaError_t error);
__global__ void get_connection_kernel(double X[4], double lconn[64], double a, double hslope);
