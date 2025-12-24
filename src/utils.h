/*
Declaration of the functions in the utils.cu file
*/

#ifndef UTILS_H
#define UTILS_H
__device__ double GPU_interp_scalar(cudaTextureObject_t var, const int mmenemonics, const int i, const int j, const int k, const double del[4]);
__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index);
__device__ double GPU_interp_scalar_pointer(const double * __restrict__ var, const int mmenemonics, const int i, const int j, const int k, const double coeff[8]);
__host__ __device__ inline int NPRIM_INDEX3D(int v, int i, int j, int k){
    #ifdef __CUDA_ARCH__
        return (((v) * d_N1 * d_N2 * d_N3) + (k + d_N3 * (j + d_N2 * i)));
    #else
        return (((v) * N1 * N2 * N3) + (k + N3 * (j + N2 * i)));
    #endif
}

__host__ __device__ inline int SPATIAL_INDEX2D(int i, int j){
    #ifdef __CUDA_ARCH__
        return (j + d_N2 * i);
    #else
        return (j + N2 * i);
    #endif
}

__host__ __device__ inline int SPATIAL_INDEX3D(int i, int j, int k){
    #ifdef __CUDA_ARCH__
        return (k + d_N3 * (j + d_N2 * i));
    #else
        return (k + N3 * (j + N2 * i));
    #endif
}
#endif
