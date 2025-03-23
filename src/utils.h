/*
Declaration of the functions in the utils.cu file
*/

#ifndef UTILS_H
#define UTILS_H
__device__ double GPU_interp_scalar(cudaTextureObject_t var, const int mmenemonics, const int i, const int j, const int k, const double del[4]);

__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index);
#endif
