/*
Declaration of the functions in the utils.cu file
*/

#ifndef UTILS_H
#define UTILS_H
__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]);
__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index);
#endif
