#include "decs.h"
#include "utils.h"


__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]){
	double interp;

	interp = coeff[0] * var[NPRIM_INDEX3D(mmenemonics, i, j, k)] + coeff[5] * var[NPRIM_INDEX3D(mmenemonics, i+1, j, k)] +
	coeff[4] * var[NPRIM_INDEX3D(mmenemonics, i, j + 1, k)] + coeff[7]  * var[NPRIM_INDEX3D(mmenemonics, i+1, j+1, k)] +
	coeff[1] * var[NPRIM_INDEX3D(mmenemonics, i, j, k+1)] + coeff[6] * var[NPRIM_INDEX3D(mmenemonics, i+1, j, k+1)] +
	coeff[2] * var[NPRIM_INDEX3D(mmenemonics, i, j+1, k+1)] + coeff[3] * var[NPRIM_INDEX3D(mmenemonics, i+1, j+1, k+1)];

	return interp;
}
__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index) {
    int left = 0;
    int right = arraySize - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (cumulativeArray[mid] > photon_index) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left; // `left` is the smallest index where cumulativeArray[left] > photon_index
}