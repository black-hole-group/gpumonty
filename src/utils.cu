#include "decs.h"
#include "utils.h"


__device__ double interp_scalar_pointer(const double * __restrict__ var, const int mmenemonics, const int i, const int j, const int k, const double coeff[8]){
	double interp;
    int kp1 = k + 1;
    int ip1 = i + 1;
    int jp1 = j + 1;
    if (k == (d_N3 - 1)){
        kp1 = 0;
    }

	interp = coeff[0] * var[NPRIM_INDEX3D(mmenemonics, i, j, k)] + coeff[5] * var[NPRIM_INDEX3D(mmenemonics, ip1, j, k)] +
	coeff[4] * var[NPRIM_INDEX3D(mmenemonics, i, jp1, k)] + coeff[7]  * var[NPRIM_INDEX3D(mmenemonics, ip1, jp1, k)] +
	coeff[1] * var[NPRIM_INDEX3D(mmenemonics, i, j, kp1)] + coeff[6] * var[NPRIM_INDEX3D(mmenemonics, ip1, j, kp1)] +
	coeff[2] * var[NPRIM_INDEX3D(mmenemonics, i, jp1, kp1)] + coeff[3] * var[NPRIM_INDEX3D(mmenemonics, ip1, jp1, kp1)];

	return interp;
}

__device__ double interp_scalar(cudaTextureObject_t var, const int mmenemonics, const int i, const int j, const int k, const double del[4]){
    if(d_N3 == 1){
        return  tex3D<float>(var, (k) * NPRIM + mmenemonics + 0.5f, (j + del[2]) + 0.5f, (i + del[1]) + 0.5f);
    }
	return (1 - del[3]) * tex3D<float>(var, (k) * NPRIM + mmenemonics + 0.5f, (j + del[2]) + 0.5f, (i + del[1]) + 0.5f)+ del[3] * tex3D<float>(var, (k+1) * NPRIM + mmenemonics + 0.5f ,(j + del[2]) + 0.5f, (i + del[1]) + 0.5f);
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
