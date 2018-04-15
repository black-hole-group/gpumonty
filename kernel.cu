#include <stdio.h>
#include "kernel.h"    
#define TPB 32


__global__ void testKernel(double *d_p, int nprim, int n1, int n2)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int l=0; l<nprim; l++) {
		for (int j=0; j<n1; j++) {
			for (int k=0; k<n2; k++) {
				printf("p[%d][%d][%d]=%lf\n", l,j,k,d_p[l][j][k]);
			}
		}
	}
}