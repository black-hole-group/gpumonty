#include <stdio.h>
#include "kernel.h"    
#define TPB 32 // number of threads per block 



__global__ 
void testKernel(double *d_p, int nprim, int n1, int n2, double *d_pharr, int nph, int nphvars)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nph) return;

	// shows X0 for each photon
	printf("photon[%d]=%lf\n", i,d_pharr[i*nphvars+2]);
}



void launchKernel(double *d_p, int nprim, int n1, int n2, double *d_pharr, int nph, int nphvars) 
{
	testKernel<<<(nph+TPB-1)/TPB, TPB>>>(d_p, nprim, n1, n2, d_pharr, nph, nphvars);
}