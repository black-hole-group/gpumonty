#include <stdio.h>
#include "kernel.h"    
#define TX 8 // number of threads per block along x-axis
#define TY 8 // number of threads per block along y-axis
#define TZ 8 // number of threads per block along z-axis

int divUp(int a, int b) { return (a + b - 1) / b; }


__global__ void testKernel(double *d_p, int nprim, int n1, int n2)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x; // column
	const int r = blockIdx.y * blockDim.y + threadIdx.y; // row
	const int s = blockIdx.z * blockDim.z + threadIdx.z; // stack
	const int h=nprim;
	const int w=n1;
	const int d=n2;
	const int i = c + r*w + s*w*h;
	if ((c >= w) || (r >= h) || (s >= d)) return;

	d_p[i]=d_p[i]*10.;
	printf("p[%d][%d][%d]=%lf\n", r,c,s,d_p[i]);
}


void launchKernel(double *d_p, int nprim, int n1, int n2) {
	const dim3 blockSize(TX, TY, TZ);
	const int H=nprim;
	const int W=n1;
	const int D=n2;
	const dim3 gridSize(divUp(W, TX), divUp(H, TY), divUp(D, TZ));

	testKernel<<<gridSize, blockSize>>>(d_p, nprim, n1, n2);
	//testKernel<<<1,1>>>(d_p, nprim, n1, n2);
	cudaDeviceSynchronize();
}