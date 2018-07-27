#include <stdio.h>

// Those functions come from the post at:
// https://parallel-computing.pro/index.php/11-openacc/53-using-cuda-device-functions-from-openacc
// (Accessed in 27/07/18)

extern "C" __device__ void acc_printi(int i)
{
    int id = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
    printf("[GPU-THREAD %d]: %d\n", id, i);
}

extern "C" __device__ void acc_printd(double d)
{
    int id = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
    printf("[GPU-THREAD %d]: %lf\n", id, d);
}

extern "C" __device__ void acc_printf(float f)
{
    int id = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
    printf("[GPU-THREAD %d]: %f\n", id, f);
}
