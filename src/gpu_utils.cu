#include <stdio.h>
#include "gpu_utils.h"

__device__
int gpu_thread_id() {
    return (blockDim.x * blockIdx.x) + threadIdx.x;
}

void cudaAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                    cudaGetErrorString(err));
            fflush(stderr);
            exit(EXIT_FAILURE);
    }
}
