#include <stdio.h>
#include <stdlib.h>

#define N 1024

// Error checking macro
#define CUDA_ERROR_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(result), __LINE__); \
        exit(result); \
    } \
} while(0)

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tex1D<float>(tex, i + 0.5);
    printf("TID = %d, x = %le\n", i, x);

}

void call_kernel(cudaTextureObject_t tex) {
    kernel <<<1, 1024>>>(tex);
    CUDA_ERROR_CHECK(cudaGetLastError()); // Check for kernel launch error
    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish
}

int main() {
    // declare and allocate memory
    float * h_buffer;
    h_buffer = (float *) malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        h_buffer[i] = i;
    }

    // create CUDA array
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CUDA_ERROR_CHECK(cudaMallocArray(&cuArray, &channelDesc, N, 0));
    CUDA_ERROR_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_buffer, N * sizeof(float), cudaMemcpyHostToDevice));

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex;
    CUDA_ERROR_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    // call kernel
    call_kernel(tex);

    // destroy texture object
    CUDA_ERROR_CHECK(cudaDestroyTextureObject(tex));

    // free CUDA array
    CUDA_ERROR_CHECK(cudaFreeArray(cuArray));

    // free host memory
    free(h_buffer);

    return 0;
}
