#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel function to print values from texture
__global__ void printTextureKernel(double * texObj, int width, int height)
{
    for (int y = 0; y < height; y++) {
        printf("Row %d: ", y);
        for (int x = 0; x < width; x++) {
            // Read from the texture object
            float value = tex2D<float>(texObj, x, y);
            printf("%.1f ", value);
        }
        printf("\n");
    }
}

int main()
{
    // Define the 10x5 2D array with 1's only in row 1 at positions (1,0), (1,1), and (1,2)
    const int width = 10;
    const int height = 5;
    float h_array[height][width] = {0}; // Initialize all elements to 0
    
    // Set the specified positions to 1
    h_array[1][0] = 1.0f; // Position (1,0)
    h_array[1][1] = 1.0f; // Position (1,1)
    h_array[1][2] = 1.0f; // Position (1,2)

    // Print the matrix on the host for verification
    printf("Host matrix:\n");
    for (int y = 0; y < height; y++) {
        printf("Row %d: ", y);
        for (int x = 0; x < width; x++) {
            printf("%.1f ", h_array[y][x]);
        }
        printf("\n");
    }
    printf("\n");

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CHECK_CUDA_ERROR(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Copy data from host to device
    size_t spitch = width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, h_array, spitch, 
                                         width * sizeof(float), height, 
                                         cudaMemcpyHostToDevice));

    // Specify texture object parameters
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture description parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // Using non-normalized coordinates

    // Create texture object
    double * texObj = 0;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Launch kernel to print texture values
    printf("Device matrix (from texture):\n");
    printTextureKernel<<<1, 1>>>(texObj, width, height);

    // Wait for GPU to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Destroy texture object and free CUDA array
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    CHECK_CUDA_ERROR(cudaFreeArray(cuArray));

    // Reset device
    CHECK_CUDA_ERROR(cudaDeviceReset());
    
    return 0;
}