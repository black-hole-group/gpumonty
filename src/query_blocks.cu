#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    // You can check if a device exists here to be safe
    if (cudaGetDevice(&device) != cudaSuccess) {
        // Fallback if no GPU is found during compilation
        printf("1024"); 
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate the number
    int max_blocks = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;
    
    // Print ONLY the number
    printf("%d", max_blocks);
    return 0;
}