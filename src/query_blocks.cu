#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);

    if (err != cudaSuccess) {
        // Print warning to stderr so user sees it
        fprintf(stderr, " [PROBE] No GPU detected or CUDA driver error.\n");
        fprintf(stderr, " [PROBE] Defaulting to 1024 blocks.\n");
        // Print number to stdout for the Makefile
        printf("1024");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_blocks = 1 * prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;

    // --- LOGS (sent to stderr so they appear in terminal) ---
    fprintf(stderr, " [PROBE] Current GPU in use: %s\n", prop.name);
    fprintf(stderr, " [PROBE] Multiprocessors: %d\n", prop.multiProcessorCount);
    fprintf(stderr, " [PROBE] Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    fprintf(stderr, " [PROBE] Calculated optimal N_BLOCKS: %d\n", max_blocks);
    
    // --- RESULT (sent to stdout so Makefile captures it) ---
    printf("%d", max_blocks);

    return 0;
}
