#include <stdio.h>
#include <curand_kernel.h>

#define N_BLOCKS 10 // Number of blocks
#define THREADS_PER_BLOCK 256

__global__ void process(float *random_values, curandState *state) {
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    float random_value;

    // Initialize the random number generator for each thread
    curand_init(block_id, tid, 0, &state[tid]);

    // Each block handles one iteration of the outer loop
    do {
        // All threads generate a random number
        random_value = curand_uniform(&state[tid]);
        random_values[block_id * THREADS_PER_BLOCK + tid] = random_value;

        // Synchronize threads within the block
        __syncthreads();

        // If any thread's random value > 0.8, store and exit
        if (random_value > 0.8) {
            printf("Block %d, Thread %d: Random value > 0.8 = %f\n", block_id, tid, random_value);
            break; // Exit the inner loop
        }
        __syncthreads(); // Ensure all threads see the stored value
    } while (true); // Just one iteration for demonstration
}

int main() {
    float *random_values;
    curandState *devStates;
    cudaMalloc((void **)&random_values, N_BLOCKS * THREADS_PER_BLOCK * sizeof(float));
    cudaMalloc((void **)&devStates, THREADS_PER_BLOCK * sizeof(curandState));

    // Launch one kernel per block
    process<<<N_BLOCKS, THREADS_PER_BLOCK>>>(random_values, devStates);
    cudaDeviceSynchronize(); // Wait for all threads to finish

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaFree(random_values);
    cudaFree(devStates);

    return 0;
}
