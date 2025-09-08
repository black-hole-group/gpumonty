#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define N_BLOCKS 176
#define N_THREADS 256
#define N_RANDOM_NUMBERS 1000000

// Declare my_curand_state as a pointer
__device__ curandState my_curand_state[N_BLOCKS * N_THREADS];
gsl_rng *r;
extern "C" {
    // Kernel to initialize the random states
    __global__ void GPU_init_monty_rand(time_t time) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int seed = 139 * tid * time;
        curand_init(seed, tid, 0, &my_curand_state[tid]);
    }

    // Kernel to generate random numbers
    __global__ void GPU_monty_generate(double *random_numbers, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = tid; i < n; i += N_BLOCKS * N_THREADS) {
            random_numbers[i] = curand_uniform_double(&my_curand_state[tid]);
        }
    }

    // Host function to initialize random states
    void init_random_states() {
        // Allocate memory for my_curand_state on the device
        cudaError_t err;
        GPU_init_monty_rand<<<N_BLOCKS, N_THREADS>>>(time(NULL));
        cudaDeviceSynchronize(); // Ensure the kernel has finished executing

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in GPU_init_monty_rand: %s\n", cudaGetErrorString(err));
        }
    }

    // Host function to generate random numbers
    void generate_random_numbers(double *random_numbers, int n) {
        double *d_random_numbers;
        cudaError_t err = cudaMalloc(&d_random_numbers, n * sizeof(double));
        if (err != cudaSuccess) {
            printf("CUDA malloc error in generate_random_numbers: %s\n", cudaGetErrorString(err));
            return;
        }

        // Generate random numbers
        GPU_monty_generate<<<N_BLOCKS, N_THREADS>>>(d_random_numbers, n);
        cudaDeviceSynchronize(); // Ensure the kernel has finished executing

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in GPU_monty_generate: %s\n", cudaGetErrorString(err));
        }

        // Copy generated random numbers back to host
        err = cudaMemcpy(random_numbers, d_random_numbers, n * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA memcpy error in generate_random_numbers: %s\n", cudaGetErrorString(err));
        }

        cudaFree(d_random_numbers);
    }
}

void init_monty_rand(int seed)
{
	r = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
	gsl_rng_set(r, seed);
}

double monty_rand()
{
	return (gsl_rng_uniform(r));
}

// Function to generate GSL random numbers
void generate_gsl_random_numbers(double *gsl_random_numbers, int n) {
    init_monty_rand(139 + time(NULL));

    for (int i = 0; i < n; i++) {
        gsl_random_numbers[i] = gsl_rng_uniform(r); // Generate uniform random number
    }
}

// Main function to generate and dump random numbers
int main() {
    double *random_numbers = (double *)malloc(N_RANDOM_NUMBERS * sizeof(double));
    double *gsl_random_numbers = (double *)malloc(N_RANDOM_NUMBERS * sizeof(double));

    // Initialize random states for CUDA
    init_random_states();

    // Generate random numbers using CUDA
    generate_random_numbers(random_numbers, N_RANDOM_NUMBERS);

    // Generate random numbers using GSL
    generate_gsl_random_numbers(gsl_random_numbers, N_RANDOM_NUMBERS);

    // Open a binary file to write the CUDA random numbers
    FILE *cuda_file = fopen("cuda_random_numbers.bin", "wb");
    if (cuda_file == NULL) {
        printf("Error opening file for writing CUDA random numbers\n");
        free(random_numbers);
        free(gsl_random_numbers);
        return -1;
    }
    fwrite(random_numbers, sizeof(double), N_RANDOM_NUMBERS, cuda_file);
    fclose(cuda_file);

    // Open a binary file to write the GSL random numbers
    FILE *gsl_file = fopen("gsl_random_numbers.bin", "wb");
    if (gsl_file == NULL) {
        printf("Error opening file for writing GSL random numbers\n");
        free(random_numbers);
        free(gsl_random_numbers);
        return -1;
    }
    fwrite(gsl_random_numbers, sizeof(double), N_RANDOM_NUMBERS, gsl_file);
    fclose(gsl_file);

    // Print the count of valid random numbers
    int count_cuda = 0;
    int count_gsl = 0;

    for (int i = 0; i < N_RANDOM_NUMBERS; ++i) {
        if (random_numbers[i] != 0) {
            count_cuda++;
        }
        if (gsl_random_numbers[i] != 0) {
            count_gsl++;
        }
    }
    printf("Count of valid CUDA random numbers = %d\n", count_cuda);
    printf("Count of valid GSL random numbers = %d\n", count_gsl);

    // Free host memory
    free(random_numbers);
    free(gsl_random_numbers);

    // Cleanup random states (if necessary)
    gsl_rng_free(r); // Free the GSL random number generator
    cudaFree(my_curand_state); // Free device memory for curandState

    return 0;
}
