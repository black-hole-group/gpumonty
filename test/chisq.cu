#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>

#define NUM_SAMPLES 1000000  // Number of chi-squared samples to generate
#define BLOCK_SIZE 256      // Threads per block
#define GRID_SIZE 64        // Number of blocks
__device__ curandState my_curand_state[GRID_SIZE * BLOCK_SIZE]; // Array of curandState structures

// Initialize the random state for each thread
__device__ double GPU_monty_rand(curandState *state) {
    return curand_uniform_double(state);  // Using curand for uniform random
}

// Standard Exponential distribution using Monte Carlo method
__device__ double legacy_standard_exponential(curandState *state) {
    return -log(1 - GPU_monty_rand(state));
}

// Standard Gaussian using Box-Muller method
__device__ void legacy_gauss(curandState *state, double* out1, double* out2) {
    double f, x1, x2, r2;

    do {
        x1 = 2.0 * GPU_monty_rand(state) - 1.0;
        x2 = 2.0 * GPU_monty_rand(state) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    f = sqrt(-2.0 * log(r2) / r2);
    *out1 = f * x1;
    *out2 = f * x2;
}

// Standard Gamma distribution with a given shape
__device__ double legacy_standard_gamma(curandState * state, double shape) {
	double U, V, X, Y;

	if (shape == 1.0) {
		return legacy_standard_exponential(state);
	}
	else if (shape == 0.0) {
		return 0.0;
	} else if (shape < 1.0) {
		for (;;) {
		U = GPU_monty_rand(state);
		V = legacy_standard_exponential(state);
		if (U <= 1.0 - shape) {
			X = pow(U, 1. / shape);
			if (X <= V) {
			return X;
			}
		} else {
			Y = -log((1 - U) / shape);
			X = pow(1.0 - shape + shape * Y, 1. / shape);
			if (X <= (V + Y)) {
			return X;
			}
		}
		}
	} else {
        double b, c;
		b = shape - 1. / 3.;
		c = 1. / sqrt(9 * b);
		double out1, out2;
		for (;;) {
            legacy_gauss(state, &out1, &out2);
            X = out1;
		do {
            if(X == out2){
                legacy_gauss(state, &out1, &out2);
                X = out1;
            }else{
                X = out2;
            }
			V = 1.0 + c * X;
		} while (V <= 0.0);

		V = V * V * V;
		U = GPU_monty_rand(state);
		if (U < 1.0 - 0.0331 * (X * X) * (X * X))
			return (b * V);
		if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
			return (b * V);
		}
	}
}


// Chi-Squared distribution with 'df' degrees of freedom
__device__ double chi_square(curandState *state, int df) {
    return 2.0 * legacy_standard_gamma(state, 0.5 * df);
}
__device__ void GPU_init_monty_rand(int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &my_curand_state[tid]);
}

// Kernel to generate chi-squared random variables
__global__ void generate_chi_squared(int df, double *d_output, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize random state for each thread
    seed = seed + idx;
    GPU_init_monty_rand(seed);

    curandState * localState = my_curand_state[idx];


    if (idx < NUM_SAMPLES) {
        d_output[idx] = chi_square(&localstate, df);
        printf("d_output: %f\n", d_output[idx]);
        if(d_output[idx] == 0){
            printf("Error: %d\n", idx);
        }
    }
}

int main() {
    const int df = 5; // Example degrees of freedom
    double *d_output, *h_output;
    size_t size = GRID_SIZE * BLOCK_SIZE * sizeof(double);

    // Allocate memory for output data
    h_output = (double*)malloc(size);
    cudaMalloc(&d_output, size);

    // Set up CUDA kernel
    unsigned long long seed = time(NULL) + 139;  // Example seed value
    generate_chi_squared<<<GRID_SIZE, BLOCK_SIZE>>>(df, d_output, seed);
    cudaDeviceSynchronize();

    // Copy the data back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Write the results to a file
    FILE *f = fopen("chi_squared_data.txt", "w");
    for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; i++) {
        fprintf(f, "%f\n", h_output[i]);
    }
    fclose(f);

    // Clean up
    free(h_output);
    cudaFree(d_output);

    return 0;
}
