#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H

// #define N_BLOCKS 20
// #define BLOCK_SIZE 256

#define CUDASAFE(err) cudaAssert((err), __FILE__, __LINE__)
#define CUDAERRCHECK() cudaAssert(cudaGetLastError(), __FILE__, __LINE__)

// Assert the possible error err and abort execution with an appropriate
// message if it's an error.
void cudaAssert(cudaError_t err, const char *file, int line);

// Return's this thread unique id.
// Note: expects both grid and blocks to be unidimensional with y = z = 1
__device__
int gpu_thread_id();

#endif
