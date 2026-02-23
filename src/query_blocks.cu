/*
 * GPUmonty - query_blocks.cu
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */
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
