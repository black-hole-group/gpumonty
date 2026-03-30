/*
 * GPUmonty - memory.h
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
/*
Declaration of the memory.cu functions
*/

#ifndef MEMORY_H
#define MEMORY_H


/**
 * @brief Synchronizes global host simulation parameters with device memory symbols.
 *
 * Copies physics constants, grid dimensions, lookup tables and model-specific parameters from the 
 * host to the device using `cudaMemcpyToSymbol`.
 *
 * @param stream CUDA stream to perform asynchronous memory transfers, allowing overlap with kernel execution for improved performance.
 *
 * @return void
 */
__host__ void transferParams(cudaStream_t stream);

/**
 * @brief Configures the optimal number of blocks to be used by CUDA kernels and generates GPU diagnostics.
 *
 * It multiplies the maximum number of blocks per multiprocessor by the total number of streaming multiprocessors (SMs) to 
 * determine the total maximum number of blocks that can be launched concurrently on the GPU.
 *
 * @return The calculated maximum number of blocks for kernel launches.
 *
 */
__host__ int setMaxBlocks();

/**
 * @brief Computes a cumulative sum of superphoton counts for zone-based photon sampling.
 *
 * Generates a cumulative sum array where the value at zone i equals the total number 
 * of superphotons to be generated up to and including zone i-1.
 *
 * @param generated_photons_arr Array containing the number of photons to generate in each zone.
 * @param d_index_to_ijk Array storing the cumulative sum of photons per zone.
 * @return void
 */
__host__ void cummulativePhotonsPerZone(unsigned long long * generated_photons_arr, unsigned long long * d_index_to_ijk);

/**
 * @brief 
 */
void symbolToDevice(const void* symbol, const void* src, size_t size, cudaStream_t stream);

void symbolFromDevice(void* dst, const void* symbol, size_t size, cudaStream_t stream);

/**
 * @brief Calculates the batch size for GPU photon processing.
 *
 * Evaluates the available GPU RAM memory to determine the optimal number of 
 * partitions. This prevents memory overflows by scaling the batch size 
 * against the footprint of the `of_photon` structure.
 *
 * @param tot_nph Total number of photons to be processed in the simulation.
 * @param batch_divisions Calculated number of batches.
 * @return The number of photons assigned to each individual GPU batch.
 */
__host__ unsigned long long photonsPerBatch(unsigned long long tot_nph, int * batch_divisions);

/**
 * @brief Allocates device memory for photon data using a Structure of Arrays (SoA) layout.
 *
 * Initializes a `of_photonSOA` structure by allocating separate memory 
 * buffers on the GPU for every photon property for both original and scattered photons at each batch. 
 * * Using a Structure of Arrays (SoA) instead of an Array of Structures (AoS) is a 
 * critical optimization that enables **coalesced memory access** patterns, 
 * significantly increasing throughput for CUDA kernels.
 * 
 *
 * @param ph Pointer to the `of_photonSOA` structure to be initialized.
 * @param size The number of photon slots to allocate in each array.
 *
 */
__host__ void allocatePhotonData(struct of_photonSOA *ph, unsigned long long size);


/**
 * @brief Deallocates device memory for a `of_photonSOA` structure.
 *
 * This function releases all individual GPU memory buffers that were previously allocated by `allocatePhotonData`. 
 *
 * @param ph Pointer to the `of_photonSOA` structure whose device members are to be freed.
 *
 */
__host__ void freePhotonData(struct of_photonSOA * ph);


/**
 * @brief Creates a 3D texture object from a 4D data grid for the plasma primitive properties.
 *
 * Converts input `double` data to `float`, uploads it to a 3D CUDA array, and 
 * initializes a texture object with point filtering and clamp addressing.
 *
 * @param texObj Pointer to the resulting texture object that stores the plasma primitive properties.
 * @param dP  Input plasma properties.
 * @param cuArray Pointer to the allocated 3D CUDA array resource.
 *
 * @note Maps 4D data into a 3D extent by combining the `nx` and `ny` dimensions.
 */
__host__ void createdPTextureObj(cudaTextureObject_t * texObj, double * dP, cudaArray_t * cuArray);



/**
 * @brief Function to transfer of_photonSoA structures between two different device structures.
 * 
 * @param from Pointer from where the memory is being transferred from.
 * @param to Pointer to where the memory is being transferred to.
 * @param size size of the arrays of the SoA
 */
__host__ void transferPhotonDataDevtoDev(struct of_photonSOA to, struct of_photonSOA from, unsigned long long size);



/**
 * @brief Allocates a 3D array of structure of_spectrum on the host and initializes it to zero.
 * 
 * @param dim1 Size of the first dimension.
 * @param dim2 Size of the second dimension.
 * @param dim3 Size of the third dimension.
 * 
 * @return A pointer to the allocated spectrum 3D array.
 */
__host__ struct of_spectrum*** Malloc3D_Contiguous(int dim1, int dim2, int dim3);


/**
 * @brief Frees a previously allocated spectrum 3D array.
 * 
 * @param ptr Pointer to the 3D array to be freed.
 * @param dim1 Size of the first dimension.
 * @param dim2 Size of the second dimension.
 * @param dim3 Size of the third dimension.
 * 
 * @return void
 */
void Free3D_Contiguous(struct of_spectrum ***ptr, int dim1);
#endif