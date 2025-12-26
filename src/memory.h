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
 * @return void
 */
__host__ void transferParams();

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
 * @brief Creates a 1D texture object from a 1D data array.
 *
 * Converts input `double` data to `float`, uploads it to a 1D CUDA array, and 
 * initializes a texture object with linear filtering and clamp addressing. 
 *
 * This is used for the Modified Bessel function lookup table (\f$ K_2 \f$).
 * @param texObj Pointer to the resulting texture object.
 * @param ptr  Input 1D data array.
 * @param cudaArray Pointer to the allocated 1D CUDA array resource.
 * 
 * @return void
 */
__host__ void create1DTextureObj(cudaTextureObject_t * texObj, double * ptr, cudaArray_t * cudaArray);

#endif