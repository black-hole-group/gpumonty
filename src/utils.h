/*
Declaration of the functions in the utils.cu file
*/

#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Interpolates a scalar field using hardware-accelerated CUDA Texture Objects.
 *
 * This function retrieves the value of a physical variable at a sub-grid location. 
 * Unlike `GPU_interp_scalar_pointer`, which performs a manual 8-point trilinear 
 * interpolation in software, this version uses the GPU's dedicated texture 
 * mapping units (TMUs) to accelerate the lookups.
 *
 * @note To handle 4D data (3 spatial dimensions + multiple physical variables), the 
 * spatial index 'k' and the variable index 'mmenemonics' are flattened into 
 * the texture's X-axis.
 * 
 * @note By fetching two discrete points, `k * NPRIM` and `(k+1) * NPRIM`, and blending 
 * them with `del[3]`, we ensure we are only interpolating the *same* variable 
 * across two different spatial slices.
 *
 * @param var The CUDA 3D texture object containing the fluid data.
 * @param mmenemonics The index of the specific variable (e.g., RHO, UU).
 * @param i, j, k Integer base indices for the grid cell.
 * @param del[4] Array of fractional offsets [0, 1] for the sub-grid position at (i,j,k).
 * 
 * @return The interpolated value of the scalar field.
 */
__device__ double GPU_interp_scalar(cudaTextureObject_t var, const int mmenemonics, const int i, const int j, const int k, const double del[4]);

/**
 * @brief Maps a global photon ID to a specific grid cell or source index using binary search.
 *
 * This function performs a search over a prefix cumulative array 
 * to determine which "bucket" a specific superphoton belongs to. This is used to identify at which spatial cell a superphoton 
 * was emitted from based on its unique global ID.
 *
 * @param cumulativeArray Pointer to the prefix sum array (e.g., cumulative photon counts per cell).
 * @param arraySize The number of elements in the array.
 * @param photon_index The unique ID of the photon being processed.
 * 
 * @return The index of the cell or source responsible for this photon.
 */
__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index);

/**
 * @brief Performs trilinear interpolation of a scalar field at a sub-grid location.
 *
 * This function calculates the value of a specific fluid variable at an arbitrary position within a 3D grid cell. It uses eight 
 * precomputed weights (coefficients) and the values at the eight surrounding 
 * grid nodes.
 *
 * @param var Pointer to the flattened 3D array containing the fluid data.
 * @param mmenemonics The index (mnemonic) of the specific scalar variable to interpolate.
 * @param i Base index in the first dimension ($N_1$).
 * @param j Base index in the second dimension ($N_2$).
 * @param k Base index in the third dimension ($N_3$).
 * @param coeff An array of 8 weights representing the volumetric contribution of each corner.
 *
 * @return The interpolated scalar value at the target location.
 *
 * @note The function assumes periodicity in the third dimension ($k$). If the 
 * index $k$ is at the grid boundary ($d\_N3 - 1$), it wraps around to index 0.
 */
__device__ double GPU_interp_scalar_pointer(const double * __restrict__ var, const int mmenemonics, const int i, const int j, const int k, const double coeff[8]);

/**
 * @brief Maps a 4D index (Variable + 3D Space) to a 1D linear memory offset.
 * * This macro implements a "Variable-Major" or "Structure of Arrays" (SoA) layout. 
 *
 * **Memory Layout Order**: [Variable $v$][Dimension $i$][Dimension $j$][Dimension $k$]
 * - $k$ is the fastest-varying index (unit stride).
 * - $v$ is the slowest-varying index.
 *
 * \f$ \text{Index} = v \times (N_1 \times N_2 \times N_3) + i \times (N_2 \times N_3) + j \times N_3 + k \f$
 * 
 * @param v The index of the physical variable (mnemonic).
 * @param i, j, k The 3D spatial coordinates.
 * @return The 1D linear index in the primitive variable array.
 */
__host__ __device__ inline int NPRIM_INDEX3D(int v, int i, int j, int k){
    #ifdef __CUDA_ARCH__
        return (((v) * d_N1 * d_N2 * d_N3) + (k + d_N3 * (j + d_N2 * i)));
    #else
        return (((v) * N1 * N2 * N3) + (k + N3 * (j + N2 * i)));
    #endif
}


/**
 * @brief Maps 2D spatial coordinates (i, j) to a 1D linear memory offset.
 *
 * This function performs a standard row-major (or height-major) flattening for 
 * 2D grids. 
 * 
 * **Memory Layout Order**: [Dimension $i$][Dimension $j$]
 *
 * \f$ \text{Index} = j + (N_2 \times i) \f$
 * 
 * @param i The radial spatial index.
 * @param j The angular spatial index.
 * 
 * @return The 1D linear integer offset for the given coordinates.
 *
 */
__host__ __device__ inline int SPATIAL_INDEX2D(int i, int j){
    #ifdef __CUDA_ARCH__
        return (j + d_N2 * i);
    #else
        return (j + N2 * i);
    #endif
}


/**
 * @brief Maps 3D spatial coordinates (i, j, k) to a 1D linear memory offset.
 *
 * This function flattens a 3D volume into a 1D array.
 *
 * **Memory Layout Order**: [Dimension $i$][Dimension $j$][Dimension $k$]
 * 
 * \f$ \text{Index} = k + N_3 \times (j + N_2 \times i) \f$
 *
 * @param i The radial spatial index (typically the radial or X dimension).
 * @param j The polar spatial index (typically the polar or Y dimension).
 * @param k The azimuthal spatial index (typically the azimuthal or Z dimension).
 * 
 * @return The 1D linear integer offset for the given 3D coordinates.
 *
 */
__host__ __device__ inline int SPATIAL_INDEX3D(int i, int j, int k){
    #ifdef __CUDA_ARCH__
        return (k + d_N3 * (j + d_N2 * i));
    #else
        return (k + N3 * (j + N2 * i));
    #endif
}
#endif
