/*
 * GPUmonty - utils.h
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
Declaration of the functions in the utils.cu file
*/

#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Interpolates a scalar field using hardware-accelerated CUDA Texture Objects.
 *
 * This function retrieves the value of a physical variable at a sub-grid location. 
 * Unlike `interp_scalar_pointer`, which performs a manual 8-point trilinear 
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
__device__ double interp_scalar(cudaTextureObject_t var, const int mmenemonics, const int i, const int j, const int k, const double del[4]);



/**
 * @brief Evaluates the Gauss Hypergeometric function 2F1(a, b; c; z) using a power series.
 * 
 * This device function computes the hypergeometric series directly. It is intended 
 * for use as a building block for the full hypergeometric evaluation, typically 
 * when |z| < 1 or as part of a transformation formula.
 * 
 * @param a Parameter 'a' of the 2F1 function.
 * @param b Parameter 'b' of the 2F1 function.
 * @param c Parameter 'c' of the 2F1 function.
 * @param z The argument of the function.
 * 
 * @return The computed value of the hypergeometric function as a double.
 * 
 * @note This implementation has been cross-validated against GSL (gsl_sf_hyperg_2F1) 
 * and SciPy (scipy.special.hyp2f1). Internal testing indicates that while 
 * this version maintains high agreement with SciPy across the tested domain, 
 * GSL (v2.7+) exhibits significant precision loss and stability issues 
 * when X < -0.5 (corresponding to z > 0.5), leading to large discrepancies 
 * compared to both this CUDA implementation and SciPy.
 */
__device__ double cuda_hyperg_2F1(double a, double b, double c, double z);

// /**
//  * @brief Computes the Gamma function, \f$\Gamma(z)\f$, for GPU device code.
//  *
//  * @details This function serves as a CUDA `__device__` substitute for `gsl_sf_gamma`. 
//  * It calculates the double-precision Gamma function using the [https://en.wikipedia.org/wiki/Lanczos_approximation](Lanczos approximation) 
//  * (with \f$g = 7\f$). It implements Euler's reflection formula to accurately handle 
//  * inputs where \f$z < 0.5\f$ (including negative numbers) and uses logarithmic 
//  * exponentiation to prevent intermediate floating-point overflow for large inputs.
//  *
//  * @param z The real-valued input for which to compute the Gamma function.
//  * @return The computed value of \f$\Gamma(z)\f$.
//  *
//  * @note Because the Gamma function grows factorially, inputs strictly greater 
//  * than 171.624 will overflow the IEEE 64-bit double limit and return `inf`.
//  * 
//  * @note This function has been tested against `gsl_sf_gamma` for a range of inputs, including negative values, and shows near truncation value agreements.
//  */
// __device__ double cuda_sf_gamma(double z);

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
 * @param i Base index in the first dimension (\f$N_1\f$).
 * @param j Base index in the second dimension (\f$N_2\f$).
 * @param k Base index in the third dimension (\f$N_3\f$).
 * @param coeff An array of 8 weights representing the volumetric contribution of each corner.
 *
 * @return The interpolated scalar value at the target location.
 *
 * @note The function assumes periodicity in the third dimension (\f$k\f$). If the 
 * index \f$k\f$ is at the grid boundary (\f$d\_N3 - 1\f$), it wraps around to index 0.
 */
__device__ double interp_scalar_pointer(const double * __restrict__ var, const int mmenemonics, const int i, const int j, const int k, const double coeff[8]);

/**
 * @brief Maps a 4D index (Variable + 3D Space) to a 1D linear memory offset.
 * * This macro implements a "Variable-Major" or "Structure of Arrays" (SoA) layout. 
 *
 * **Memory Layout Order**: [Variable \f$v\f$][Dimension \f$i\f$][Dimension \f$j\f$][Dimension \f$k\f$]
 * - \f$k\f$ is the fastest-varying index (unit stride).
 * - \f$v\f$ is the slowest-varying index.
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
 * **Memory Layout Order**: [Dimension \f$i\f$][Dimension \f$j\f$]
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
 * **Memory Layout Order**: [Dimension \f$i\f$][Dimension \f$j\f$][Dimension \f$k\f$]
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
