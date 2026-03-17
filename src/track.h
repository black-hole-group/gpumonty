/*
 * GPUmonty - track.h
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
Declarations of the functions in the track.cu file
*/

#ifndef TRACK_H
#define TRACK_H

/**
 * 
 * @brief Main function responsible for tracking a superphoton through the spacetime while accounting for absorption and scattering.
 *
 * Contains the loop that iteratively updates the photon's position and momentum, checks for interactions, and modifies the photon's properties accordingly.
 *
 * @param ph The superphoton structure containing its properties.
 * @param d_p Device pointer or texture object for plasma data, depending on compilation settings.
 * @param d_table_ptr Pointer to the scattering probability and cross-section lookup tables.
 * @param scat_ofphoton A secondary SoA used to store the relevant local and instantaneous information of the scattered photon to be processed later.
 * @param starting_scattering_index The starting global photon index for the current scattering batch.
 * @param round_scat The current scattering layer index.
 * @param photon_index The global index of the photon being tracked.
 * @param localState Pointer to the curand state for random number generation.
 * 
 * @return void
 */
__device__ void track_super_photon(struct of_photonSOA ph , 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
    	double * __restrict__ d_p,
    #else
    	cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long starting_scattering_index, const int round_scat, const unsigned long long photon_index, curandState * localState);

/**
 * @brief Initializes the change in superphoton momentum per unit affine parameter \f$ \frac{dK^{\mu}}{d\lambda} \f$.
 * 
 * This function computes the initial values of \f$ \frac{dK^{\mu}}{d\lambda} \f$ based on the superphoton's position and momentum four-vectors.
 * 
 * @param X The superphoton's position four-vector.
 * @param Kcon The superphoton's contravariant momentum four-vector.
 * @param dK The output array to store the initialized \f$ \frac{dK^{\mu}}{d\lambda} \f$ values.
 * 
 * @return void
 * 
 */
__device__ void init_dKdlam(double X[], double Kcon[], double dK[]);


/**
 * @brief Pushes the superphoton's position and momentum forward by a small affine parameter step.
 * 
 * This function updates the superphoton's position and momentum four-vectors based on the computed changes per unit affine parameter and the specified step size.
 * 
 * @param X The superphoton's position four-vector.
 * @param Kcon The superphoton's contravariant momentum four-vector.
 * @param dKcon The change in superphoton's contravariant momentum per unit affine parameter.
 * @param dl The step size in affine parameter.
 * @param E0 Pointer to the superphoton's energy.
 * 
 * @return void
 */
__noinline__ __device__ void push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], const double dl, double *E0);
#endif