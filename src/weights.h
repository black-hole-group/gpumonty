/*
 * GPUmonty - weights.h
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
Declaration of the functions used in weights.cu file
*/

#ifndef WEIGHTS_H
#define WEIGHTS_H



/**
 * @brief Precomputes the superphoton weight table based on the total integrated emissivity of the fluid.
 * This function evaluates the volume integral of the emissivity function \f$j_\nu\f$ across the entire 
 * simulation grid to determine the weight \f$\omega_\nu\f$ for each frequency bin.
 * * Implements: \f$ \omega_{\nu}(\nu) = \frac{\Delta t \Delta \ln \nu}{h N_s} \int \sqrt{-g} \, d^3x \int j_{\nu} d\Omega \f$
 *
 * @note The resulting weights are stored in log-space within the global `wgt` array for faster interpolation.
 */
__host__ void init_weight_table();

/**
 * @brief Precomputes lookup tables for the expected superphoton count and rejection sampling envelopes.
 * This function integrates the ratio of local emissivity \f$j_\nu\f$ to the precomputed superphoton 
 * weights \f$\omega_\nu\f$ across all frequency bins for a range of magnetic field strengths.
 * * Implements the integrated photon density:
 * 
 * \f$ N_{int}(B) = \Delta^3 x \Delta t \sqrt{-g} \int \int \frac{j_\nu}{h \omega_\nu} d\ln\nu d\Omega \f$
 * 
 * - `nint[i]`: Stores globally the log of the total number of superphotons to be generated per unit volume.
 * - `dndlnu_max[i]`: Stores globally the log of the maximum value of \f$dn/d\ln\nu\f$, used as the proposal 
 * distribution envelope for frequency sampling.
 */
__host__ void init_nint_table(void);

/**
 * @brief Computes the linear interpolation weight for a given frequency.
 * 
 * Calculates the interpolated superphoton weight between precomputed values in the weight table based on the input frequency `nu`.
 *
 * @param nu The frequency at which to compute the interpolated weight (\omega).
 * @return The computed linearly interpolated superphoton weight.
 */
__device__ double linear_interp_weight(const double nu);
#endif
