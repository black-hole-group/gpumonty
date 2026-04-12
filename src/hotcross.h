/*
 * GPUmonty - hotcross.h
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
Declaration of the functions in the hotcross.cu file
*/


#ifndef HOTCROSS_H
#define HOTCROSS_H

/**
 * @brief Initializes and validates the Compton cross-section lookup table.
 * 
 * This function prepares a 2D lookup table for the total Compton cross-section 
 * as a function of dimensionless photon frequency (w) and electron 
 * temperature (Theta_e), now accounting for the kappa distribution normalization.
 * 
 * 
 * @note The table is stored in log10-log10 space.
 * 
 * @return void
 */
__host__ void init_hotcross(void);

/**
 * @brief Computes the total Compton cross-section via lookup table and approximations.
 * 
 * This function selects the optimal physical regime to calculate the cross-section 
 * based on photon frequency \f$ w \f$ and electron temperature \f$ \Theta_e \f$.
 * 
 * **Routine**:
 * 1. **Thomson**: If \f$ w \Theta_e < 10^{-6} \f$, returns the Thomson cross-section.
 * 2. **Klein-Nishina**: If below table temperature limits, uses analytical expressions.
 * 3. **Interpolation**: If in-bounds, performs bilinear interpolation in \f$ \log_{10}-\log_{10} \f$ space.
 * 4. **Fallback**: If out-of-bounds, triggers a direct numerical calculation.
 * 
 * @param w Dimensionless photon frequency.
 * @param thetae Dimensionless electron temperature.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * @param d_table_ptr Restrict-qualified pointer to the device lookup table.
 * 
 * @return The calculated Compton cross-section.
 */
__device__ double total_compton_cross_lkup(double w, double thetae, double kappa, const double * __restrict__ d_table_ptr);

/**
 * @brief Performs numerical integration of the total Compton cross-section.
 * 
 * This function evaluates the cross-section by integrating over the electron Lorentz factor \f$ \gamma_e \f$ 
 * and collision angle \f$ \mu_e \f$. It utilizes a double loop to sum the contributions of the 
 * electron distribution function and includes analytical shortcuts for the Thomson and 
 * Klein-Nishina limits in low-temperature regimes.
 * 
 * @param w Dimensionless photon frequency.
 * @param thetae Dimensionless electron temperature.
 * @param norm Normalization constant for the electron distribution function when using non-thermal distributions like the kappa EDF.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * @return The numerically integrated cross-section normalized by \f$ \sigma_T \f$.
 */
__host__ __device__ double total_compton_cross_num(double w, double thetae, double norm, double kappa);

/**
 * @brief Transformed integrand for the kappa distribution normalization over u.
 *
 * Applies a u-substitution where u = \f$ \sqrt(\gamma - 1)\f$. This mathematical 
 * transformation completely eliminates the infinite derivative singularity 
 * at the lower limit of the integration domain, yielding a perfectly smooth 
 * function that is highly optimized for fixed-grid numerical integration.
 *
 * @param u The transformed integration variable, u = \f$ \sqrt(\gamma - 1) \f$.
 * @param thetae The dimensionless electron temperature.
 * @return The value of the transformed integrand at u.
 */
__host__ __device__ double kappa_integrand_u(double u, double thetae);

/**
 * @brief Evaluates the integral of the u-transformed kappa function using Composite Simpson's Rule.
 *
 * Performs numerical integration using a fixed, uniform grid. Because the 
 * integrand is perfectly smooth after the u-substitution, this method achieves 
 * great accuracy without the need for adaptive step sizing.
 *
 * @param a The lower bound of integration.
 * @param b The upper bound of integration.
 * @param N The number of subintervals (determines grid resolution).
 * @param thetae The dimensionless electron temperature.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * 
 * @return The definite integral of d_kappa_function_int_u from a to b.
 */
__host__ __device__ double simpsons_rule_u(double a, double b, int N, double thetae, double kappa);

/**
 * @brief Calculates the normalization constant for the kappa distribution on the GPU.
 *
 * Computes the integral of the kappa distribution over the physical domain and 
 * returns its inverse (1.0 / integral). The integration bounds are scaled dynamically 
 * based on the provided electron temperature.
 *
 * @note CPU libraries like GSL rely heavily on adaptive quadrature algorithms (like QAG), such as the algorithm used in kmonty and igrmonty was set. 
 * These algorithms use recursive interval subdivision to hit error targets. On a GPU, 
 * recursion and dynamic branching cause bad warp divergence, and explicit stack 
 * management blows up register pressure.
 * To solve this, we applied a mathematical u-substitution to eliminate the integrand's 
 * singularity, allowing us to drop the adaptive algorithm entirely. We replaced it with 
 * a fixed-grid Composite Simpson's Rule. We validated this approach against the original 
 * CPU implementation, achieving relative errors far below the ones used in igrmonty. Since 
 * the singularity is resolved, the grid resolution parameter N is hardcoded inside the function 
 * but can be easily dialed up or down to control the exact performance-to-precision tradeoff.
 *
 * @param thetae The dimensionless electron temperature.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * @return The calculated normalization constant.
 */
__host__ __device__ double dNdgammae_kappa_norm(double thetae, double kappa);


/**
 * @brief Wrapper to select the appropriate normalization constant for the electron distribution function based on the simulation parameters.
 * 
 * @param thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * 
 * @return The normalization constant for the selected distribution function. For power-law and thermal distributions, this returns 1.0 since they are already normalized by construction.
 */
__host__ __device__ double getnorm_dNdgammae(double thetae, double kappa);

/**
 * @brief Wrapper to select the appropriate electron distribution function based on the simulation parameters to calculate the electron distribution function for thermal or nonthermal electrons.
 * 
 * @param thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param gammae Electron Lorentz factor \f$ \gamma_e \f$.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component. This parameter is only used if the kappa distribution is selected; otherwise, it can be ignored or set to a default value.
 * 
 * @return The value of \f$ dN/d\gamma_e \f$ for the selected distribution function.
 */
__host__ __device__ double dNdgammae(double thetae, double gammae, double kappa);
/**
 * @brief Calculates the kappa electron distribution function.
 *
 * Returns the number of electrons per unit Lorentz factor \f$ \gamma_e \f$.
 *
 * @param thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param gammae Electron Lorentz factor \f$ \gamma_e \f$.
 * @param kappa The kappa parameter that characterizes the non-thermal tail of the distribution. This parameter controls the slope of the high-energy tail, with lower values indicating a more pronounced non-thermal component.
 * 
 * @return The value of \f$ dN/d\gamma_e \f$ for a non-thermal distribution.
 */
__host__ __device__ double dNdgammae_kappa(double thetae, double gammae, double kappa);

/**
 * @brief Calculates the Maxwell-Jüttner electron distribution function.
 * 
 * Returns the number of electrons per unit Lorentz factor \f$ \gamma_e \f$, normalized 
 * per unit proper electron number density. The implementation uses the modified Bessel function \f$ K_2(1/\Theta_e) \sim \Theta_{\rm e}^2\f$
 * approximation to ensure numerical stability in both relativistic and non-relativistic regimes.
 * 
 * @param thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param gammae Electron Lorentz factor \f$ \gamma_e \f$.
 * 
 * @return The value of \f$ dN/d\gamma_e \f$.
 */
__host__ __device__ double dNdgammae_th(double thetae, double gammae);


/**
 * @brief Calculates the powerlaw electron distribution function.
 * 
 * Returns the number of electrons per unit Lorentz factor \f$ \gamma_e \f$ for a power-law distribution
 * 
 * @param thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param gammae Electron Lorentz factor \f$ \gamma_e \f$.
 * 
 * @return The value of \f$ dN/d\gamma_e \f$ for a power-law distribution.
 */
__host__ __device__ double dNdgammae_powerlaw(double thetae, double gammae);

/**
 * @brief Computes the Doppler-boosted Compton cross-section in the lab frame.
 * 
 * Transforms the lab-frame photon frequency \f$ w \f$ into the electron rest frame 
 * to evaluate the Klein-Nishina cross-section. The result is weighted by the 
 * relative interaction rate \f$ (1 - \mu_e \beta) \f$ to account for the beaming 
 * and relative velocity of the collision.
 * 
 * @param w Dimensionless photon frequency in the lab frame.
 * @param mue Cosine of the angle between photon and electron velocity.
 * @param gammae Electron Lorentz factor \f$ \gamma_e \f$.
 * 
 * @return The effective boosted cross-section.
 */
__host__ __device__ double boostcross(double w, double mue, double gammae);

/**
 * @brief Evaluates the normalized Klein-Nishina cross-section \f$ \sigma_{KN} / \sigma_T \f$.
 *
 * Computes the total cross-section for a photon with dimensionless energy \f$ w \f$ in the electron rest frame. 
 * For \f$ w < 10^{-3} \f$, it uses the expansion \f$ 1 - 2w \f$. Otherwise, it calculates:
 * \f[ \sigma = \frac{3}{4} \left[ \frac{2}{w^2} + \left( \frac{1}{2w} - \frac{1+w}{w^3} \right) \ln(1+2w) + \frac{1+w}{(1+2w)^2} \right] \f]
 *
 * @param we Dimensionless photon frequency in the electron rest frame.
 * 
 * @return The total cross-section normalized by the Thomson cross-section.
 */
__host__ __device__ double hc_klein_nishina(double we);


/**
 * @brief Modified Bessel function of the first kind, order zero: \f$ I_0(x) \f$.
 * 
 * Uses polynomial approximations and asymptotic expansions for numerical stability.
 * 
 * @see [Numerical Recipes in C](https://ui.adsabs.harvard.edu/abs/1992nrca.book.....P/abstract)

 * @param xbess The input value \f$ x \f$.

 * @return The value of \f$ I_0(x) \f$.
 */
__host__ __device__ double bessi0(double xbess);

/**
 * @brief Modified Bessel function of the first kind, order one: \f$ I_1(x) \f$.
 * 
 * Implemented using polynomial fits for small \f$ x \f$ and asymptotic forms for large \f$ x \f$.
 * 
 * @see [Numerical Recipes in C](https://ui.adsabs.harvard.edu/abs/1992nrca.book.....P/abstract)
 *
 * @param xbess The input value \f$ x \f$.
 * 
 * @return The value of \f$ I_1(x) \f$.
 */
__host__ __device__ double bessi1(double xbess);

/**
 * @brief Modified Bessel function of the second kind, order zero: \f$ K_0(x) \f$.
 * Evaluated via log-polynomial approximations for \f$ x \le 2 \f$ and exponential-asymptotic forms for \f$ x > 2 \f$.
 * 
 * @see [Numerical Recipes in C](https://ui.adsabs.harvard.edu/abs/1992nrca.book.....P/abstract)
 *
 * @param xbess The input value \f$ x \f$.
 * 
 * @return The value of \f$ K_0(x) \f$.
 */
__host__ __device__ double bessk0(double xbess);

/**
 * @brief Modified Bessel function of the second kind, order one: \f$ K_1(x) \f$.
 * Employs rational approximations and asymptotic expansions to maintain precision across scales.
 * 
 * @see [Numerical Recipes in C](https://ui.adsabs.harvard.edu/abs/1992nrca.book.....P/abstract)
*
 * @param xbess The input value \f$ x \f$.
 * 
 * @return The value of \f$ K_1(x) \f$.
 */
__host__ __device__ double bessk1(double xbess);

/**
 * @brief Modified Bessel function of the second kind, order two: \f$ K_2(x) \f$.
 * 
 * Computed using the recurrence relation: \f$ K_{n+1}(x) = K_{n-1}(x) + \frac{2n}{x}K_n(x) \f$,
 * starting from the values of \f$ K_0 \f$ and \f$ K_1 \f$.
 * 
 * @see [Numerical Recipes in C](https://ui.adsabs.harvard.edu/abs/1992nrca.book.....P/abstract)
 *
 * @param xbess The input value \f$ x \f$.
 * 
 * @return The value of \f$ K_2(x) \f$.
 */
__host__ __device__ double bessk2(double xbess);
#endif
