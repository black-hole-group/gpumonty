/*
Declaration of the functions in the hotcross.cu file
*/


#ifndef HOTCROSS_H
#define HOTCROSS_H

/**
 * @brief Initializes the Compton cross-section lookup table.
 * 
 * This function prepares a 2D lookup table for the total Compton cross-section 
 * as a function of dimensionless photon frequency \f$ w \f$ and electron 
 * temperature \f$ \Theta_e \f$. 
 * 
 * **Routine**:
 * 1. Attempts to open the file specified by `HOTCROSS`.
 * 2. If the file is found, it loads the precomputed values into memory.
 * 3. If the file is missing, it triggers a numerical calculation of the table 
 * using `total_compton_cross_num()`.
 * 
 * @note The table is stored in \f$ \log_{10}-\log_{10} \f$ space.
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
 * @param d_table_ptr Restrict-qualified pointer to the device lookup table.
 * 
 * @return The calculated Compton cross-section.
 */
__device__ double total_compton_cross_lkup(double w, double thetae, const double * __restrict__ d_table_ptr);

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
 * 
 * @return The numerically integrated cross-section normalized by \f$ \sigma_T \f$.
 */
__host__ __device__ double total_compton_cross_num(double w, double thetae);

/**
 * @brief Calculates the normalized Maxwell-Jüttner electron distribution function.
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
__host__ __device__ double dNdgammae(double thetae, double gammae);

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
