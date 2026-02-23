/*
 * GPUmonty - compton.h
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
Declaration of the functions in compton.cu file
*/

#ifndef COMPTON_H
#define COMPTON_H


/**
 * @brief Computes the scattered photon 4-momentum after a collision with an electron.
 * This high-level routine simulates the physical interaction between a photon and a
 * specific electron. It handles the transition between classical Thomson scattering
 * and quantum Klein-Nishina scattering based on the photon's energy in the electron 
 * rest frame.
 * 
 * **Routine**:
 * 1. **Frame Shift**: The incoming photon 4-momentum \f$ k^\mu \f$ is boosted into 
 * the electron's rest frame.
 * 2. **Regime Selection**: 
 * - If \f$ E'_{photon} > 10^{-4} m_e c^2 \f$: Uses **Klein-Nishina** sampling for 
 * energy loss and angular deflection.
 * - If \f$ E'_{photon} \le 10^{-4} m_e c^2 \f$: Uses **Thomson** sampling 
 * (elastic scattering).
 * 3. **Angular Sampling**: Constructs a local coordinate system aligned with the 
 * incident photon to sample the scattering angle \f$ \theta \f$ and azimuthal 
 * angle \f$ \phi \f$.
 * 4. **Inverse Transformation**: The resulting scattered momentum is boosted back 
 * from the electron rest frame to the laboratory/fluid frame.
 * 
 * @param k Incoming photon 4-momentum in the laboratory frame.
 * @param p 4-momentum of the scattering electron.
 * @param kp [out] Scattered photon 4-momentum in the laboratory frame.
 * @param localState Pointer to the curand RNG state for the current thread.
 * 
 * @return void
 */
__device__ void sample_scattered_photon(double k[4], double p[4], double kp[4], curandState * localState);

/**
 * @brief Performs a general Lorentz boost into the frame of a given 4-velocity.
 *
 * Transforms a 4-vector \f$ v \f$ from the laboratory frame into the frame 
 * moving with 4-velocity \f$ u \f$. The implementation uses the general 
 * Lorentz transformation matrix to handle boosts in arbitrary directions.
 *
 * @param v Input 4-vector in the laboratory frame.
 * @param u 4-velocity of the frame to boost into (where \f$ u^0 = \gamma \f$).
 * @param vp Resulting boosted 4-vector in the co-moving frame.
 * 
 * @return void
 */
__device__ void boost(double v[4], double u[4], double vp[4]);

/**
 * @brief Samples the scattering angle cosine for Thomson scattering.
 * 
 * Uses rejection sampling to pick the cosine of the scattering angle \f$ \mu = \cos\theta \f$ 
 * according to the Thomson differential cross-section distribution:
 * \f[ \frac{dP}{d\mu} = \frac{3}{8}(1 + \mu^2) \f]
 * 
 * @param localState Pointer to the curand RNG state for the current thread.
 * 
 * @return The sampled cosine of the scattering angle \f$ \mu \f$.
 */
__device__ double sample_thomson(curandState * localState);

/**
 * @brief Samples the post-scattering photon energy using the Klein-Nishina differential cross-section.
 * 
 * This function uses a rejection sampling algorithm to determine the scattered photon frequency 
 * \f$ k' \f$ in the electron rest frame. The sampling occurs within the kinematically allowed range:
 * \f[ \frac{k}{1+2k} \le k' \le k \f]
 * where the lower bound corresponds to a head-on collision (\f$ \theta = \pi \f$) and the upper 
 * bound corresponds to no deflection (\f$ \theta = 0 \f$).
 *
 * @param k0 Dimensionless photon frequency in the electron rest frame before scattering.
 * @param localState Pointer to the curand RNG state for the current thread.
 * 
 * @return The sampled dimensionless photon frequency after scattering \f$ k' \f$.
 */
__device__ double sample_klein_nishina(double k0, curandState * localState);

/**
 * @brief Computes the differential Klein-Nishina cross-section for a given frequency shift.
 *
 * This function evaluates the probability density of a photon with dimensionless 
 * energy \f$ a \f$ scattering to energy \f$ a' \f$. It is used as the target 
 * distribution for the rejection sampling in the Klein-Nishina scattering routine.
 *
 * The implementation calculates:
 * \f[ \frac{1}{a^2} \left( \frac{a}{a'} + \frac{a'}{a} - \sin^2 \theta \right) \f]
 * where the scattering angle \f$ \theta \f$ is implicitly determined by the 
 * Compton formula: \f$ \cos \theta = 1 + \frac{1}{a} - \frac{1}{a'} \f$.
 *
 * @param a Initial dimensionless photon energy (\f$ h\nu / m_e c^2 \f$).
 * @param ap Scattered dimensionless photon energy (\f$ h\nu' / m_e c^2 \f$).
 * 
 * @return The value of the differential cross-section (unnormalized).
 */
__device__ double klein_nishina(const double a, const double ap);


/**
 * @brief Samples the 4-momentum of a scattering electron from a thermal distribution.
 *
 * Implements the rejection-sampling scheme described by Canfield (1987) and Wong (2022).
 * The routine picks an electron velocity and orientation relative to the incoming 
 * photon, weighted by the frame-dependent Klein-Nishina cross-section.
 *
 * **Routine**:
 * 1. Sample \f$ \gamma_e \f$ and \f$ \beta_e \f$ from a Maxwell-Jüttner distribution.
 * 2. Sample the relative angle cosine \f$ \mu \f$ between the photon and electron.
 * 3. Calculate the rest-frame photon energy \f$ K \f$.
 * 4. Accept the candidate electron if a random variable is less than \f$ \sigma_{KN}(K)/\sigma_T \f$.
 * 5. Construct a local orthonormal basis aligned with the photon's direction to 
 * assign the electron's spatial momentum components.
 *
 * @param k Incoming photon 4-momentum in the local tetrad frame.
 * @param p [out] Sampled electron 4-momentum in the local tetrad frame.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param localState Pointer to the curand RNG state.
 * 
 * @return void
 */
__device__ void sample_electron_distr_p(double k[4], double p[4], double Thetae, curandState * localState);

/**
 * @brief Samples the electron Lorentz factor and velocity from a thermal distribution.
 *
 * This function determines the energy state of a scattering electron by sampling 
 * a transformed energy variable \f$ y \f$ and converting it into the Lorentz 
 * factor \f$ \gamma_e \f$ and the dimensionless velocity \f$ \beta_e \f$.
 *
 * The sampling is based on the Maxwell-Jüttner distribution. The transformation 
 * used is:
 * \f[ \gamma_e = y^2 \Theta_e + 1 \f].
 *
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param gamma_e [out] Pointer to store the sampled Lorentz factor \f$ \gamma \f$.
 * @param beta_e [out] Pointer to store the sampled dimensionless velocity \f$ \beta = v/c \f$.
 * @param localState Pointer to the curand RNG state for the current thread.
 * 
 * @return void
 */
__device__ void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e, curandState * localState);

/**
 * @brief Samples the auxiliary energy variable \f$ y \f$ for the Maxwell-Jüttner distribution.
 * * This function implements the composition-rejection algorithm described by 
 * [Canfield et al. (1987)](https://ui.adsabs.harvard.edu/abs/1987ApJ...323..565C/abstract) 
 * to efficiently sample electron energies. It decomposes the distribution into a weighted sum of four components, each sampled from a 
 * Chi-Square distribution with varying degrees of freedom.
 * 
 * * **Routine**:
 * 1. **Weights**: Calculates weights \f$ \pi_3, \pi_4, \pi_5, \pi_6 \f$ based on 
 * the dimensionless temperature \f$ \Theta_e \f$.
 * 2. **Composition**: Randomly selects one of the four components based on these 
 * weights using a uniform random variable.
 * 3. **Sampling**: Samples \f$ x \f$ from a \f$ \chi^2 \f$ distribution with 
 * 3, 4, 5, or 6 degrees of freedom.
 * 4. **Transformation**: Converts \f$ x \f$ to the auxiliary variable \f$ y = \sqrt{x/2} \f$.
 * 5. **Rejection**: Applies a final rejection step to correct the shape of the 
 * approximated distribution using the probability:
 * \f[ P = \frac{\sqrt{1 + \frac{1}{2} \Theta_e y^2}}{1 + y \sqrt{\frac{\Theta_e}{2}}} \f]
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param localState Pointer to the curand RNG state for the current thread.
 * 
 * @return The sampled auxiliary variable \f$ y \f$, where \f$ \gamma_e = y^2 \Theta_e + 1 \f$.
 */
__device__ double sample_y_distr(const double Thetae, curandState * localState);


/**
 * @brief Samples the cosine of the angle between the photon and the electron.
 * This function performs an inversion sampling of the relative orientation angle \f$ \mu = \cos\alpha \f$ 
 * between the incoming photon and the scattering electron. In the Monte Carlo scattering 
 * scheme, the probability of a collision is proportional to the relative velocity factor:
 * \f[ P(\mu) \propto (1 - \beta_e \mu) \f]
 * where \f$ \mu \in [-1, 1] \f$. 
 * * **Mathematical Derivation**:
 * The Cumulative Distribution Function (CDF) is obtained by integrating the probability:
 * \f[ R = \frac{\int_{-1}^{\mu} (1 - \beta_e x) dx}{\int_{-1}^{1} (1 - \beta_e x) dx} \f]
 * Solving for \f$ \mu \f$ via the quadratic formula yields the implementation:
 * \f[ \mu = \frac{1 - \sqrt{(1 + \beta_e)^2 - 4 \beta_e R}}{\beta_e} \f]
 * 
 * @param beta_e Electron velocity in units of the speed of light (\f$ v/c \f$).
 * @param random A uniform random number in the range [0, 1].
 * 
 * @return The sampled cosine of the relative angle \f$ \mu \f$.
 */
__device__ double sample_mu_distr(const double beta_e, double random);

/**
 * @brief Manages the Compton scattering process for a superphoton on the GPU.
 * This function handles the full scattering pipeline for a single photon indexed in a 
 * Structure of Arrays (SOA). It transforms the photon into the local fluid frame, 
 * samples a scattering electron, executes the scattering event, and updates the 
 * photon's coordinate-frame properties.
 *
 *  **Routine**:
 * - **Tetrad Construction**: Builds a local orthonormal tetrad using the fluid velocity \f$ u^\mu \f$ and magnetic field \f$ b^\mu \f$.
 * - **Frame Transformation**: Rotates the photon 4-momentum \f$ k^\mu \f$ from the 
 * coordinate basis to the local tetrad basis.
 * - **Electron Sampling**: Select a specific electron momentum from the Maxwell-Jüttner distribution.
 * - **Scattering**: Executes the scattering selecting the scattered photon momentum.
 * - **Coordinate Recovery**: Transforms the scattered momentum back to the global coordinate frame and updates the photon's scattering history and energy.
 * 
 * @param ph Input photon Structure of Arrays (SOA).
 * @param php Output (scattered) photon Structure of Arrays (SOA).
 * @param Ne Local electron number density.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param B Magnetic field strength in cgs.
 * @param Ucon Fluid 4-velocity \f$ u^\mu \f$.
 * @param Bcon Magnetic field 4-vector \f$ b^\mu \f$.
 * @param Gcov Covariant metric tensor \f$ g_{\mu\nu} \f$.
 * @param localState Pointer to the curand RNG state for this thread.
 * @param photon_index The index of the photon within the SOA.
 * 
 * @return void
 */
__device__ void scatter_super_photon(struct of_photonSOA ph, struct of_photonSOA php,double Ne, double Thetae, double B, double Ucon[NDIM], double Bcon[NDIM], double Gcov[NDIM][NDIM], curandState * localState, unsigned long long photon_index);
#endif