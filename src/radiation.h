/*
Declaration of the functions in radiation.cu file
*/

#ifndef RADIATION_H
#define RADIATION_H

/**
 * @brief Calculates the invariant emissivity for synchrotron radiation.
 *
 * This function computes the local synchrotron emissivity \f$ j_\nu \f$ through means of the function `jnu_synch` and converts 
 * it to the Lorentz-invariant form \f$ j_\nu / \nu^2 \f$. 
 *
 * - Emissivity \f$ j_\nu \f$ is determined by the local plasma properties (\f$ N_e, \Theta_e, B \f$) 
 * and the emission angle \f$ \theta \f$ relative to the magnetic field.
 *
 * @param nu Photon frequency \f$ \nu \f$ in the fluid frame.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e = k_B T_e / m_e c^2 \f$.
 * @param Ne Electron number density \f$ N_e \f$ in cgs.
 * @param B Magnetic field strength \f$ B \f$ in Gauss.
 * @param theta Pitch angle \f$ \theta \f$ between the photon wavevector and the magnetic field.
 * @param besselTexObj CUDA texture object used for fast lookup of modified Bessel functions.
 *
 * @return The invariant emissivity \f$ j_\nu / \nu^2 \f$.
 */
__device__ double jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta, cudaTextureObject_t besselTexObj);

/**
 * @brief Calculates the invariant Planck source function for a thermal distribution.
 *
 * This function computes the frequency-dependent Planck function \f$ B_\nu(T) \f$ 
 * and returns it in its Lorentz-invariant form: \f$ B_\nu / \nu^3 \f$. This quantity 
 * represents the invariant intensity of a blackbody in thermal equilibrium.
 *
 * @note To prevent precision loss and division by zero at low frequencies, the function uses a 4th-order Taylor expansion of \f$ e^x - 1 \f$ when 
 * \f$ x < 10^{-3} \f$:
 * \f[ e^x - 1 \approx x + \frac{x^2}{2} + \frac{x^3}{6} + \frac{x^4}{24} \f]
 *
 * The invariant source function is defined as:
 * \f[ \mathcal{B} = \frac{B_\nu}{\nu^3} = \frac{2h}{c^2} \frac{1}{\exp\left(\frac{h\nu}{k_B T}\right) - 1} \f]
 * where \f$ k_B T = m_e c^2 \Theta_e \f$.
 *
 * @param nu Photon frequency \f$ \nu \f$ in the fluid frame.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * 
 * @return The invariant Planck function \f$ B_\nu / \nu^3 \f$ in CGS units.
 */
__device__ double Bnu_inv(const double nu, const double Thetae);

/**
 * @brief Calculates the electron scattering opacity in CGS units.
 *
 * This function determines the scattering opacity $\kappa_{es}$ by looking up the 
 * total Compton cross-section for a given photon frequency and electron temperature. 
 * It assumes a pure hydrogen composition to convert the cross-section (area) 
 * into opacity (area per unit mass).
 *
 * The opacity is calculated as:
 * \[ \kappa_{es} = \frac{\sigma_{Compton}(E_g, \Theta_e)}{m_p} \]
 * where \f$ E_{\rm g} = \frac{h\nu}{m_{\rm e} c^2} \f$ is the dimensionless photon energy and \f$ m_{\rm p} \f$ 
 * is the proton mass.
 *
 * @param nu Photon frequency in the fluid frame (Hz).
 * @param Thetae Dimensionless electron temperature \f$ \Theta_\rm{e} \f$.
 * @param d_table_ptr Pointer to the GPU memory containing the precomputed cross-section lookup table.
 * 
 * @return The electron scattering opacity in \f$ \text{cm}^2/\text{g} \f$.
 */
__device__ double kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr);

/**
 * @brief Computes the photon frequency in the local fluid frame (Hz).
 *
 * This function projects the photon 4-wavevector \f$K^\mu\f$ onto the covariant 
 * fluid 4-velocity \f$U_\mu\f$ to determine the energy as measured by a co-moving 
 * observer.
 *
 * The energy of a particle 
 * with 4-momentum \f$p^\mu\f$ measured by an observer with 4-velocity \f$u^\mu\f$ 
 * is given by the scalar \f$E = -p^\mu u_\mu\f$.
 * 
 * @note The energy is initially calculated in 
 * dimensionless electron rest-mass units (\f$m_e c^2\f$). This is converted to 
 * physical frequency using the relation \f$\nu = E \cdot (m_e c^2 / h)\f$.
 *
 * @param X The 4-position of the photon (primarily used for diagnostic logging).
 * @param K The photon 4-wavevector \f$K^\mu\f$.
 * @param Ucov The covariant fluid 4-velocity \f$U_\mu\f$.
 *
 * @return The physical frequency \f$\nu\f$ in Hz.
 *
 */
__device__ double get_fluid_nu(const double X[NDIM] , const double K[NDIM] , const double Ucov[NDIM]);

/**
 * @brief Calculates the Lorentz invariant scattering coefficient.
 *
 * This function converts the scattering opacity (cross-section per unit mass) 
 * into the invariant scattering coefficient \f$\nu \alpha_{s}\f$. 
 *
 * \[ \text{Result} = \nu \cdot \alpha_{s} = \nu \cdot (\rho \cdot \kappa_{es}) \]
 * where \f$\rho \approx N_e m_p\f$ (assuming a pure hydrogen plasma).
 *
 * @param nu Photon frequency in the fluid frame (Hz).
 * @param Thetae Dimensionless electron temperature \f$\Theta_\rm{e} \f$.
 * @param Ne Electron number density \f$N_e\f$ (cm\f$^{-3}\f$).
 * @param d_table_ptr Pointer to the Compton cross-section lookup table.
 * 
 * @return The invariant scattering coefficient \f$\nu \alpha_{s}\f$ [units: Hz \f$ \rm{cm}^{-1} \f$].
 */
__device__ double alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double * __restrict__ d_table_ptr);


/**
 * @brief Calculates the Lorentz invariant absorption coefficient via Kirchhoff's Law.
 *
 * This function determines the invariant absorption coefficient $\nu \alpha_{a}$ 
 * by relating the invariant emissivity to the invariant Planck function (source function).
 *
 * Based on Kirchhoff's Law of Thermal Radiation, the absorption coefficient \f$\alpha_\nu\f$ 
 * is the ratio of emissivity to the Planck function:
 * \[ \alpha_\nu = \frac{j_\nu}{B_\nu} \]
 * In terms of the invariant quantities provided by the internal helper functions:
 * \[ \nu \alpha_\nu = \frac{j_\nu / \nu^2}{B_\nu / \nu^3} \]
 *
 * @param nu Photon frequency in the fluid frame (Hz).
 * @param Thetae Dimensionless electron temperature \f$\Theta_\rm{e} \f$.
 * @param Ne Electron number density \f$N_e\f$ (cm\f$^{-3}\f$).
 * @param B Magnetic field strength (Gauss).
 * @param theta Pitch angle between photon and magnetic field.
 * @param besselTexObj CUDA texture for Bessel function lookups.
 * @return The invariant absorption coefficient \f$\nu \alpha_{a}\f$ [units: Hz \f$ \rm{cm}^{-1} \f$].
 *
 * @note A small epsilon (\f$ 10^{-100} \f$) is added to the denominator to prevent division by zero.
 */
__device__ double alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B, double theta, cudaTextureObject_t besselTexObj);


/**
 * @brief Calculates the pitch angle between the photon wavevector and the magnetic field.
 *
 * This function determines the angle \f$\theta\f$ between the photon's
 * propagation direction and the local magnetic field vector, as measured in the 
 * local fluid frame.
 *
 *
 *
 * @param X Photon 4-position.
 * @param K Photon 4-wavevector \f$K^\mu\f$.
 * @param Ucov Covariant fluid 4-velocity \f$U_\mu\f$.
 * @param Bcov Covariant magnetic field 4-vector \f$B_\mu\f$.
 * @param B Magnetic field strength in Gauss (CGS).
 * 
 * @return The pitch angle \f$\theta\f$ in radians \f$[0, \pi]\f$.
 */
__device__ double get_bk_angle(const double X[NDIM], const double K[NDIM] , const double Ucov[NDIM] , const double Bcov[NDIM], const double B);
#endif