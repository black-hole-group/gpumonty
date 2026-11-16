/*
 * GPUmonty - radiation.h
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
 * @param kappa The shape parameter of the kappa distribution, which controls the slope of the non-thermal tail. This allows for flexibility in modeling different electron energy distributions.
 *
 * @return The invariant emissivity \f$ j_\nu / \nu^2 \f$.
 */
__device__ double jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa);

/**
 * @brief Retrieves the kappa parameter for the non-thermal electron distribution at a given position for VARIABLE_KAPPA model using Ball+2016 model.
 * 
 * @param X The 4-position of the photon.
 * @param d_p Pointer to the device memory containing the primitive variables of the iharm model, including the plasma beta and magnetization that are used to calculate kappa.
 * 
 * @return The kappa parameter value at the given position. If VARIABLE_KAPPA is not defined, it returns a constant KAPPA_SYNCH.
 */
 __device__ double get_model_kappa(double X[NDIM] 
    #ifndef SPHERE_TEST
	, double * d_p
	#endif
);


/**
 * @brief Retrieves the kappa parameter for the non-thermal electron distribution at a given cell-center for VARIABLE_KAPPA model using Ball+2016 model.
 * 
 * @param i The radial index of the cell.
 * @param j The polar index of the cell.
 * @param k The azimuthal index of the cell.
 * @param d_p Pointer to the device memory containing the primitive variables of the iharm model, including the plasma beta and magnetization that are used to calculate kappa.
 * 
 * @return The kappa parameter value at the given cell. If VARIABLE_KAPPA is not defined, it returns a constant KAPPA_SYNCH.
 */
__host__ __device__ double get_model_kappa_ijk(const int i, const int j, const int k, const double * d_p);
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
 * This function determines the scattering opacity \f$\kappa_{\rm es}\f$ by looking up the 
 * total Compton cross-section for a given photon frequency and electron temperature. 
 * It assumes a pure hydrogen composition to convert the cross-section (area) 
 * into opacity (area per unit mass).
 *
 * The opacity is calculated as:
 * \[ \kappa_{es} = \frac{\sigma_{Compton}(E_g, \Theta_{\rm e})}{m_p} \]
 * where \f$ E_{\rm g} = \frac{h\nu}{m_{\rm e} c^2} \f$ is the dimensionless photon energy and \f$ m_{\rm p} \f$ 
 * is the proton mass.
 *
 * @param nu Photon frequency in the fluid frame (Hz).
 * @param Thetae Dimensionless electron temperature \f$ \Theta_{\rm e} \f$.
 * @param d_table_ptr Pointer to the GPU memory containing the precomputed cross-section lookup table.
 * @param kappa The shape parameter of the kappa distribution, which controls the slope of the non-thermal tail. This allows for flexibility in modeling different electron energy distributions.
 * 
 * @return The electron scattering opacity in \f$ \text{cm}^2/\text{g} \f$.
 */
__device__ double kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr, const double kappa);

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
__device__ double alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double kappa, const double * __restrict__ d_table_ptr);


/**
 * @brief Calculates the Lorentz invariant absorption coefficient via Kirchhoff's Law.
 *
 * This function determines the invariant absorption coefficient \f$\nu \alpha_{\rm a}\f$ 
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
 * @return The invariant absorption coefficient \f$\nu \alpha_{a}\f$ [units: Hz \f$ \rm{cm}^{-1} \f$].
 *
 * @note A small epsilon (\f$ 10^{-100} \f$) is added to the denominator to prevent division by zero.
 */
__device__ double alpha_inv_abs_thermal(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa);


/**
 * @brief Wrapper to calculate the total invariant absorption coefficient, including both thermal and non-thermal contributions.
 * 
 * @param nu Photon frequency in the fluid frame (Hz).
 * @param Thetae Dimensionless electron temperature \f$\Theta_\rm{e} \f$.
 * @param Ne Electron number density \f$N_e\f$ (cm\f$^{-3}\f$).
 * @param B Magnetic field strength (Gauss).
 * @param theta Pitch angle between photon and magnetic field.
 * @param kappa The shape parameter of the kappa distribution, which controls the slope of the non-thermal tail. This allows for flexibility in modeling different electron energy distributions.
 * @return The total invariant absorption coefficient \f$\nu \alpha_{a}\f$ [units: Hz \f$ \rm{cm}^{-1} \f$], including contributions from both thermal and non-thermal processes.
 */
__device__ double alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa);

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


/**
 * @brief Computes the synchrotron absorption coefficient for a kappa electron distribution.
 *
 * This function calculates the synchrotron absorption based on the analytical 
 * approximations provided in Pandya et al. (2016).
 *
 * @param nu      The frequency of the incident photon [Hz].
 * @param Ne      The electron number density [cm^-3].
 * @param Thetae  The dimensionless electron temperature (kT_e / m_e c^2).
 * @param B       The local magnetic field strength [Gauss].
 * @param theta   The pitch angle between the photon wavevector and the magnetic field [radians].
 * @param kappa  The shape parameter of the kappa distribution, which controls the slope of the non-thermal tail. This allows for flexibility in modeling different electron energy distributions.
 *
 * @return The synchrotron absorption coefficient. Returns 0.0 if the pitch angle 
 * is practically zero, the temperature is below THETAE_MIN, or if the 
 * normalized frequency (X_kappa) exceeds physically relevant bounds.
 */
__device__ double anu_synch_kappa(double nu, double Ne, double Thetae, double B, double theta, const double kappa);

/**
 * @brief Computes the synchrotron absorption coefficient for a power-law electron distribution.
 *
 * This function calculates the synchrotron absorption for a non-thermal, 
 * pure power-law electron energy distribution. 
 * 
 *
 * @param nu      The frequency of the incident photon [Hz].
 * @param Ne      The electron number density [cm^-3].
 * @param B       The local magnetic field strength [Gauss].
 * @param theta   The pitch angle between the photon wavevector and the magnetic field [radians].
 *
 * @return The synchrotron absorption coefficient. Returns 0.0 if the pitch 
 * angle is practically zero to prevent division by zero or NaN values.
 */
__device__ double anu_synch_powerlaw(double nu, double Ne, double B, double theta);
#endif