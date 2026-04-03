/*
 * GPUmonty - jnu_mixed.h
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
Declaration of the functions in the jnu_mixed.cu file
*/

#ifndef JNU_MIXED_H
#define JNU_MIXED_H


/**
 * @brief Kappa electron distribution function for synchrotron emissivity calculations.
 * 
 * @param nu Photon frequency in the plasma frame.
 * @param Ne Electron number density (\f$n_{\rm e}\f$).
 * @param Thetae Dimensionless electron temperature (\f$\Theta_{\rm e}\f$).
 * @param B Magnetic field strength.
 * @param theta Angle between the magnetic field and the wave vector (\f$\theta\f$).
 * 
 * @note Equation taken from \f$\kappa\f$-monty [Davelaar et al.(2023)](https://arxiv.org/pdf/2303.15522)
 * 
 * @return The value of the synchrotron emissivity for a kappa distribution of electrons.
 */
__device__ double jnu_synch_nonthermal_kappa(double nu, double Ne, double Thetae, double B,double theta);

/**
 * @brief Power-law electron distribution function for synchrotron emissivity calculations.
 * 
 * @param nu Photon frequency in the plasma frame.
 * @param Ne Electron number density (\f$n_{\rm e}\f$).
 * @param Thetae Dimensionless electron temperature (\f$\Theta_{\rm e}\f$).
 * @param B Magnetic field strength.
 * @param theta Angle between the magnetic field and the wave vector (\f$\theta\f$).
 * 
 * @note Equation taken from \f$\kappa\f$-monty [Davelaar et al.(2023)](https://arxiv.org/pdf/2303.15522), [Pandya et al. 2016](https://iopscience.iop.org/article/10.3847/0004-637X/822/1/34) 
 */
__device__ double jnu_synch_nonthermal_powerlaw(double nu, double Ne, double Thetae, double B, double theta);
/**
 * @brief Calculates the thermal synchrotron emissivity \f$j_\nu(\nu, \theta)\f$ following [Leung et al.(2011)](https://iopscience.iop.org/article/10.1088/0004-637X/737/1/21/pdf).
 * 
 * This function implements the analytical fit for the emissivity of a thermal distribution of electrons:
 * \f$ j_\nu(\nu, \theta) = \frac{\sqrt{2}\pi e^2 n_e \nu_s}{3 c K_2(1/\Theta_e)} (X^{1/2} + 2^{11/12} X^{1/6})^2 \exp(-X^{1/3}) \f$
 * 
 * @param nu Photon frequency in the plasma frame.
 * @param Ne Electron number density (\f$n_{\rm e}\f$).
 * @param Thetae Dimensionless electron temperature (\f$\Theta_{\rm e}\f$).
 * @param B Magnetic field strength.
 * @param theta Angle between the magnetic field and the wave vector (\f$\theta\f$).
 * @param K2 Precomputed value of the modified Bessel function \f$ K_2(1/\Theta_e) \f$ for efficiency.
 * @return The local emissivity \f$ j_\nu \f$ in CGS units.
 */
__host__ __device__ double jnu_synch(const double nu, const double Ne, const double Thetae, const double B,
    const double theta, const double K2);

/**
 * @brief Calculates the angle-integrated thermal synchrotron emissivity \f$ J_\nu \f$ in CGS units.
 * 
 * This function evaluates the total energy emitted per unit time, per unit volume, and per unit frequency, 
 * integrated over all solid angles \f$ \int j_\nu d\Omega \f$. 
 * 
 * * Implements:
 * \f$ J_\nu = \frac{\sqrt{2} e^3 n_e B \Theta_e^2}{27 m_e c^2 K_2(1/\Theta_e)} F(\Theta_e, B, \nu) \f$
 * 
 * @param Ne Electron number density \f$ n_e \f$.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_{\rm e} \f$.
 * @param Bmag Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param K2 Precomputed value of the modified Bessel function \f$ K_2(1/\Theta_e) \f$ for efficiency.
 * @return The integrated emissivity in \f$ \text{erg} \cdot \text{s}^{-1} \cdot \text{cm}^{-3} \cdot \text{Hz}^{-1} \f$.
 */
__host__ __device__ double int_jnu_thermal_synch(double Ne, double Thetae, double Bmag, double nu, double K2);

/**
 * @brief Provides the integrand for calculating the angle-integrated thermal synchrotron emissivity.
 * 
 * This function is used by the GSL integrator to evaluate the integral over solid angles 
 * \f$ \int j_\nu \sin\theta d\theta \f$. 
 * 
 * The integrand is given by: 
 * \f[ \sin^2\theta \left( (K/\sin\theta)^{1/2} + 2^{11/12}(K/\sin\theta)^{1/6} \right)^2 \exp\left[-(K/\sin\theta)^{1/3}\right] \f]

 * @param th The angle \f$ \theta \f$ between the magnetic field and the photon wave vector.
 * @param params A pointer to the dimensionless frequency parameter \f$ K \f$.
 * @return The value of the integrand at the given angle.
 */
double jnu_integrand(double th, void *params);


/**
 * @brief Initializes lookup tables for the angle-integrated emissivity function \f$ F \f$ and the modified Bessel function \f$ K_2 \f$.
 * 
 * Precomputes \f$ F(K) \f$ by integrating the angular dependence of synchrotron emission over the solid angle using GSL integration and stores 
 * the second-order modified Bessel function \f$ K_2(1/\Theta_e) \f$ in log-space for rapid retrieval during the simulation.
 * 
 * The integration goes as:
 * 
 * \f$F[k] = \ln \left( 4\pi \int_{0}^{\pi/2} \sin^2\theta \left[ \sqrt{\frac{K}{\sin\theta}} + 2^{11/12} \left( \frac{K}{\sin\theta} \right)^{1/6} \right]^2 \exp\left[ -\left( \frac{K}{\sin\theta} \right)^{1/3} \right] d\theta \right)\f$
 * 
 * @return void
 */
__host__ void init_emiss_tables(void);


/** 
 * @brief Evaluation of the modified Bessel function of the second kind, order 2, \f$ K_2(1/\Theta_e) \f$.
 * 
 * This function retrieves the value of \f$ K_2(1/\Theta_e) \f$ from precomputed lookup tables for efficient computation during simulations.
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * 
 * @note for large \f$ \Theta_e \f$, we can use the asymptotic expansion \f$ K_2(1/\Theta_e) \approx 2 \Theta_e^2 \f$.
 * @note TMAX and THETA_MIN define the valid range for \f$ \Theta_e \f$ in the lookup tables.
 * 
 * @return The value of \f$ K_2(1/\Theta_e) \f$.

 */
__host__ __device__ double K2_eval(const double Thetae);

/**
 * @brief Evaluates the frequency-dependent component \f$ F(K) \f$ of the angle-integrated emissivity.
 * This function calculates the dimensionless frequency parameter \f$ K \f$ and retrieves the 
 * corresponding value of the precomputed integral \f$ \int j_\nu d\Omega \f$.
 * 
 * * @details 
 * - If \f$ K > K_{\rm MAX} \f$: Returns 0 due to the exponential decay of the synchrotron spectrum.
 * - If \f$ K < K_{\rm MIN} \f$: Uses a power-law approximation based on the asymptotic behavior of the emissivity at low frequencies.
 * 
 * - Otherwise: Performs a linear interpolation using the precomputed `F` table.
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param Bmag Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param ACCZONE Integer flag indicating whether the evaluation is for an acceleration zone (1) or not (0), which may affect the choice of approximation.
 * 
 * @return The value of the integrated emissivity function \f$ F \f$.
 */
__host__ __device__ double F_eval(const double Thetae, const double Bmag, const double nu, int ACCZONE);

/** 
 * @brief Performs linear interpolation on the precomputed emissivity function for thermal synchrotron \f$ F(K) \f$.
 * 
 * @param K Dimensionless frequency parameter \f$ K \f$.
 *
 * @return The interpolated value of \f$ F(K) \f$.
 */
__host__ __device__ double linear_interp_F_th(const double K);



/** 
 * @brief Performs linear interpolation on the precomputed modified Bessel function \f$ K_2(1/\Theta_e) \f$.
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * 
 * @return The interpolated value of \f$ K_2(1/\Theta_e) \f$.
 */
__host__ __device__ double linear_interp_K2(const double Thetae);

/**
 * @brief Calculates the thermal bremsstrahlung emissivity \f$ j_\nu \f$ for a Maxwellian distribution of electrons.
 * 
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param Ne Electron density.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * 
 * @note This method is from Method from Rybicki & Lightman, ultimately from Novikov & Thorne.
 * 
 * @return The value of the bremsstrahlung emissivity \f$ j_\nu \f$.
 */
__host__ __device__ double jnu_bremss(const double nu, const double Ne, const double Thetae);

/**
 * @brief Calculates the total emissivity \f$ j_\nu \f$ by summing contributions from synchrotron and bremsstrahlung processes.
 * 
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param Ne Electron density.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param B Magnetic field strength \f$ B \f$.
 * @param theta Angle between the magnetic field and the line of sight.
 * @param K2 Precomputed value of the modified Bessel function \f$ K_2(1/\Theta_e) \f$ for efficiency.
 * 
 * @return The value of the total emissivity \f$ j_\nu \f$.
 */
__device__  double jnu_total(const double nu, const double Ne, const double Thetae, const double B, const double theta, const double K2);

/**
 * @brief Calculates the angle-integrated thermal bremsstrahlung emissivity \f$ J_\nu \f$ for a Maxwellian distribution of electrons.
 * 
 * @param Ne Electron density.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * 
 * @return The value of the angle-integrated bremsstrahlung emissivity \f$ J_\nu \f$.
 */
__host__ __device__ double int_jnu_bremss(const double Ne, const double Thetae, const double nu);


/**
 * @brief Calculates the total angle-integrated emissivity \f$ J_\nu \f$ by summing contributions from synchrotron and bremsstrahlung processes.
 * 
 * @param Ne Electron density.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param B Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param K2 Precomputed value of the modified Bessel function \f$ K_2(1/\Theta_e) \f$ for efficiency.
 * 
 * @return The value of the total angle-integrated emissivity \f$ J_\nu \f$.
 */
__host__ __device__ double int_jnu_total(const double Ne, const double Thetae, const double Bmag, const double nu, const double K2);


// Nonthermal declarations

/**
 * @brief Provides the integrand for calculating the angle-integrated synchrotron emissivity for a kappa distribution of electrons.
 * 
 * This function is used by the GSL integrator to evaluate the integral over solid angles
 * 
 * @param th The angle \f$ \theta \f$ between the magnetic field and the photon wave vector.
 * @param params A pointer to the dimensionless frequency parameter \f$ K \f$ for the kappa distribution.
 * 
 * @return The value of the integrand at the given angle for the kappa distribution.
 * 
 */
__host__ double jnu_integrand_kappa(double th, void *params);

/**
 * @brief Initializes lookup tables for the angle-integrated emissivity function \f$ F \f$ for a kappa distribution of electrons.
 * 
 * Precomputes \f$ F(K) \f$ by integrating the angular dependence of synchrotron emission over the solid angle using GSL integration and stores the results in log-space for rapid retrieval during the simulation.
 * 
 * @return void
 */
__host__ void init_emiss_tables_nth(void);

/**
 * @brief Evaluates the frequency-dependent component \f$ F(K) \f$ of the angle-integrated emissivity for a kappa distribution of electrons.
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param Bmag Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * 
 * @return The value of the integrated emissivity function \f$ F \f$ for a kappa distribution of electrons.
 */
__host__ __device__ double F_eval_kappa(double Thetae, double Bmag, double nu);

/**
 * @brief Provides the emissivity \f$ j_\nu \f$ for a powerlaw distribution of electrons.
 * 
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param Ne Electron number density \f$ n_e \f$.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param B Magnetic field strength \f$ B \f$.
 * @param theta Angle between the magnetic field and the wave vector \f$ \theta \f$.
 * 
 * @return The value of the emissivity \f$ j_\nu \f$ for a powerlaw distribution of electrons.
 */
__device__ double jnu_synch_nonthermal_powerlaw(double nu, double Ne, double Thetae, double B,double theta);

/**
 * @brief Provides the emissivity \f$ j_\nu \f$ for a kappa distribution of electrons.
 * 
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * @param Ne Electron number density \f$ n_e \f$.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param B Magnetic field strength \f$ B \f$.
 * @param theta Angle between the magnetic field and the wave vector \f$ \theta \f$.
 * 
 * @return The value of the emissivity \f$ j_\nu \f$ for a kappa distribution of electrons.

 */
__device__ double jnu_synch_nonthermal_kappa(double nu, double Ne, double Thetae, double B, double theta);

/**
 * @brief Calculates the angle-integrated emissivity \f$ J_\nu \f$ for both a kappa and a power-law distribution of electrons based on F_eval function.
 * 
 * @param Ne Electron number density \f$ n_e \f$.
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param Bmag Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * 
 * @return The value of the angle-integrated emissivity \f$ J_\nu \f$ for a kappa or power-law distribution of electrons.
 */
__host__ __device__  double int_jnu_nth(double Ne, double Thetae, double Bmag, double nu);



/**
 * @brief Performs linear interpolation on the precomputed emissivity function for non-thermal synchrotron \f$ F(K) \f$.
 * 
 * @param K Dimensionless frequency parameter \f$ K \f$.
 * 
 * @return The interpolated value of \f$ F(K) \f$ for non-thermal synchrotron emission.
 */
__host__ __device__ double linear_interp_F_nth(double K);

/**
 * @brief Evaluates the frequency-dependent component \f$ F(K) \f$ of the angle-integrated emissivity for a powerlaw distribution of electrons.
 * 
 * @param Thetae Dimensionless electron temperature \f$ \Theta_e \f$.
 * @param Bmag Magnetic field strength \f$ B \f$.
 * @param nu Photon frequency \f$ \nu \f$ in the plasma frame.
 * 
 * @return The value of the integrated emissivity function \f$ F \f$ for a powerlaw distribution of electrons.
 */
__host__ __device__ double F_eval_powerlaw(double Thetae, double Bmag, double nu);

/**
 * @brief Provides the integrand for calculating the angle-integrated synchrotron emissivity for a power-law distribution of electrons.
 * 
 * @param th The angle \f$ \theta \f$ between the magnetic field and the photon wave vector.
 * @param params A pointer to the dimensionless frequency parameter \f$ K \f$ for the power-law distribution.
 * 
 * @return The value of the integrand at the given angle for the power-law distribution.
 */
__host__ double jnu_integrand_powerlaw(double th, void *params);
#endif