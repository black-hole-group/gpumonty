/*
 * GPUmonty - main.h
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

#ifndef MAIN_H
#define MAIN_H

/**
 * @brief Entry point for the GPUmonty.
 * * This function manages the high-level execution flow of the application.
 *
 *
 * @param argc Argument count.
 * @param argv Argument vector (parameters and model file paths).
 * @return 0 on successful completion.
 */
__host__ int main(int argc, char *argv[]);

/**
 * @brief Initializes and synchronizes physical units between the Host and the Device.
 *
 * This function defines the transformation scales between "code units" (dimensionless 
 * values used in the GRMHD simulation) and "CGS units" (physical values). The scaling
 * parameters are defined as follows:
 * 
 * - **Mass Scaling** (\f$ M_{\rm unit} \f$): User-defined mass unit to scale density and other mass-related quantities.
 * - **Black Hole Mass** (\f$ M_{\rm BH} \f$): Mass of the central black hole, used to define length and time scales.
 * - **Length scale** (\f$ L_{\rm unit} \f$): Defined as the gravitational radius \f$ R_g = GM_{BH}/c^2 \f$.
 * - **Time scale** (\f$ T_{\rm unit} \f$): Defined as the light-crossing time \f$ R_g/c \f$.
 * - **Density scale** (\f$ \rho_{\rm unit} \f$): Calculated as \f$ M_{\rm unit}/L_{\rm unit}^3 \f$.
 * - **Magnetic Field scale** (\f$ B_{\rm unit} \f$): Scaled such that the magnetic pressure \f$ B^2/4\pi \f$
 * matches the energy density scale \f$ \rho c^2 \f$ in Gaussian units.
 * - **Energy Density Unit** (\f$ U_{\rm unit} \f$): Defined as \f$ \rho_{ \rm unit} c^2 \f$.
 * - **Number Density scale** (\f$ N_{e,\rm unit} \f$): Assumes a fully ionized hydrogen plasma (\f$ m_p + m_e \f$).
 * 
 * @param params The simulation parameters containing the Black Hole mass (\f$M_{BH}\f$) 
 * and the mass scaling unit (\f$M_{\rm unit}\f$).
 * 
 * @return void
 */
__host__ void set_units(Params params);

/**
 * @brief Organizes the main CPU tasks of initialization of the simulation environment and precomputed tables.
 *
 * This function is the primary setup routine for the simulation. It prepares the physical units, 
 * loads the fluid model, initializes the spacetime geometry (metric), and generates the 
 * lookup tables (LUTs).
 *
 * **Routine**:
 * 1. **Set Unit Scaling**: Establishes the relationship between code units and CGS physical units by means of the function `set_units()`.
 * 2. **Data & Geometry**: Loads the GRMHD fluid snapshots and precomputes metric coefficients 
 * across the grid by means of the functions `init_model()` and `init_geometry()`.
 * 3. **Table Generation**: Precomputes LUTs in the following order: Cross section table, emissivity table, weight table and 
 * solid angle integrated emissivity table. This is computed by means of the functions `init_hotcross()`, `init_emiss_tables()`, `init_weight_table()` and `init_nint_table()`.
 *
 * @param args Command line arguments.
 */
__host__ void init_model(char *args[]);

/**
 * @brief Precomputes the spacetime metric components across the simulation grid.
 *
 * This function populates the global geometry array (`geom`) with the covariant metric, 
 * contravariant metric, and the metric determinant.
 *
 *
 * @note This function assumes **axisymmetry**, where the geometry does not change with the 
 * azimuthal angle \f$\phi\f$ (represented by index \f$k\f$), therefore geom is a 2D array.
 *
 * @return void 
 */
__host__ void init_geometry();

/**
 * @brief Processes energy and angled binned simulation data to generate and save the final spectrum.
 *
 * This function converts the raw energy and photon counts accumulated in the 
 * spectral grid into physical units. It calculates the SED across different inclination angles, determines 
 * the average optical depths, and computes global simulation diagnostics such 
 * as total luminosity and accretion efficiency.
 *
 *
 * @param N_superph_made Total number of superphotons generated during the run.
 * @param spect 2D array containing the accumulated spectral data (Energy/Theta bins).
 * @param filename Name of the output file (saved in the `./output/` directory).
 */
__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum ***spect, const char * filename);
#endif