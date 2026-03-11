/*
 * GPUmonty - par.h
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
Declaration of the par.cu functions
*/


/**
* @brief Data type definition for integer parameter reading.
*/
#define TYPE_INT (1)

/**
* @brief Data type definition for double parameter reading.
*/
#define TYPE_DBL (2)

/**
* @brief Data type definition for string parameter reading.
*/
#define TYPE_STR (3)

#ifndef PAR_H
#define PAR_H


/**
 * @brief Initializes simulation parameters with default values and parses command-line arguments.
 * 
 * This function sets up the initial environment for the Monte Carlo simulation.
 * 
 * **Default Parameter Values:**
 * - **seed**: If not specified, the random seed is set to -1, which triggers time-based seeding for randomness.
 * - **scattering**: Enables scattering by default (set to 1).
 * - **bias_guess**: Default bias tuning factor is set to 1.0. BIAS TUNING STILL NOT IMPLEMENTED.
 * - **fitBias**: Default fitting bias factor is set to 1.0. FITTING BIAS STILL NOT IMPLEMENTED.
 * - **fitBiasNs**: Scale input number of superphotons generated in each zone. Default is 1.0 (no scaling). FITTING BIAS STILL NOT IMPLEMENTED.
 * - **tp_over_te**: Sets the default proton-to-electron temperature ratio \f$ T_p/T_e \f$ for the simulation. Default is 3.0.
 * - **beta_crit**: Sets the default critical plasma beta \f$ \beta_{\rm crit} \f$ for R-low/R-high models. Default is 1.0.
 * - **trat_small**: Sets the minimum proton-to-electron temperature ratio \f$ T_p/T_e \f$ for R-low/R-high models. Default is 1.0.
 * - **trat_large**: Sets the maximum proton-to-electron temperature ratio \f$ T_p/T_e \f$ for R-low/R-high models. Default is 10.0.
 * - **Thetae_max**: Sets the maximum electron temperature \f$ \Theta_e \f$ for R-low/R-high models. Default is \f$ 10^{100} \f$.
 * - **Black Hole Mass**: Defaults to \f$ 4.1 \times 10^6 M_\odot \f$ (Sgr A*).
 * 

 * @note R-high and R-low models are only applied to iharm_model simulations.

 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @param params Pointer to the global `Params` structure to be populated.
 * 
 * @return void
 */
__host__ void load_par_from_argv(int argc, char *argv[], Params *params);

/**
 * @brief Parses the simulation parameter file and generates a formatted initialization report.

 * - **File Parsing**: Scans for keywords defined in the Params structure and reads their corresponding values.
 * - **Validation Report**: Outputs a color-coded terminal summary (using ANSI escape codes) to inform the user which parameters were successfully loaded from the file (`[SET]`) and which are falling back to hardcoded defaults (`[MISSING]`).
 * - **Status Flag**: Sets `params->loaded = 1` upon completion to signal that the simulation environment is ready.
 * @param fname Path to the input parameter file.
 * @param params Pointer to the global `Params` structure to be populated.
 * 
 * @return void
 */
__host__ void load_par (const char *fname, Params *params);

/**
 * @brief Reads a single parameter from the input file and assigns it to the provided variable.
 * 
 * This function supports reading integer, double, and string parameters based on the specified type.
 * 
 * @param fname Path to the input parameter file.
 * @param parname Name of the parameter to read.
 * @param var Pointer to the variable where the read value will be stored.
 * @param type Data type of the parameter (TYPE_INT, TYPE_DBL, TYPE_STR).
 * 
 * @return void
 */
__host__ void read_param (const char *line, const char *key, void *val, int type);
#endif