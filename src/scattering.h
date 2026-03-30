

/*
 * GPUmonty - scattering.h
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


#ifndef SCATTERING_H
#define SCATTERING_H

/**
 * @brief Control the flow of scattering process. It has the loop that will launch the kernel for every scattering layer.
 * @param num_scat_phs Number of scattering superphotons for each layer.
 * @param scat_ofphoton The scattering superphoton SOA that will be used in the scattering process.
 * @param d_spect device pointer to the spectrum structure used to save the spectrum.
 * @param instant_photon_number The number of superphotons in the current batch in the current batch.
 * @param max_block_number The maximum number of blocks that can be launched in the scattering kernel.
 * @param d_table_ptr Device pointer to the table.
 * @param d_p Device pointer to the plasma properties
 * @param dPTableTexObj The texture object for the plasma table values.
 * @param local_stream The CUDA stream to be used for the scattering kernel launches.
 * @returns void
 */
void scattering_flow_control(unsigned long long num_scat_phs[MAX_LAYER_SCA], struct of_photonSOA * scat_ofphoton, struct of_spectrum *d_spect, unsigned long long instant_photon_number, int max_block_number, double *d_table_ptr, double *d_p, cudaTextureObject_t dPTableTexObj, cudaStream_t local_stream);
#endif