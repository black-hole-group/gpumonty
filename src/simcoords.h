
/*
 * GPUmonty - simcoords.h
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


#ifndef SIMCOORDS_H
    #define SIMCOORDS_H
    /**
     * @brief Allocate and load the ks_r and ks_h arrays from file, then transfer them to the GPU.
     * This is used in the simcoords method Wong et al (2022) Patoka's paper, section 3.2.3.
     * 
     * @param d_ks_r Device pointer to store the ks_r array on the GPU.
     * @param d_ks_h Device pointer to store the ks_h array on the GPU
     * 
     * @return void
     */
    __host__ void load_simcoord_info_from_file(double * d_ks_r, double * d_ks_h);
    __host__ void ijktoX(int i, int j, int k, double X[NDIM]);
    __host__ __device__ int simcoordijk_to_eks(const int i, const int j, const int k, double eks[NDIM], double * d_ks_r, double *d_ks_h);
    __host__ void initialize_simgrid(size_t interp_n1, size_t interp_n2, double x1i, double x1f, double x2i, double x2f);
    __host__ void finalize_simgrid();

#endif