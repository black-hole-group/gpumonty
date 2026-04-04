/*
 * GPUmonty - kernels.h
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
Declarations of the kernels.cu file functions
*/

#ifndef KERNELS_H
#define KERNELS_H


/**
 * @brief Kernel that will divide the work of generating superphotons among the GPU threads.
 *
 * * Initialize the curand state for each thread based on a unique seed derived from the current time and thread ID.
 * * Calls init_zone to initialize the photon generation for each zone in the simulation grid.
 * * The loop follows a **grid-stride pattern** to ensure all zones are covered. Each thread \f$ n \f$ 
 * processes a sequence of zones defined by:
 \f[ {\rm i_n} = n + k \cdot N_{\text{total}}\f]
 where \f$ {\rm i_n}\f$ is the current zone \f$ \rm i \f$ that thread \f$ \rm n \f$ is analyzing,
 * \f$ n \f$ is the global thread ID, \f$ N_{\text{total}} \f$ is the total number of threads in the grid, 
 * and \f$ k \in \{0, 1, 2, \dots\} \f$.
 * 
 * @param d_geom Pointer to the device geometry structure that carries the metric information \f$ g_{μν}, g^{μν},\ \rm{and}\ \sqrt{- g}  \f$ .
 * @param d_p Pointer to the device array containing the plasma parameters of the simulation.
 * @param generated_photons_arr Pointer to the device array that will store the number of photons generated per zone.
 * @param dnmax_arr Pointer to the device array that will store the maximum value of \f$ \frac{dN}{d\ln\nu} \f$ per zone.
 * @return void
 */
__global__ void generate_photons(const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, unsigned long long * __restrict__  generated_photons_arr, double * __restrict__ dnmax_arr);    



/**
 * @brief Initializes the random number generator states for each GPU thread.
 *
 * @param time Current epoch time used as part of the seed for RNG initialization to ensure variability across runs.
 * @param GPUindex The index of the GPU being used, included in the seed to ensure different RNG states across multiple GPUs.
 * @return void
 */
__global__ void InitializeRNGStates(const time_t time, int GPUindex);
/**
 * @brief Initializes the photon generation parameters for a specific grid zone.
 *
 * - This function retrieves local fluid properties (density, temperature, and magnetic field) 
 * for the zone \f$(i, j, k)\f$ and calculates the expected number of superphotons to 
 * be generated based on the local thermal synchrotron emissivity.
 *
 * The expected number of superphotons \f$ N_{s,i} \f$ in zone \f$ i \f$ is governed by:
 * \f[
 * N_{s,i} = \Delta t \Delta^3 x \sqrt{-g} \int \int d\nu d\Omega \frac{1}{\omega} \frac{j_\nu}{h\nu}
 * \f]
 * where \f$ \sqrt{-g} \f$ is the metric determinant, \f$ j_\nu \f$ is the emissivity, 
 * and \f$ \omega \f$ is the superphoton weight. 
 *
 *
 * The final number of superphotons \f$ n_{2gen} \f$ is determined stochastically to 
 * handle the fractional part of the expected count \f$ n_z \f$:
 * \f[
 * n_{2gen} = 
 * \begin{cases} 
 * \lfloor n_z \rfloor + 1 & \text{if } \text{rand} < (n_z \pmod 1) \\
 * \lfloor n_z \rfloor     & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * @param i Radial grid index.
 * @param j Poloidal grid index.
 * @param k Azimuthal grid index.
 * @param n2gen Pointer to store the integer number of superphotons to be generated.
 * @param dnmax Pointer to store the maximum of the distribution \f$ dN_s/d\ln\nu \f$, 
 * required for subsequent rejection sampling.
 * @param d_geom Pointer to the device geometry structure that carries the metric information \f$ g_{μν}, g^{μν},\ \rm{and}\ \sqrt{- g}  \f$ .
 * @param d_p Pointer to the device simulation plasma properties.
 * @param d_Ns_par Target superphoton count parameter.
 * @param localState Pointer to the curandState for the current thread.
 * @return void
 */
__device__ void init_zone(const int i, const int j, const int k, unsigned long long * __restrict__  n2gen, double * __restrict__ dnmax, const struct of_geom * __restrict__  d_geom, const double * __restrict__ d_p, const int d_Ns_par, curandState  * localState);

/**
 * @brief Samples superphoton properties for the current batch using dynamic workload allocation.
 *
 * - This kernel performs the rejection sampling and initial coordinate assignment for superphotons. 
 * Unlike a standard grid-stride pattern where thread work is predetermined by ID, this kernel 
 * utilizes a **dynamic workload allocation** strategy since some zones may generate significantly more superphotons than others, leading to 
 * extreme load imbalance. To mitigate this, threads fetch work dynamically using an atomic counter. This ensures that as soon as a
 * thread finishes sampling a photon, it immediately pulls the next 
 * available index, keeping GPU execution units saturated regardless of the variation in workload 
 * per zone.
 *
 * - The kernel maps the superphoton to the grid zone that generated it \f$ (i, j, k) \f$ by performing 
 * a binary search on the cumulative distribution array `index_to_ijk`. Once the `zone_index` is 
 * retrieved, it is decomposed into 3D grid coordinates:
 *
 * @param ph_init The Structure of Arrays (SoA) where sampled photon data is stored.
 * @param d_geom Pointer to the device geometry structure that carries the metric information \f$ g_{μν}, g^{μν},\ \rm{and}\ \sqrt{- g}  \f$ .
 * @param d_p Pointer to the device simulation plasma properties.
 * @param generated_photons_arr Array containing the number of photons to be generated per zone.
 * @param dnmax_arr The maximum of the distribution \f$ dN_s/d\ln\nu \f$ for each zone, used for rejection sampling.
 * @param max_partition_ph The maximum number of photons assigned to this batch.
 * @param photons_processed_sofar Offset for the current batch relative to the total simulation count.
 * @param index_to_ijk Cumulative sum array mapping global photon indices to zone indices.
 * @return void
 */
__global__ void sample_photons_batch(struct of_photonSOA ph_init, const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const unsigned long long * __restrict__  generated_photons_arr, const double * __restrict__ dnmax_arr, const int max_partition_ph,  const unsigned long long photons_processed_sofar, const unsigned long long * __restrict__  index_to_ijk, int GPU_id);

/**
 * @brief  Samples the physical properties of a single superphoton within a zone.
 *
 * It uses a series of rejection sampling loops to determine the frequency and angular distribution of 
 * emission, then transforms the resulting momentum vector from the local fluid tetrad into the global coordinate frame.
 *
 * **Frequency Sampling**: The photon frequency \f$ \nu \f$ is sampled using rejection sampling over the range 
 * \f$ [\nu_{min}, \nu_{max}] \f$. The selection is weighted by the local distribution 
 * \f$ dN_s/d\ln\nu \f$:
 *
 * **Angular Distribution**: The polar angle \f$ \theta \f$ (relative to the local magnetic field) is sampled 
 * based on the angular dependence of the synchrotron emissivity \f$ j_\nu(\theta) \f$. 
 * A second rejection sampling loop ensures that:
 * \f[
 * \text{rand} < \frac{j_\nu(\theta)}{j_{\nu, max}}
 * \f]
 * where \f$ j_{\nu, max} \f$ is the emissivity at \f$ \theta = \pi/2 \f$.
 *
 * **Tetrad Transformation**: The momentum vector \f$ K \f$ is initially constructed in a local orthonormal 
 * tetrad frame aligned with the fluid four-velocity \f$ u^\mu \f$ and the magnetic 
 * field \f$ b^\mu \f$. It is then transformed to the coordinate frame using:
 * \f[
 * k^\mu = e^\mu_{(\alpha)} k^{(\alpha)}
 * \f]
 * where \f$ e^\mu_{(\alpha)} \f$ is the tetrad basis built by `make_tetrad`.
 *
 * @param i Radial grid index.
 * @param j Poloidal grid index.
 * @param k Azimuthal grid index.
 * @param dnmax The maximum of the distribution \f$ dN_s/d\ln\nu \f$ for this zone.
 * @param ph The Structure of Arrays (SoA) used for photon storage.
 * @param d_geom Pointer to the device geometry structure that carries the metric information \f$ g_{μν}, g^{μν},\ \rm{and}\ \sqrt{- g}  \f$ .
 * @param d_p Pointer to plasma properties array.
 * @param zone_flag Boolean flag; if true, this is the first photon in this zone for this particular thread and a tetrad must be constructed, otherwise reuse the existing tetrad.
 * @param ph_arr_index The specific index in the SoA to store this photon.
 * @param Econ Tetrad basis (coordinate to orthonormal).
 * @param Ecov Dual tetrad basis (orthonormal to coordinate).
 * @param localState Pointer to the curandState for the current thread.
 * @return void
 */
__device__ void sample_zone_photon(const int i, const int j, const int k, const double dnmax, struct of_photonSOA ph, const struct of_geom * d_geom, const double * d_p, const int zone_flag, const unsigned long long ph_arr_index, double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState * localState, int GPU_id);

/**
 * @brief Assign each superphoton to a thread through dynamic load balancing to be evolved through the geodesic.
 *
 * To handle the variable computational cost of different 
 * geodesics (where some photons escape quickly and others undergo multiple scatterings), 
 * this kernel uses dynamic load balancing Threads fetch a global photon index \f$ n \f$ 
 * dynamically, ensuring that as soon as a thread finishes processing a photon, it immediately
 * pulls the next available index. This keeps all GPU execution units busy regardless of the
 * variation in workload per photon.
 *
 *
 * @note **Progress Monitoring**: Thread with global index 0 acts as a monitor, calculating and 
 * printing the simulation progress to the console every 5% completion.

 * @param ph The Structure of Arrays (SoA) containing the informations of the superphotons.
 * @param d_p Pointer or Texture Object to the device plasma parameters, depending on the 
 * `DO_NOT_USE_TEXTURE_MEMORY` flag.
 * @param d_table_ptr Pointer to the scattering probability and cross-section lookup tables.
 * @param scat_ofphoton A secondary SoA used to store the relevant local and instantaneous information of the scattered photon 
 * to be processed later.
 * @param max_partition_ph Total number of superphotons assigned to this specific tracking batch.
 * @return void
 */
__global__ void track(
    struct of_photonSOA ph, 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
        double * __restrict__ d_p,
    #else
        cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, 
    struct of_photonSOA scat_ofphoton, 
    const unsigned long long max_partition_ph);


/**
 * @brief Assign each scattered superphoton to a thread through dynamic load balancing to be evolved through the geodesic.
 *
 * To handle the variable computational cost of different 
 * geodesics (where some photons escape quickly and others undergo multiple scatterings), 
 * this kernel uses dynamic load balancing Threads fetch a global photon index \f$ n \f$ 
 * dynamically, ensuring that as soon as a thread finishes processing a photon, it immediately
 * pulls the next available index. This keeps all GPU execution units busy regardless of the
 * variation in workload per photon.
 *
 *
 * @note **Progress Monitoring**: Thread with global index 0 acts as a monitor, calculating and 
 * printing the simulation progress to the console every 5% completion.

 * @param ph The Structure of Arrays (SoA) containing the informations of the superphotons.
 * @param d_p Pointer or Texture Object to the device plasma parameters, depending on the 
 * `DO_NOT_USE_TEXTURE_MEMORY` flag.
 * @param d_table_ptr Pointer to the scattering probability and cross-section lookup tables.
 * @param scat_ofphoton A secondary SoA used to store the relevant local and instantaneous information of the scattered photon 
 * to be processed later.
 * @param n Current scattering layer index.
 * @param round_num_scat_init The starting global photon index for this scattering batch.
 * @param round_num_scat_end The ending global photon index for this scattering batch.
 * @param bias_tuning_step The current bias tuning step index, used to adjust the bias guess for the scattering kernel launches. This variable is for printing purposes only.
 * @return void
 */
__global__ void track_scat(struct of_photonSOA ph, 
    #ifdef DO_NOT_USE_TEXTURE_MEMORY
        double * __restrict__ d_p,
    #else
        cudaTextureObject_t d_p,
    #endif
    const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int n, unsigned long long round_num_scat_init, unsigned long long round_num_scat_end, const int bias_tuning_step);


/**
 * @brief Final processing kernel to record escaped superphotons into the observed spectrum by dynamically allocating each superphoton to each thread.
 *
 * This kernel iterates through the processed full evolved batch of photons and identifies those that 
 * have satisfied the "escape" criteria. For each successful photon, it updates the global spectrum object.
 *
 *
 * @param ph The Structure of Arrays (SoA) containing the final state of the photons.
 * @param d_spect Pointer to the device spectrum structure where intensity is accumulated.
 * @param max_partition_ph The total number of photons to be checked in this batch.
 * @param nblocks The number of CUDA blocks used in the grid.
 * @return void
 */
__global__ void record(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks);

/**
 * @brief  Recording kernel for photons that have been created due to scattering events by dynamically allocating each superphoton to each thread.
 *
 * This kernel iterates through the processed full evolved batch of photons and identifies those that 
 * have satisfied the "escape" criteria. For each successful photon, it updates the global spectrum object.
 * 
 * @note Unlike the primary record kernel, this function must calculate the correct memory offsets 
 * into the scattering SoA, as photons from different scattering orders are stored 
 * sequentially.
 *
 * @param ph The Structure of Arrays (SoA) containing the scattered superphotons.
 * @param d_spect Pointer to the device spectrum object for intensity accumulation.
 * @param max_partition_ph Maximum photon capacity (included for signature consistency).
 * @param nblocks Number of CUDA blocks in the grid.
 * @param n The current scattering order (round) being processed.
 * @return void
 */
__global__ void record_scattering(
    struct of_photonSOA ph, 
    struct of_spectrum * __restrict__ d_spect, 
    const unsigned long long max_partition_ph, 
    const int nblocks, 
    const int n
);
__global__ void record_scattering(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks, const int n);



/**
 * @brief The main host-side controller for the GPU-accelerated Monte Carlo simulation.
 *
 * This function manages the high-level execution flow, coordinating memory management, 
 * data transfers, and kernel launches. It implements a multi-pass approach to handle 
 * large photon populations through batching and manages iterative scattering rounds.
 *
 * **Execution Workflow**:
 * 
 * a)  **Initialization**: Allocates device memory for plasma properties, geometry, 
 * scattering tables, and the final spectrum.
 * 
 * b)  **Generation Phase**: Determines the total 
 * number of grid superphotons to be emitted based on local fluid conditions.
 * 
 * c)  **Batch Processing Loop**: Divides the total photon count into manageable 
 * partitions to prevent GPU memory overflow. For each batch:
 * 
 * - **Sampling**: Initializes photon positions and momenta.
 * - **Tracking**: Integrates geodesics in the curved spacetime.
 * - **Recording**: Bins escaped "primary" photons into the spectrum.
 * 
 * d)  **Scattering Iterations**: Enters a secondary loop to process Inverse Compton 
 * scattering. It tracks and records photons through multiple 
 * scattering "rounds" until the population of scattered photons is depleted or the `MAX_LAYER_SCA` 
 * limit is reached.
 * 
 * e)  **Finalization**: Transfers the accumulated spectrum back to the host and prints I/O output diagnostics.
 *
 * @param time Current epoch time used for seeding the random number generators.
 * @param p Pointer to the host-side array of primitive fluid variables.
 * @return void
 */
__host__ void mainFlowControl(time_t time, double * p);
#endif
