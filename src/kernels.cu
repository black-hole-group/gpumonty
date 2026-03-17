/*
 * GPUmonty - kernels.cu
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

#include "decs.h"
#include "memory.h"
#include "kernels.h"
#include "weights.h"
#include "tetrads.h"
#include "utils.h"
#include "compton.h"
#include "metrics.h"
#include "radiation.h"
#include "jnu_mixed.h"
#include "curand.h"
#include "track.h"
#include "main.h"
#include "scattering.h"

// __global__ void check(struct of_photonSOA ph_old, struct of_photonSOA ph_new){
// 	unsigned long long test = 39283;
// 	printf("ph_old.K1[photon_index], ph_new.K1[photon_index] = %le, %le\n", ph_old.K1[test], ph_new.K1[test]);
// }

// __global__ void checkbias(){
// 	printf("dbias[0], dbias[1], dbias[2] = %g, %g, %g\n", d_bias_guess[0], d_bias_guess[1], d_bias_guess[2]);
// }

__host__ void mainFlowControl(time_t time, double * p){
	/*
	Launches the kernels that will generate the photons and sample them. It will also track the photons along the geodesics and solve the scattered photons.

	Parameters:
	@time: This is the usual C function that returns the number of seconds since the epoch. It is used to seed the random number generator
	@p: Array of the primitive variables at each grid cell

	Variables:
	@start: Start event to measure the time of the kernel
	@stop: Stop event to measure the time of the kernel
	@milliseconds: Time in milliseconds that the kernel took to execute
	@d_spect: Array of struct of_spectrum in device. This is where we save the spectrum properties of the photons.
	@d_p: Array of the primitive variables in the device
	@d_table_ptr: Array of the table values in the device
	@d_geom: Array of the struct of_geom in the device
	@d_index_to_ijk: Array of the cumulative sum of photons per zone in the device
	@gen_superph: Number of generated photons
	@generated_photons_arr: Array of the number of photons to be generated in each zone
	@dnmax_arr: Array of the dnmax values in each zone
	@max_block_number: Maximum number of blocks that can be launched
	@instant_photon_number: Number of photons to be generated in each batch
	@offset: Offset to be used in the last batch
	@ideal_nblocks: Ideal number of blocks to be launched
	@batch_divisions: Number of batches to divide the photons into
	@initial_photon_states: Array of the initial photon states
	@scat_ofphoton: Array of the scattered photons
	@num_scat_phs: Array of the number of scattered photons
	@instant_partition: Number of the current batch
	@photons_processed: Number of photons processed so far
	@reset: Variable to reset the counters
	@n: Counter to keep track of the number of scattering rounds
	@quit_flag_sca: Flag to quit the scattering process
	@scatterings_performed: Number of scatterings performed
	@nblocks: Number of blocks to be launched

	*/
    cudaEvent_t start, stop;

    float milliseconds = 0;

	cudaEventCreate(&start);

    cudaEventCreate(&stop);

	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	transferParams();
	
    struct of_spectrum* d_spect;
    gpuErrchk(cudaMalloc((void**)&d_spect, N_TYPEBINS * N_THBINS * N_EBINS * sizeof(struct of_spectrum)));
	double * d_p; 
    gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice));

	cudaTextureObject_t dPTableTexObj = 0;
	cudaArray_t dPTableCuArray;
	createdPTextureObj(&dPTableTexObj, p, &dPTableCuArray);
	free(p);

	
	double *d_table_ptr;
    gpuErrchk(cudaMalloc((void**)&d_table_ptr, (NW + 1) * (NT + 1) * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_table_ptr, table, (NW + 1) * (NT + 1) * sizeof(double), cudaMemcpyHostToDevice));



	struct of_geom *d_geom;
	gpuErrchk(cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom)));
    gpuErrchk(cudaMemcpy(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice));



	int max_block_number = setMaxBlocks();

	unsigned long long gen_superph = 0;
	unsigned long long * generated_photons_arr;
	gpuErrchk(cudaMalloc(&generated_photons_arr, N1 * N2 * N3 * sizeof(unsigned long long)));
	double * dnmax_arr;
	gpuErrchk(cudaMalloc(&dnmax_arr, N1 * N2 * N3 * sizeof(double)));

	fprintf(stderr, "Generating super photons!\n");
	cudaEventRecord(start, 0);
    generate_photons<<<N_BLOCKS,N_THREADS>>>(d_geom, d_p, time, generated_photons_arr, dnmax_arr);	
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Generation kernel, execution time: %f s\n",milliseconds/1000.);
	cudaMemcpyFromSymbol(&gen_superph, photon_count, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
	fprintf(stderr, "Number of generated photons: %llu\n", gen_superph);


	unsigned long long * d_index_to_ijk;
	gpuErrchk(cudaMalloc(&d_index_to_ijk, N1 * N2 * N3 * sizeof(unsigned long long)));
	cummulativePhotonsPerZone(generated_photons_arr, d_index_to_ijk);


	/*Creating array of initial superphotons state*/
	int instant_partition = 1;
	int offset = 0;
	struct of_photonSOA initial_photon_states;
	struct of_photonSOA scat_ofphoton;
	unsigned long long num_scat_phs[MAX_LAYER_SCA];
	unsigned long long instant_photon_number = 0;
	unsigned long long photons_processed =0;
	int ideal_nblocks;
	int batch_divisions = 1;
	instant_photon_number = photonsPerBatch(gen_superph, &batch_divisions);



	while(instant_partition <= batch_divisions){
		printf("\n\n\033[1m===========================================\033[0m\n");
		printf("\033[1;34mStarting partition %d out of %d\033[0m\n", instant_partition, batch_divisions);
		//If in the last partition and there is an offset, just do it;
		if(instant_partition == batch_divisions){
			offset = gen_superph % batch_divisions;
			printf("Last partition with an offset of =%d\n", offset);
		}
		instant_photon_number = (unsigned long long)((gen_superph/batch_divisions) + offset);
		printf("Superphotons processed so far %llu. Superphotons to be processed in this batch %llu\n", photons_processed, instant_photon_number);
		ideal_nblocks = (int)ceil((double) instant_photon_number / (double) N_THREADS);

		if(ideal_nblocks == 0){
			ideal_nblocks = 1;
		}

		allocatePhotonData(&initial_photon_states, instant_photon_number);
		{
			if(params.fitBias){
				double ScatteringDynamicalSize = max(2.0 *  params.targetRatio, (double) SCATTERINGS_PER_PHOTON);
				allocatePhotonData(&scat_ofphoton, ScatteringDynamicalSize * instant_photon_number);
			}else{
				allocatePhotonData(&scat_ofphoton, SCATTERINGS_PER_PHOTON *instant_photon_number);
			}

		}

		fprintf(stderr, "\nSampling the photons!\n");
		cudaEventRecord(start, 0);
		if(ideal_nblocks > max_block_number){
			sample_photons_batch<<<N_BLOCKS,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk);
		}else{
			if (ideal_nblocks == 0)
			ideal_nblocks = 1;
			sample_photons_batch<<<N_BLOCKS,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk);
		}
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Sampling kernel execution time: %f s\n", milliseconds/1000.);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in sample_photons_batch %s, partition (%d)\n", cudaGetErrorString(cudaStatus), instant_partition);
			fprintf(stderr, "If the error is invalid memory location, there is probably too much scattering photons, try changing the bias function.\n");
			exit(1);
		}

		unsigned long long reset = 0;
		cudaMemcpyToSymbol(tracking_counter_sampling, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
		fprintf(stderr, "Photon sampling process completed!\n");


	
		fprintf(stderr, "\nTracking photons along the geodesics\n");

		// Checkpoint array to save photon states before evolving. Since we are going to do bias tuning, we might want to retrack
		// the same photons with a different bias parameter.
		struct of_photonSOA PhotonStateCheckPoint;
		if(params.fitBias){
			allocatePhotonData(&PhotonStateCheckPoint, instant_photon_number);
			transferPhotonDataDevtoDev(PhotonStateCheckPoint, initial_photon_states, instant_photon_number);
			// Turn params.bias_guess to the last value used for bias tuning
			gpuErrchk(cudaMemcpyFromSymbol(&(params.bias_guess), d_bias_guess, sizeof(double), 0 * sizeof(double)));
			printf("Using bias_guess parameter %.3e for the tracking\n", params.bias_guess);
		}

		int RedoTuning = 1;
		double InferiorAcceptance = 0.8 * params.targetRatio;
		double SuperiorAcceptance = 1.2 * params.targetRatio;
		int BiasTuning_index = 0;
		double PreviousRatio = 0;
		do{
			cudaEventRecord(start, 0);
			if(ideal_nblocks > max_block_number){
				#ifdef DO_NOT_USE_TEXTURE_MEMORY
					track<<<max_block_number,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, scat_ofphoton, instant_photon_number);
				#else
					track<<<max_block_number,N_THREADS>>>(initial_photon_states, dPTableTexObj, d_table_ptr, scat_ofphoton, instant_photon_number);
				#endif
			}else{
				if (ideal_nblocks == 0)
				ideal_nblocks = 1;
				#ifdef DO_NOT_USE_TEXTURE_MEMORY
					track<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, scat_ofphoton, instant_photon_number);
				#else
					track<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, dPTableTexObj, d_table_ptr, scat_ofphoton, instant_photon_number);
				#endif
			}		

			cudaDeviceSynchronize();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop); 
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Tracking kernel execution time: %f s\n", milliseconds/1000.);
			Flag("The tracking kernel - illegal memory access encountered often means too much scattering happening, try changing the bias tunning or SCATTERINGS_PER_PHOTON in config.h");

			gpuErrchk(cudaMemcpyFromSymbol(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));


			Flag("the tracking kernel");

			if(params.fitBias){
				double Ratio = ((double)num_scat_phs[0])/((double)instant_photon_number);
				double RelativeImprovement = abs(Ratio - PreviousRatio)/PreviousRatio;
				BiasTuning_index++;
				if((Ratio < InferiorAcceptance || Ratio > SuperiorAcceptance) && BiasTuning_index < MAXITER_BIASTUNING && RelativeImprovement > 0.1){
					if (Ratio == 0) Ratio = 1e-5; //Don't allow division by 0.
					params.bias_guess *= params.targetRatio/Ratio;
					printf("\033[1;31mRatio of Scattering/Created is %.3e, should be in the interval[%.3e, %.3e] \033[0m\n", Ratio, InferiorAcceptance, SuperiorAcceptance);
					printf("\033[1;31mTrying new BiasTuning parameter %.3e \033[0m\n", params.bias_guess);
					gpuErrchk(cudaMemcpyToSymbol(d_bias_guess, &(params.bias_guess), sizeof(double), 0 * sizeof(double)));
					//Transfer from the checkpoint to the initial_photon_states, since we want to retrack the same photons with a different bias parameter
					transferPhotonDataDevtoDev(initial_photon_states, PhotonStateCheckPoint, instant_photon_number);

					//Resetting all the arrays and global variables that keep track of progress
					unsigned long long reset = 0;
					memset(num_scat_phs, 0, MAX_LAYER_SCA * sizeof(unsigned long long));
					gpuErrchk(cudaMemcpyToSymbol(tracking_counter, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
					gpuErrchk(cudaMemcpyToSymbol(d_num_scat_phs, num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
				}else{
						if(RelativeImprovement <= 0.1){
							printf("\033[1;33mNo improvement found by enhancing the biasguess, medium is too optically thin \033[0m\n");
							
						}else if(BiasTuning_index < MAXITER_BIASTUNING){
							printf("\033[1;32mBias Found! Ratio of Scattering/Created is %.3e, Relative Improvement: %.3e\033[0m\n",  Ratio, RelativeImprovement);
						}else{
							printf("\033[1;33mBias Tuning limit reached! Latest Ratio is going to be considered.\033[0m\n");
						}
					RedoTuning = 0;
				}
				PreviousRatio = Ratio;
			}
		}while(params.fitBias && RedoTuning && BiasTuning_index < MAXITER_BIASTUNING);



		if(ideal_nblocks > max_block_number){
			record<<<max_block_number,N_THREADS>>>(initial_photon_states, d_spect, instant_photon_number, max_block_number);
		}else{
			if (ideal_nblocks == 0)
			ideal_nblocks = 1;
			record<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_spect, instant_photon_number, ideal_nblocks);
		}			
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in record %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
	 	freePhotonData(&initial_photon_states);
		if(params.fitBias)
		freePhotonData(&PhotonStateCheckPoint);


		gpuErrchk(cudaMemcpyToSymbol(tracking_counter, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
		if(params.scattering){
			printf("number of scattered photons generated = %llu in round 0\n", num_scat_phs[0]);
			N_scatt += num_scat_phs[0];
			printf("\nSolving the scattered photons...\n");
			printf("Code is programed to handle up to %d layers of scattering\n", MAX_LAYER_SCA - 1);
		}


		scattering_flow_control(num_scat_phs, scat_ofphoton, d_spect, instant_photon_number, max_block_number, d_table_ptr, d_p, dPTableTexObj);
		instant_partition +=1;
		photons_processed += instant_photon_number;
	}

	struct of_spectrum ***spect = Malloc3D_Contiguous(N_TYPEBINS, N_THBINS, N_EBINS);	

	gpuErrchk(cudaMemcpy(spect[0][0], d_spect, N_TYPEBINS * N_THBINS * N_EBINS * sizeof(struct of_spectrum), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&N_superph_recorded, d_N_superph_recorded, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
	#ifndef IHARM
		report_spectrum(gen_superph, spect, params.spectrum);
	#else
		report_spectrum_h5(gen_superph, spect, params.spectrum);
	#endif
	
	cudaFree(d_spect);
	cudaFree(generated_photons_arr); 
	cudaFree(dnmax_arr);
	cudaFree(d_geom);
	cudaFree(d_table_ptr);
	cudaFree(d_p);
    cudaFree(d_index_to_ijk);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	Free3D_Contiguous(spect, N_TYPEBINS);
	cudaDestroyTextureObject(dPTableTexObj);
	cudaFreeArray(dPTableCuArray);

}

__launch_bounds__(N_THREADS)
__global__ void generate_photons(const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const time_t time, unsigned long long * __restrict__  generated_photons_arr, double * __restrict__ dnmax_arr){
	unsigned long long generated_photons;
	double dnmax;
	int i, j, k;
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int seed = 139 * global_index + time;
	init_monty_rand(seed);
	curandState localState = my_curand_state[global_index]; 
	
	/*This is how we'll split things between blocks and threads*/
	/*We'll divide N1 * N2 * N3 between blocks*/
	for(int a = global_index; a < d_N1 * d_N2 * d_N3; (a += N_BLOCKS * N_THREADS)){
		k = a % d_N3;
		j = (a/d_N3) % d_N2;
		i = (a/(d_N2 * d_N3));

		/*This portion of the code will estimate the number of photons that are going to be generated in each zone (n2gen). It will also estimate the dnmax
		which will be used when sampling the photons*/
		init_zone(i,j,k, &generated_photons, &dnmax, d_geom, d_p, d_Ns, &localState);
		
		generated_photons_arr[a] = generated_photons;


		dnmax_arr[a] = dnmax;
		atomicAdd(&photon_count, generated_photons);
	}
	my_curand_state[global_index] = localState;
	return;
}



__device__ void init_zone(const int i, const int j, const int k, unsigned long long * __restrict__  n2gen, double * __restrict__ dnmax, const struct of_geom * __restrict__  d_geom, const double * __restrict__ d_p, const int d_Ns_par, curandState * localState)
{
	int l;
	double Ne, Thetae, Bmag, lbth;
	double dl, dn, ninterp;
	double Ucon[NDIM], Bcon[NDIM];
	double lb_min = log(BTHSQMIN);
	double dlb = log(BTHSQMAX / BTHSQMIN) / NINT;
	double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);

	#ifdef __CUDA_ARCH__
    	get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);
	#else
		Thetae = 0.;
		Ne = 0.;
		Bmag = 0.;
	#endif

	if (Ne == 0. || Thetae < THETAE_MIN) {
		*n2gen = 0.;
		*dnmax = 0.;
		return;
	}
	double K2 = K2_eval(Thetae);

	lbth = log(Bmag * Thetae * Thetae);

	dl = (lbth - lb_min) / dlb;
	l = (int) dl;
	dl = dl - l;
	if (l < 0) {
		*dnmax = 0.;
		*n2gen = 0.;
		return;
	} else if (l >= NINT || 1) {
		//printf( "Outside of range! Change Nint!. B * th**2 = %le, lbth = %le, lb_min = %le, dlb = %le l = %d, (i,j) = (%d, %d)\n", Bmag * Thetae * Thetae, lbth, lb_min, dlb, l,i, j);
		ninterp = 0.;
		*dnmax = 0.;
		for (int m = 0; m <= N_ESAMP; m++) {
			// dn = F_eval(Thetae, Bmag,
			// 		exp(m * dlnu +
			// 		lnu_min)) / (exp(d_wgt[m]) +
			// 				1.e-100);
			dn = int_jnu_total(Ne, Thetae, Bmag, exp(m * dlnu + lnu_min), K2) / (exp(d_wgt[m]) + 1.e-100);
			if (dn > *dnmax)
				*dnmax = dn;
			ninterp += dlnu * dn;
		}
		// ninterp *= d_dx[1] * d_dx[2] * d_dx[3] * d_L_unit * d_L_unit * d_L_unit
		// 	* M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		// 	* 1. / HPL;
		ninterp *= d_dx[1] * d_dx[2] * d_dx[3] * d_L_unit * d_L_unit * d_L_unit * 1./HPL;

	} else {
		if (isinf(d_nint[l]) || isinf(d_nint[l + 1])) {
			ninterp = 0.;
			*dnmax = 0.;
		} else {
			ninterp =
				exp((1. - dl) * d_nint[l] + dl * d_nint[l + 1]);	

			*dnmax =
				exp((1. - dl) * d_dndlnu_max[l] +
				dl * d_dndlnu_max[l + 1]);
		}

	}
		


	if (K2 == 0.) {
		*n2gen = 0.;
		*dnmax = 0.;
		return;
	}
	
	//double nz = d_geom[SPATIAL_INDEX2D(i,j)].g * Ne * Bmag * Thetae * Thetae * ninterp / K2;
	double nz = d_geom[SPATIAL_INDEX2D(i,j)].g * ninterp;
	if (nz > d_Ns_par * log(NUMAX / NUMIN)) {
		printf(
			"Something very wrong in zone %d %d: \n Ne = %le, B=%g  Thetae=%g  K2=%g  ninterp=%g\n", i, j, Ne, Bmag, Thetae, K2, ninterp);
		printf(
			"Something very wrong in zone %d %d: nz = %le, d_Ns = %d, g = %le\n",i, j, nz, d_Ns_par, d_geom[SPATIAL_INDEX2D(i,j)].g);
		printf("dl = %le, d_nint[l] = %le, d_ninit[l+1] = %le, logratio = %le\n", dl, d_nint[l], d_nint[l+1], log(NUMAX/NUMIN));
		*n2gen = 0.;
		*dnmax = 0.;
	}else{
		if (fmod(nz, 1.) > curand_uniform_double(localState)) {
			*n2gen = (int) (nz) + 1; 
		} else {
			*n2gen = (int) (nz);
		}
	}

	return;
}

__launch_bounds__(N_THREADS)
__global__ void sample_photons_batch(struct of_photonSOA ph_init, const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const unsigned long long * __restrict__  generated_photons_arr, const double * __restrict__ dnmax_arr, const int max_partition_ph, 
	const unsigned long long photons_processed_sofar, const unsigned long long * __restrict__  index_to_ijk){
		int i,j,k;
		unsigned long long photon_index = 0;
		const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
		int zone_index = 0;
		double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
		int past_zone = (d_N1 * d_N2 * d_N3);
		curandState localState = my_curand_state[global_index];
		while (true) {
			photon_index = (atomicAdd(&tracking_counter_sampling, 1)-1);
			if (photon_index >= max_partition_ph){
				break;
			}
	
			zone_index = findPhotonIndex(index_to_ijk, d_N1 * d_N2 * d_N3, photons_processed_sofar +photon_index);
			k = zone_index % d_N3;
			j = (zone_index/d_N3) % d_N2;
			i = (zone_index/(d_N2 * d_N3));


			/*Sample all the photons generated in init_zone*/
			sample_zone_photon(i,j,k, dnmax_arr[zone_index], ph_init, d_geom, d_p, (past_zone == zone_index? 0 : 1), photon_index, Econ, Ecov, &localState);
			past_zone = zone_index;
			
		}
		my_curand_state[global_index] = localState;
}

__device__ void sample_zone_photon(const int i, const int j, const int k, const double dnmax, 
    struct of_photonSOA ph, const struct of_geom *  d_geom, 
    const double *  d_p, const int zone_flag, const unsigned long long ph_arr_index,
    double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState *  localState)
{
    double nu, weight;
    
    // Scope 1: Initial setup and coordinate transformation
    {
        double Xarray[NDIM];
        coord(i, j, k, Xarray);
        // Store back immediately, don't keep Xarray alive
        ph.X0[ph_arr_index] = Xarray[0];
        ph.X1[ph_arr_index] = Xarray[1];
        ph.X2[ph_arr_index] = Xarray[2];
        ph.X3[ph_arr_index] = Xarray[3];
    } // Xarray goes out of scope here
    
    // Scope 2: Get fluid properties
    double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM];
	#ifdef __CUDA_ARCH__
    	get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);
	#else
		Thetae = 0.;
		Ne = 0.;
		Bmag = 0.;
	#endif
    
    // Scope 3: Sample frequency
    {
        const double lnu_min = log(NUMIN);
        const double Nln = log(NUMAX) - lnu_min;
		const double K2 = K2_eval(Thetae);
        do {
            nu = exp(curand_uniform_double(localState) * Nln + lnu_min);
            weight = linear_interp_weight(nu);
        //} while (curand_uniform_double(localState) > (F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / dnmax);
		}while (curand_uniform_double(localState) > (int_jnu_total(Ne, Thetae, Bmag, nu, K2) / (weight + 1.e-100)) / dnmax);
		ph.w[ph_arr_index] = weight;
    } // lnu_min, Nln go out of scope
    
    // Scope 4: Sample angles  
    double cth;
    {
		const double K2 = K2_eval(Thetae);
        const double jmax = jnu_total(nu, Ne, Thetae, Bmag, M_PI / 2., K2);
		double j_th;
		double th;
        do {
            cth = 2. * curand_uniform_double(localState) - 1.;
            th = acos(cth);
        	j_th = jnu_total(nu, Ne, Thetae, Bmag, th, K2);
        } while (curand_uniform_double(localState) > j_th / jmax);
		ph.ratio_brems[ph_arr_index] = jnu_ratio_brems(nu, Ne, Thetae, Bmag, th, K2);

    } // jmax, th, j_th go out of scope
    
    // Reuse arrays - use one array for both K_tetrad and final storage
    double K_data[NDIM]; // This will serve as both K_tetrad and tmpK
    
    // Scope 5: Build momentum vector
    {
        const double sth = sqrt(1. - cth * cth);
        const double phi = 2. * M_PI * curand_uniform_double(localState);
        const double E = nu * HPL / (ME * CL * CL);
        
        K_data[0] = E;
        K_data[1] = E * cth;
        K_data[2] = E * cos(phi) * sth;
        K_data[3] = E * sin(phi) * sth;
    } // sth, phi, E go out of scope, cphi/sphi never created
    
    // Scope 6: Handle tetrad if needed
    if (zone_flag) {
        double bhat[NDIM];
        if (Bmag > 0.) {
            const double inv_Bmag = d_B_unit / Bmag;
            for (int l = 0; l < NDIM; l++) {
                bhat[l] = Bcon[l] * inv_Bmag;
            }
        } else {
            bhat[0] = bhat[2] = bhat[3] = 0.;
            bhat[1] = 1.;
        }
        make_tetrad(Ucon, bhat, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Econ, Ecov);
    }
    
    // Scope 7: Final transformations - reuse K_data array
    {
        double Karray[4]; // Temporary for coordinate transformation
        tetrad_to_coordinate(Econ, K_data, Karray);
        
        K_data[0] *= -1.; // Modify in place
        tetrad_to_coordinate(Ecov, K_data, K_data); // Reuse K_data for output


        ph.E[ph_arr_index] = ph.E0[ph_arr_index] = ph.E0s[ph_arr_index] = -K_data[0];


        ph.K0[ph_arr_index] = Karray[0];
        ph.K1[ph_arr_index] = Karray[1];
        ph.K2[ph_arr_index] = Karray[2];
        ph.K3[ph_arr_index] = Karray[3];
    }
    
    // Store final values
    ph.tau_scatt[ph_arr_index] = 0.;
    ph.tau_abs[ph_arr_index] = 0.;
    ph.X1i[ph_arr_index] = ph.X1[ph_arr_index];
    ph.X2i[ph_arr_index] = ph.X2[ph_arr_index];
    ph.nscatt[ph_arr_index] = 0;
}

__launch_bounds__(N_THREADS)
__global__ void track(struct of_photonSOA ph, 
	#ifdef DO_NOT_USE_TEXTURE_MEMORY
	double * __restrict__  d_p,
	#else
	cudaTextureObject_t d_p,
	#endif
	const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long max_partition_ph){
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long photon_index = 0;
	int n = 1;
	curandState localState = my_curand_state[global_index];
	/*track each photon we created along its geodesic*/

	if(global_index == 0)
		printf("d_max_tau_scatt = %le for round %d\n", d_max_tau_scatt, n);
    while (true) {
        // Each thread grabs the next available photon index atomically
        photon_index = (atomicAdd(&tracking_counter, 1) - 1);

        // If all photons are processed, exit the loop
        if (photon_index >= max_partition_ph) break;
        
		// Track the photon
        track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, 0, 0, photon_index, &localState);

        // Progress indicator
        if (global_index == 0) {
            float percentage = 100 - ((max_partition_ph-  photon_index) * 100) / max_partition_ph;
            if (percentage >= n * 5.0f) {
				printf("\rProgress: \033[1;32m%llu%%\033[0m   ", (unsigned long long)percentage);
				n++;
            }
        }
	}

	/*Set to 100%*/
	if(global_index == 0){
		printf("\rProgress: \033[1;32m100%%\033[0m   \n");
	}
	if(global_index > N_BLOCKS * N_THREADS){
		printf("Warning! Too many threads! Some threads are not being used!\n");
	}
	my_curand_state[global_index] = localState;
}
	
__launch_bounds__(N_THREADS)
__global__ void record(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks)
{
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	/*track each photon we created along its geodesic*/
	for(unsigned long long a = global_index; a < max_partition_ph; (a += nblocks * N_THREADS)){
		if(record_criterion(ph.X1[a]))
		record_super_photon(ph, d_spect, a);
	}
}


__launch_bounds__(N_THREADS)
__global__ void track_scat(struct of_photonSOA ph, 
	#ifdef DO_NOT_USE_TEXTURE_MEMORY
	 double * __restrict__ d_p, 
	#else
	 cudaTextureObject_t d_p,
	#endif
	const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int n, unsigned long long round_num_scat_init, unsigned long long round_num_scat_end){
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = my_curand_state[global_index];
	double Ne, Thetae, B;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
	unsigned long long scattering_counter_local = round_num_scat_init - 1;
	int n_progress = 1;
	

	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Interval going from %llu to %llu in round! %d\n", round_num_scat_init, round_num_scat_end, n);
	}
	
	while(true){
		scattering_counter_local = (round_num_scat_init) + (atomicAdd(&scattering_counter, 1));

		if (scattering_counter_local >= round_num_scat_end) break;

		double X[NDIM] = {ph.X0[scattering_counter_local], ph.X1[scattering_counter_local], ph.X2[scattering_counter_local], ph.X3[scattering_counter_local]};
		#ifndef SPHERE_TEST
			get_fluid_params(X, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, d_p);
		#else
			get_fluid_params(X, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);
		#endif
		double Gcov[NDIM][NDIM];
		gcov_func(X, Gcov);
		scatter_super_photon(ph, ph, Ne, Thetae, B, Ucon, Bcon, Gcov, &localState, scattering_counter_local);
		if (ph.w[scattering_counter_local] < 1.e-100) {	/* must have been a problem popping k back onto light cone */
			continue;
		}
		track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, round_num_scat_end, n, scattering_counter_local, &localState);
        // Progress indicator
		if (global_index == 0) {		
			float percentage = (float)(scattering_counter_local - round_num_scat_init) * 100.0f / (float)(round_num_scat_end - round_num_scat_init);
			if (percentage >= n_progress * 5.0f) {
				printf("\rProgress: \033[1;32m%llu%%\033[0m   ", (unsigned long long)percentage);
				n_progress++;
            }
		}
	}
	/*Set to 100%*/
	if(global_index == 0){
		printf("\rProgress: \033[1;32m100%%\033[0m   \n");
	}
	my_curand_state[global_index] = localState;
}

__launch_bounds__(N_THREADS)
__global__ void record_scattering(struct of_photonSOA ph, struct of_spectrum * __restrict__ d_spect, const unsigned long long  max_partition_ph, const int nblocks, const int n)
{
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	/*track each photon we created along its geodesic*/

	for(unsigned long long a = global_index; a < d_num_scat_phs[n-1]; (a += nblocks * N_THREADS)){

		if(record_criterion(ph.X1[a]))
		record_super_photon(ph, d_spect, a);
	}
}
