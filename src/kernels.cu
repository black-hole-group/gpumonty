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


__host__ void CreateCUDAStartStop(cudaEvent_t * start, cudaEvent_t * stop){
	cudaEventCreate(start);
	cudaEventCreate(stop);
    cudaEventRecord(*start, 0);
}

__host__ void DiagnosticRunTime(cudaEvent_t start, cudaEvent_t stop, const char * kernel_name){
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%s kernel execution time: %f s\n", kernel_name, milliseconds/1000.);
}

__host__ void GPUWorker(unsigned long long photons_per_batch, unsigned long long totph_per_GPU, int batch_divisions, struct of_geom *d_geom, double *d_p, unsigned long long * generated_photons_arr, double * dnmax_arr, unsigned long long * d_index_to_ijk, double *d_table_ptr, cudaTextureObject_t dPTableTexObj, char * log_filename, int gpu_id, cudaStream_t local_stream, struct of_spectrum* d_spect)
{
	cudaEvent_t start, stop;
	CreateCUDAStartStop(&start, &stop);
	float milliseconds = 0;
	cudaError_t cudaStatus;
	int max_block_number = setMaxBlocks();
	int instant_partition = 1;
	int offset = 0;
	struct of_photonSOA initial_photon_states;
	struct of_photonSOA scat_ofphoton;
	unsigned long long num_scat_phs[MAX_LAYER_SCA];
	unsigned long long instant_photon_number = 0;
	unsigned long long photons_processed =0;
	int ideal_nblocks;
	double saved_tracking_bias = params.bias_guess;
	double saved_scat_bias[MAX_LAYER_SCA];
	for(int k = 0; k < MAX_LAYER_SCA; k++)
    	saved_scat_bias[k] = params.bias_guess;


	// When doing bias tuning, this will be necessary
	// We are gonna assign different bias for each GPU. The way we are dividing is within zone cells
	// That way, theoretically, they are sampling just lower resolution runs...
	double local_bias_guess = params.bias_guess;

	// Each GPU should write to its own file to avoid write conflicts and make debugging easier.
	FILE *log_file = fopen(log_filename, "w");
	fprintf(log_file, "Log for GPU number %d\n", gpu_id);
	setvbuf(log_file, NULL, _IONBF, 0); // Disables buffering


	//Here we start iterating through different partitions created due to memory limittations
	while(instant_partition <= batch_divisions){

		fprintf(log_file, "\n\n\033[1m===========================================\033[0m\n");
		fprintf(log_file, "\033[1;34mStarting partition %d out of %d\033[0m\n", instant_partition, batch_divisions);
		printf("\033[1;34m GPU %d Starting partition %d out of %d\033[0m\n", gpu_id, instant_partition, batch_divisions);
		// instant_photon_number is the number of photons getting processed in this batch/partition.
		instant_photon_number = (unsigned long long)(photons_per_batch);
		//If in the last partition and there is an offset, just do it;
		if(instant_partition == batch_divisions){
			offset = totph_per_GPU % batch_divisions;
			instant_photon_number += offset;
		}
		
		fprintf(log_file, "Superphotons processed so far %llu. Superphotons to be processed in this batch %llu\n", photons_processed, instant_photon_number);
		
		// Don't launch more blocks than necessary. If we can resolve all the photons with n blocks of N_THREADS, why launch more?
		// Sometimes it was becoming 0, so if you have less photons than threads into a single block, just launch 1 block.
		ideal_nblocks = max((int)ceil((double) instant_photon_number / (double) N_THREADS), 1);

		//Allocating the photon states for the grid photons of this batch.
		allocatePhotonData(&initial_photon_states, instant_photon_number);

		//Allocate the photon states for the scattering photons of this layer in particular
		if(params.fitBias){
			// Before, if the person chose a targetRatio that was bigger than Scatterings_per_photon, it would crash.
			// Now, we do it dynamically, but also, we give a margin to the target ratio, so it can wobble around up and down the ratio
			double ScatteringDynamicalSize = max(2.0 *  params.targetRatio, (double) SCATTERINGS_PER_PHOTON);
			allocatePhotonData(&scat_ofphoton, ScatteringDynamicalSize * instant_photon_number);
		}else{
			allocatePhotonData(&scat_ofphoton, SCATTERINGS_PER_PHOTON *instant_photon_number);
		}

		fprintf(log_file, "\nSampling the photons!\n");
		cudaEventRecord(start, 0);
		//We start sampling the photons, in this function we choose the superphoton energy, theta and phi and weight.
		sample_photons_batch<<<N_BLOCKS,N_THREADS, 0, local_stream>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk, gpu_id);
		cudaStreamSynchronize(local_stream);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&milliseconds, start, stop);
		fprintf(log_file, "Sampling kernel execution time: %f s\n", milliseconds/1000.);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(log_file, "in sample_photons_batch %s, partition (%d)\n", cudaGetErrorString(cudaStatus), instant_partition);
			fprintf(log_file, "If the error is invalid memory location, there is probably too much scattering photons, try changing the bias function.\n");
			exit(1);
		}

		//We keep a tracker of the number of photons sampling to do it dynamically
		//Since we have already sampled them, now just reset it back to 0
		unsigned long long reset = 0;
		symbolToDevice(&tracking_counter_sampling, &reset, sizeof(unsigned long long), local_stream);
		fprintf(log_file, "Photon sampling process completed!\n");

		fprintf(log_file, "\nTracking photons along the geodesics\n");

		// Checkpoint array to save photon states before evolving. Since we are going to do bias tuning, we might want to retrack
		// the same photons with a different bias parameter.
		struct of_photonSOA PhotonStateCheckPoint;
		if(params.fitBias){
			allocatePhotonData(&PhotonStateCheckPoint, instant_photon_number);
			transferPhotonDataDevtoDev(PhotonStateCheckPoint, initial_photon_states, instant_photon_number, local_stream);
			// Turn params.bias_guess (local_bias_guess) to the last value used for bias tuning
			symbolToDevice(&d_bias_guess, &(local_bias_guess), sizeof(double), local_stream);
			fprintf(log_file, "Using bias_guess parameter %.3e for the tracking\n", local_bias_guess);
		}

		//Parameters for bias tuning, we allow 20% wobble around the target ratio, and we also check if there is a significant improvement in the ratio to continue tuning, if not, we just stop, since we might be in a optically thin medium where bias tuning doesn't help that much.
		int RedoTuning = 1;
		double InferiorAcceptance = 0.8 * params.targetRatio;
		double SuperiorAcceptance = 1.2 * params.targetRatio;
		int BiasTuning_index = 0;
		double PreviousRatio = 0;

		do{
			cudaEventRecord(start, 0);
			//Track the grid superphotons in this kernel.
			track<<<min(ideal_nblocks, max_block_number),N_THREADS, 0, local_stream>>>(initial_photon_states, 
				#ifdef DO_NOT_USE_TEXTURE_MEMORY
				 d_p,
				#else
				 dPTableTexObj,
				#endif
				d_table_ptr, scat_ofphoton, instant_photon_number);
			cudaStreamSynchronize(local_stream);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop); 
			cudaEventElapsedTime(&milliseconds, start, stop);
			fprintf(log_file, "Tracking kernel execution time: %f s\n", milliseconds/1000.);
			Flag("The tracking kernel - illegal memory access encountered often means too much scattering happening, try changing the bias tunning or SCATTERINGS_PER_PHOTON in config.h");

			//Add the number of scattered superphotons generated in first round to host array num_scat_phs.
			//We are going to use it for bias tuning.
			symbolFromDevice(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), local_stream);
			Flag("the tracking kernel");

			//Here is where the bias tunning happens, the algorithm goes as follows...
			if(params.fitBias){
				// Calculate the current Ratio of scattered photons to initial photons
				double Ratio = ((double)num_scat_phs[0])/((double)instant_photon_number);

				//Calculate the RelativeImprovement following
				// RelativeImprovement = |(Current Ratio - Previous Ratio)|/ Previous Ratio
				double RelativeImprovement = abs(Ratio - PreviousRatio)/PreviousRatio;

				//Add a counter to check how many times we have done bias tunning, we want to avoid infinite loops.
				//The Maximum is set by MAXITER_BIASTUNING in config.h, but we also stop if there is no
				// significant improvement in the ratio, since it might be a optically thin medium where bias tunning
				// doesn't help that much.
				BiasTuning_index++;

				//Check if the new Ratio is within the acceptable interval around the target ratio.
				if((Ratio < InferiorAcceptance || Ratio > SuperiorAcceptance) && BiasTuning_index < MAXITER_BIASTUNING && RelativeImprovement > 0.1){
					//Don't allow division by 0 (in case no scattered superphoton has been generated)
					if (Ratio == 0) Ratio = 1e-5; //Don't allow division by 0.
					fprintf(log_file, "\033[1;31mWith previous bias_guess parameter %.3e, Ratio of Scattering/Created is %.3e, which is out of the acceptance interval [%.3e, %.3e]. \033[0m\n", local_bias_guess, Ratio, InferiorAcceptance, SuperiorAcceptance);
					// The new guess is evolved as following
					local_bias_guess *= params.targetRatio/Ratio;
					fprintf(log_file, "\033[1;31mTrying new BiasTuning parameter %.3e \033[0m\n", local_bias_guess);

					// Update the bias guess in the device symbol
					symbolToDevice(&d_bias_guess, &(local_bias_guess), sizeof(double), local_stream);
					//Transfer from the checkpoint to the initial_photon_states, since we want to retrack the same photons with a different bias parameter
					transferPhotonDataDevtoDev(initial_photon_states, PhotonStateCheckPoint, instant_photon_number, local_stream);

					//Resetting all the arrays and global variables that keep track of progress
					unsigned long long reset = 0;
					memset(num_scat_phs, 0, MAX_LAYER_SCA * sizeof(unsigned long long));
					symbolToDevice(&tracking_counter, &reset, sizeof(unsigned long long), local_stream);
					symbolToDevice(&d_num_scat_phs, num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), local_stream);
				}else{
					//For this else, the ratio is either good enough or
					// we have reached the maximum number of iterations for bias tunning,
					//  or there is no significant improvement in the ratio.
					if(RelativeImprovement <= 0.1){
						fprintf(log_file, "\033[1;33mNo improvement found by enhancing the biasguess, medium is too optically thin \033[0m\n");
					}else if(BiasTuning_index < MAXITER_BIASTUNING){
						fprintf(log_file, "\033[1;32mBias Found! Ratio of Scattering/Created is %.3e, Relative Improvement: %.3e\033[0m\n",  Ratio, RelativeImprovement);
					}else{
						fprintf(log_file, "\033[1;33mBias Tuning limit reached! Latest Ratio is going to be considered.\033[0m\n");
					}
					RedoTuning = 0;
				}
				PreviousRatio = Ratio;
			}
		//In bias tunning, keep doing this until we find a good bias parameter that makes the ratio of scattering/created 
		//close to the target ratio, or we reach the maximum number of iterations for bias tunning, or there is no significant 
		//improvement in the ratio.
		}while(params.fitBias && RedoTuning && BiasTuning_index < MAXITER_BIASTUNING);


		//Record the superphotons tracked in this batch, we do it after bias tuning, since we want to record the photons with the best bias parameter we found.
		record<<<min(ideal_nblocks, max_block_number),N_THREADS, 0, local_stream>>>(initial_photon_states, d_spect, instant_photon_number, min(ideal_nblocks, max_block_number));		
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(log_file, "in record %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
	 	freePhotonData(&initial_photon_states);
		
		if(params.fitBias)
			freePhotonData(&PhotonStateCheckPoint);
		
		// Save the converged tracking bias
		saved_tracking_bias = local_bias_guess;

		symbolToDevice(&tracking_counter, &reset, sizeof(unsigned long long), local_stream);	

		if(params.scattering){
			fprintf(log_file, "number of scattered photons generated = %llu in round 0\n", num_scat_phs[0]);
			#pragma omp atomic
			N_scatt += num_scat_phs[0];
			fprintf(log_file, "\nSolving the scattered photons...\n");
			fprintf(log_file, "Code is programed to handle up to %d layers of scattering\n", MAX_LAYER_SCA - 1);
		}


		//In here we deal with te different scattering layers
		scattering_flow_control(num_scat_phs, &scat_ofphoton, d_spect, instant_photon_number, max_block_number, d_table_ptr, d_p, dPTableTexObj, local_stream, saved_scat_bias, log_file);
		local_bias_guess = saved_tracking_bias;

		//Advance one partition and add the ammount of photons processed in this batch
		instant_partition +=1;
		photons_processed += instant_photon_number;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(log_file);
}

__host__ void CummulativePhotonsPerZonePerGPU(
    unsigned long long *h_generated_photons_arr, 
    unsigned long long *h_index_to_ijk_2d, 
    unsigned long long *h_total_per_gpu, 
    int num_gpus) 
{
    // The idea behind this function is that each cpu thread will solely get
    // its specific "slice" of that array and hands it to its assigned GPU.
    int num_zones = N1 * N2 * N3;
    unsigned long long *current_cum_sum = (unsigned long long *)calloc(num_gpus, sizeof(unsigned long long));

    // Keeps track of whose turn it is to get the extra photon across the WHOLE grid
    int next_gpu_for_remainder = 0; 

    // Distribute photons and build the cumulative sums per GPU
    for (int z = 0; z < num_zones; z++) {
        unsigned long long total_in_zone = h_generated_photons_arr[z];
        unsigned long long base_per_gpu = total_in_zone / num_gpus;
        unsigned long long remainder = total_in_zone % num_gpus;

        //Give every GPU its guaranteed base share for this zone
        for (int g = 0; g < num_gpus; g++) {
            current_cum_sum[g] += base_per_gpu;
        }

        //Deal out the extra remainder photons one by one
        for (unsigned long long r = 0; r < remainder; r++) {
            current_cum_sum[next_gpu_for_remainder] += 1;
            
            // Move to the next GPU, wrapping back to 0 if we hit the end
            next_gpu_for_remainder = (next_gpu_for_remainder + 1) % num_gpus;
        }

        //Store the updated running totals in the flattened 2D array
        for (int g = 0; g < num_gpus; g++) {
            // The index here will be: (GPU_ID * NUM_ZONES) + ZONE_ID
            h_index_to_ijk_2d[g * num_zones + z] = current_cum_sum[g];
        }
    }

    // Save the grand total of photons assigned to each GPU
    for (int g = 0; g < num_gpus; g++) {
        h_total_per_gpu[g] = current_cum_sum[g];
    }
    
    free(current_cum_sum);
}

__global__ void AccumulateSpectrum(struct of_spectrum *dst, struct of_spectrum *src, int total_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins) return;

    dst[idx].dNdlE     += src[idx].dNdlE;
    dst[idx].dEdlE     += src[idx].dEdlE;
    dst[idx].nph       += src[idx].nph;
    dst[idx].nscatt    += src[idx].nscatt;
    dst[idx].X1iav     += src[idx].X1iav;
    dst[idx].X2isq     += src[idx].X2isq;
    dst[idx].X3fsq     += src[idx].X3fsq;
    dst[idx].tau_abs   += src[idx].tau_abs;
    dst[idx].tau_scatt += src[idx].tau_scatt;
    dst[idx].E0        += src[idx].E0;
}
	
  
__host__ void mainFlowControl(time_t time, double * p){
    //Figure out how many gpus we have available for us.
    int num_gpus;
    gpuErrchk(cudaGetDeviceCount(&num_gpus));
	num_gpus = 1;
    // Total number of cells
    int num_zones = N1 * N2 * N3;

	printf("\n\n \033[1mUsing %d GPUs available...\033[0m\n\n", num_gpus);
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		unsigned long long memory_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
		printf("GPU %d is: %s with memory %llu GB\n", i, prop.name, memory_gb);
	}


    //Transfer from CPU to GPU the global variables that are needed for the photon generation process.
    //Transfer the geom array that stores gdet, gcov, gcon for each zone
    struct of_geom *d_geom_gen;
    gpuErrchk(cudaMalloc(&d_geom_gen, N1 * N2 * sizeof(struct of_geom)));
    gpuErrchk(cudaMemcpy(d_geom_gen, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice));

    //Transfer the p array that stores the primitive variables for each zone
    double * d_p_gen; 
    gpuErrchk(cudaMalloc((void**)&d_p_gen, NPRIM * num_zones * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_p_gen, p, NPRIM * num_zones * sizeof(double), cudaMemcpyHostToDevice));

    // Transfer the generated_photons_arr which is the array that will store the total number of superphotons generated in each zone.
    unsigned long long * d_generated_photons_gen;
    gpuErrchk(cudaMalloc(&d_generated_photons_gen, num_zones * sizeof(unsigned long long)));

    // Transfer the dnmax_arr which is the array that will store the maximum value for superphoton generation in each zone.
    double * d_dnmax_gen;
    gpuErrchk(cudaMalloc(&d_dnmax_gen, num_zones * sizeof(double)));

    // This function transfer the GRMHD simulation header parameters to GPU memory
    transferParams(0); 

	//Initialize RNG states for photons generations on GPU 0;
	InitializeRNGStates<<<N_BLOCKS, N_THREADS>>>(time, 0);
    //In GPU 0, generate the total number of photons that will be generated in each zone.
	unsigned long long gen_superph = 0;
    {
        cudaEvent_t start, stop;
        CreateCUDAStartStop(&start, &stop);
        printf("Generating photons on GPU 0!\n");
        generate_photons<<<N_BLOCKS,N_THREADS>>>(d_geom_gen, d_p_gen, d_generated_photons_gen, d_dnmax_gen);  
        gpuErrchk(cudaDeviceSynchronize());
        DiagnosticRunTime(start, stop, "Photon Generation");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
		symbolFromDevice(&gen_superph, &photon_count, sizeof(unsigned long long), 0);
        fprintf(stderr, "Number of generated superphotons: %llu\n", gen_superph);
		cudaError_t cudaStatus;
		cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error in generate_photons function on GPU %d: %s\n", 0, cudaGetErrorString(cudaStatus));
			exit(1);
		}
    }

    //Here, we are pulling back results to host and cleaning GPU 0`s memory.
    unsigned long long *h_generated_photons = (unsigned long long *)malloc(num_zones * sizeof(unsigned long long));
    gpuErrchk(cudaMemcpy(h_generated_photons, d_generated_photons_gen, num_zones * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double *h_dnmax_arr = (double *)malloc(num_zones * sizeof(double));
    gpuErrchk(cudaMemcpy(h_dnmax_arr, d_dnmax_gen, num_zones * sizeof(double), cudaMemcpyDeviceToHost));

    // Free the initialization arrays to keep VRAM clean for the main run
    cudaFree(d_geom_gen);
    cudaFree(d_p_gen);
    cudaFree(d_generated_photons_gen);
    cudaFree(d_dnmax_gen);

    //Create one array of spectra for each GPU, since we can't write to the same array from different GPUs, we will have to merge them at the end of the process.
    //struct of_spectrum all_spectra[num_gpus][N_THBINS][N_EBINS];
    
    //After generating the photons, we need to do a cumulative sum of the number of photons generated in each zone so
    //we can know the index of the first photon generated in each zone. This will be used in the sampling process.
    //Bring the initial generated photons array back to the host (if not already there)

    //Allocate space for our 2D array and totals on the host
    unsigned long long *h_index_to_ijk_2d = (unsigned long long *)malloc(num_gpus * num_zones * sizeof(unsigned long long));
    unsigned long long *h_totals_per_gpu = (unsigned long long *)malloc(num_gpus * sizeof(unsigned long long));

    //Pre-calculate the balanced zones
    CummulativePhotonsPerZonePerGPU(h_generated_photons, h_index_to_ijk_2d, h_totals_per_gpu, num_gpus);

	// Have an array of spectra for each GPU
	struct of_spectrum **gpu_spect_ptrs = (struct of_spectrum **)malloc(num_gpus * sizeof(struct of_spectrum*));
	size_t spect_size = N_TYPEBINS * N_THBINS * N_EBINS * sizeof(struct of_spectrum);

	// Allocate the final merged spectrum on host (zero-initialized)
	struct of_spectrum *h_spect_merged = (struct of_spectrum *)calloc(N_TYPEBINS * N_THBINS * N_EBINS, sizeof(struct of_spectrum));

    #pragma omp parallel for num_threads(num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        
        //Lock this CPU thread to the specific GPU with the same id.
        gpuErrchk(cudaSetDevice(i));
		// Have each GPU has its own stream
		cudaStream_t local_stream;
		gpuErrchk(cudaStreamCreate(&local_stream));

		//Initialize the spectrum pointers for each GPU
		gpuErrchk(cudaMalloc(&gpu_spect_ptrs[i], spect_size));
		gpuErrchk(cudaMemset(gpu_spect_ptrs[i], 0, spect_size));

		// Initialize RNG states for each GPU thread with a unique seed, don't initialize for GPU 0 because it has been initialized already to generate the superphotons.
		if(i > 0)
		InitializeRNGStates<<<N_BLOCKS, N_THREADS>>>(time, i);

        
        // Ensure simulation parameters are in local GPU's global memory
        // This sets the grid/grmhd parameters to device memory
        transferParams(local_stream); 

        char log_filename[256];
        sprintf(log_filename, "./log_GPU%d.txt", i);

        //Allocate and copy ALL required arrays into local GPU's local memory
        struct of_geom *local_d_geom;
        gpuErrchk(cudaMalloc(&local_d_geom, N1 * N2 * sizeof(struct of_geom)));
        gpuErrchk(cudaMemcpy(local_d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice));

        //Some other arrays that are gonna be constant for all the gpus are the table for the scattering kernel and the p array 
        //that stores the primitive variables. We can transfer them once and then use them as arguments to the kernel that will call the main loop.
        double * local_d_p; 
        gpuErrchk(cudaMalloc((void**)&local_d_p, NPRIM * num_zones * sizeof(double)));
        gpuErrchk(cudaMemcpy(local_d_p, p, NPRIM * num_zones * sizeof(double), cudaMemcpyHostToDevice));

        cudaTextureObject_t local_dPTableTexObj = 0;
        cudaArray_t local_dPTableCuArray;
        createdPTextureObj(&local_dPTableTexObj, p, &local_dPTableCuArray);

        //Finally, we transfer the table for the scattering kernel to the GPU.
        double *local_d_table_ptr;
		#if VARIABLE_KAPPA
		int kappa_size = KAPPA_NSAMP;
		#else
		int kappa_size = 1;
		#endif
        gpuErrchk(cudaMalloc((void**)&local_d_table_ptr, kappa_size * (NW + 1) * (NT + 1) * sizeof(double)));
        gpuErrchk(cudaMemcpy(local_d_table_ptr, table, kappa_size * (NW + 1) * (NT + 1) * sizeof(double), cudaMemcpyHostToDevice));

        unsigned long long * local_generated_photons_arr;
        gpuErrchk(cudaMalloc(&local_generated_photons_arr, num_zones * sizeof(unsigned long long)));
        gpuErrchk(cudaMemcpy(local_generated_photons_arr, h_generated_photons, num_zones * sizeof(unsigned long long), cudaMemcpyHostToDevice));

        double * local_dnmax_arr;
        gpuErrchk(cudaMalloc(&local_dnmax_arr, num_zones * sizeof(double)));
        gpuErrchk(cudaMemcpy(local_dnmax_arr, h_dnmax_arr, num_zones * sizeof(double), cudaMemcpyHostToDevice));

        unsigned long long * local_d_index_to_ijk;
        gpuErrchk(cudaMalloc(&local_d_index_to_ijk, num_zones * sizeof(unsigned long long)));
        unsigned long long *my_host_slice = &h_index_to_ijk_2d[i * num_zones];
        gpuErrchk(cudaMemcpy(local_d_index_to_ijk, my_host_slice, num_zones * sizeof(unsigned long long), cudaMemcpyHostToDevice));

        //Now we know how many photons we have and therefore, we are gonna run the analysis to see how many superphotons we can take per batch per GPU.
        unsigned long long my_num = h_totals_per_gpu[i];
        int batch_divisions = 1;
        unsigned long long photons_per_batch = photonsPerBatch(my_num, &batch_divisions);

        // Pass the GPU local pointers to the GPU Worker
        GPUWorker(photons_per_batch, my_num, batch_divisions, local_d_geom, local_d_p, local_generated_photons_arr, local_dnmax_arr, local_d_index_to_ijk, local_d_table_ptr, local_dPTableTexObj, log_filename, i, local_stream, gpu_spect_ptrs[i]);
		gpuErrchk(cudaStreamSynchronize(local_stream));
		
        //Sure, after the main loop, we can free all the CUDA variables that we created here.
        cudaFree(local_d_geom);
        cudaFree(local_d_p);
        cudaFree(local_generated_photons_arr);
        cudaFree(local_dnmax_arr);
        cudaFree(local_d_table_ptr);
        cudaFree(local_d_index_to_ijk);
        cudaFreeArray(local_dPTableCuArray);
        cudaDestroyTextureObject(local_dPTableTexObj);
		cudaStreamDestroy(local_stream);
    }

	//Now merge all spectra together

	//Go back to GPU 0
	gpuErrchk(cudaSetDevice(0));

	//Have a 1D spectrum first
	struct of_spectrum *d_spect_master;
	int total_bins = N_TYPEBINS * N_THBINS * N_EBINS;
	gpuErrchk(cudaMalloc(&d_spect_master, spect_size));
	gpuErrchk(cudaMemset(d_spect_master, 0, spect_size));

	for (int i = 0; i < num_gpus; i++) {
		// Here we copy the GPU spectrum into GPU's 0 memory
		struct of_spectrum *d_spect_remote;
		gpuErrchk(cudaMalloc(&d_spect_remote, spect_size)); 
		gpuErrchk(cudaMemcpyPeer(d_spect_remote, 0, gpu_spect_ptrs[i], i, spect_size));

		int nblocks = (total_bins + N_THREADS - 1) / N_THREADS;
		AccumulateSpectrum<<<nblocks, N_THREADS>>>(d_spect_master, d_spect_remote, total_bins);
		gpuErrchk(cudaDeviceSynchronize());

		cudaFree(d_spect_remote);

		// Free the other devices other than 0 memory, since we already have the spectrum merged in GPU 0 and we want to keep VRAM clean for the final transfer back to host.
		gpuErrchk(cudaSetDevice(i));
		cudaFree(gpu_spect_ptrs[i]);
		gpuErrchk(cudaSetDevice(0));
	}

	// Now mirror exactly what we were doing before multiGPU
	struct of_spectrum ***spect = Malloc3D_Contiguous(N_TYPEBINS, N_THBINS, N_EBINS);
	gpuErrchk(cudaMemcpy(spect[0][0], d_spect_master, spect_size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&N_superph_recorded, d_N_superph_recorded, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));

	report_spectrum_h5(gen_superph, spect, params.spectrum);

	cudaFree(d_spect_master);
    free(h_generated_photons);
    free(h_dnmax_arr);
    free(h_index_to_ijk_2d);
    free(h_totals_per_gpu);
	free(h_spect_merged);
	free(gpu_spect_ptrs);
	Free3D_Contiguous(spect, N_TYPEBINS);
}

__global__ void InitializeRNGStates(const time_t time, int GPUindex) {
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int seed = GPUindex * (139 * global_index + time + GPUindex);
	init_monty_rand(seed);
}

__launch_bounds__(N_THREADS)
__global__ void generate_photons(const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, unsigned long long * __restrict__  generated_photons_arr, double * __restrict__ dnmax_arr){
	unsigned long long generated_photons;
	double dnmax;
	int i, j, k;
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
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

    get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);
	const double kappa = get_model_kappa_ijk(i, j, k, d_p);

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
			dn = int_jnu_total(Ne, Thetae, Bmag, exp(m * dlnu + lnu_min), K2, kappa) / (exp(d_wgt[m]) + 1.e-100);
			if (dn > *dnmax)
				*dnmax = dn;
			ninterp += dlnu * dn;
		}
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
	const unsigned long long photons_processed_sofar, const unsigned long long * __restrict__  index_to_ijk, int GPU_id){
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
			sample_zone_photon(i,j,k, dnmax_arr[zone_index], ph_init, d_geom, d_p, (past_zone == zone_index? 0 : 1), photon_index, Econ, Ecov, &localState, GPU_id);
			past_zone = zone_index;
			
		}
		my_curand_state[global_index] = localState;
}

__device__ void sample_zone_photon(const int i, const int j, const int k, const double dnmax, 
    struct of_photonSOA ph, const struct of_geom *  d_geom, 
    const double *  d_p, const int zone_flag, const unsigned long long ph_arr_index,
    double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState *  localState, int GPU_id)
{
    double nu, weight;
    
    // Initial setup and coordinate transformation
    {
        double Xarray[NDIM];
        coord(i, j, k, Xarray);
        // Store back immediately, don't keep Xarray alive
        ph.X0[ph_arr_index] = Xarray[0];
        ph.X1[ph_arr_index] = Xarray[1];
        ph.X2[ph_arr_index] = Xarray[2];
        ph.X3[ph_arr_index] = Xarray[3];
    } // Xarray goes out of scope here
    
    // Get fluid properties
    double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM];
    get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);


	const double kappa = get_model_kappa_ijk(i, j, k, d_p);
    // Sample frequency
    {
        const double lnu_min = log(NUMIN);
        const double Nln = log(NUMAX) - lnu_min;
		const double K2 = K2_eval(Thetae);
        do {
            nu = exp(curand_uniform_double(localState) * Nln + lnu_min);
            weight = linear_interp_weight(nu);
		}while (curand_uniform_double(localState) > (int_jnu_total(Ne, Thetae, Bmag, nu, K2, kappa) / (weight + 1.e-100)) / dnmax);
		ph.w[ph_arr_index] = weight;
    } // lnu_min, Nln go out of scope

    // Sample angles  
    double cth;
    {
		const double K2 = K2_eval(Thetae);
        const double jmax = jnu_total(nu, Ne, Thetae, Bmag, M_PI / 2., K2, kappa);
		double j_th;
		double th;
        do {
            cth = 2. * curand_uniform_double(localState) - 1.;
            th = acos(cth);
        	j_th = jnu_total(nu, Ne, Thetae, Bmag, th, K2, kappa);
        } while (curand_uniform_double(localState) > j_th / jmax);
		ph.ratio_brems[ph_arr_index] = jnu_ratio_brems(nu, Ne, Thetae, Bmag, th, K2);

    } // jmax, th, j_th go out of scope

    // Reuse arrays - use one array for both K_tetrad and final storage
    double K_data[NDIM]; // This will serve as both K_tetrad and tmpK
    
    // Build momentum vector
    {
        const double sth = sqrt(1. - cth * cth);
        const double phi = 2. * M_PI * curand_uniform_double(localState);
        const double E = nu * HPL / (ME * CL * CL);
        
        K_data[0] = E;
        K_data[1] = E * cth;
        K_data[2] = E * cos(phi) * sth;
        K_data[3] = E * sin(phi) * sth;
    } // sth, phi, E go out of scope, cphi/sphi never created
    
    // Handle tetrad if needed
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
    
    // reuse K_data array
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
	//int n = 1;
	curandState localState = my_curand_state[global_index];
	/*track each photon we created along its geodesic*/
    while (true) {
        // Each thread grabs the next available photon index atomically
        photon_index = (atomicAdd(&tracking_counter, 1) - 1);

        // If all photons are processed, exit the loop
        if (photon_index >= max_partition_ph) break;
        
		// Track the photon
        track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, 0, 0, photon_index, &localState);
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
	const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int n, unsigned long long round_num_scat_init, unsigned long long round_num_scat_end, const int bias_tuning_step){
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = my_curand_state[global_index];
	double Ne, Thetae, B;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
	unsigned long long scattering_counter_local = round_num_scat_init - 1;
	//int n_progress = 1;
	

	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		int deviceId;
		cudaGetDevice(&deviceId);
		printf("In GPU %d: Interval going from %llu to %llu in round! %d in Bias Tuning step: %d\n", deviceId, round_num_scat_init, round_num_scat_end, n, bias_tuning_step);
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
		double kappa = get_model_kappa(X, d_p);
		scatter_super_photon(ph, ph, Ne, Thetae, B, Ucon, Bcon, Gcov, kappa, &localState, scattering_counter_local);
		if (ph.w[scattering_counter_local] < 1.e-100) {	/* must have been a problem popping k back onto light cone */
			continue;
		}
		track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, round_num_scat_end, n, scattering_counter_local, &localState);
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
