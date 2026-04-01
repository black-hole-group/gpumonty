/*
 * GPUmonty - scattering.cu
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
#include "kernels.h"
#include "memory.h"


__host__ void scattering_flow_control(unsigned long long num_scat_phs[MAX_LAYER_SCA], struct of_photonSOA *scat_ofphoton, struct of_spectrum *d_spect, unsigned long long instant_photon_number, int max_block_number, double *d_table_ptr, double * d_p, cudaTextureObject_t dPTableTexObj, cudaStream_t local_stream, double *local_bias_guess, FILE *log_file){
	/*Perform scattering loop*/
		int n = 1;
		bool quit_flag_sca = false;
		unsigned long long scatterings_performed = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
		bool current_layer_freed = false;

		struct of_photonSOA CurrentLayerScattering;
		allocatePhotonData(&CurrentLayerScattering, num_scat_phs[n-1]);
		// Transfer how much was scattered in the first round to CurrentLayerScattering
		transferPhotonDataDevtoDev(CurrentLayerScattering, *scat_ofphoton, num_scat_phs[n-1]);
		freePhotonData(scat_ofphoton);
		
		while(quit_flag_sca == false && n < MAX_LAYER_SCA){
			struct of_photonSOA NextLayerScattering;
			if(params.fitBias){
				double ScatteringDynamicalSize = max(2.0 *  params.targetRatio, (double) SCATTERINGS_PER_PHOTON);
				allocatePhotonData(&NextLayerScattering, ScatteringDynamicalSize * num_scat_phs[n-1]);
			}else{
				allocatePhotonData(&NextLayerScattering, SCATTERINGS_PER_PHOTON *num_scat_phs[n-1]);
			}

			if(!params.scattering){
				freePhotonData(&NextLayerScattering);
				freePhotonData(&CurrentLayerScattering);
				current_layer_freed = true;
				break;
			}
			fprintf(log_file, "\nStarting round %d\n", n);
			int ideal_nblocks = (int)ceil((double) num_scat_phs[n-1] / (double) N_THREADS);

			struct of_photonSOA ScatteredPhotonStateCheckPoint;
			if(params.fitBias){
				allocatePhotonData(&ScatteredPhotonStateCheckPoint, num_scat_phs[n-1]);
				transferPhotonDataDevtoDev(ScatteredPhotonStateCheckPoint, CurrentLayerScattering, num_scat_phs[n-1]);
				gpuErrchk(cudaMemcpyFromSymbol(local_bias_guess, d_bias_guess, sizeof(double), n * sizeof(double)));
				fprintf(log_file, "Using bias_guess parameter %.3e for the scattering round %d\n", *local_bias_guess, n);
			}
			
			int RedoTuning = 1;
			double InferiorAcceptance = 0.8 * params.targetRatio;
			double SuperiorAcceptance = 1.2 * params.targetRatio;
			int BiasTuning_index = 0;
			double PreviousRatio = 0;
			do{
				cudaEventRecord(start, 0);
				track_scat<<<min(ideal_nblocks, max_block_number),N_THREADS, 0, local_stream>>>(CurrentLayerScattering,
					#ifdef DO_NOT_USE_TEXTURE_MEMORY
					 d_p,
					#else
					 dPTableTexObj,
					#endif
					d_table_ptr, NextLayerScattering, n, 0, num_scat_phs[n-1], BiasTuning_index);
				cudaStreamSynchronize(local_stream);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop); 
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);
				Flag("the track_scat kernel");

				fprintf(log_file, "Scattering kernel, round %d, execution time: %f s\n", n,milliseconds/1000.);
				cudaMemcpyFromSymbol(&num_scat_phs[n], d_num_scat_phs, sizeof(unsigned long long), n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
				fprintf(log_file, "number of scattered photons generated = %llu in round %d\n", num_scat_phs[n], n);
				#pragma omp atomic
				N_scatt += num_scat_phs[n];

				if(params.fitBias){
					double Ratio = (double) num_scat_phs[n] / (double) num_scat_phs[n-1];
					double RelativeImprovement = abs(Ratio - PreviousRatio)/PreviousRatio;
					// In case there haven't had scattering in layer 0;
					if(isnan(Ratio))Ratio = 0.1;
					BiasTuning_index++;
					if((Ratio < InferiorAcceptance || Ratio > SuperiorAcceptance) && BiasTuning_index < MAXITER_BIASTUNING && RelativeImprovement > 0.1){

						fprintf(log_file, "\033[1;31mIn round %d, Ratio of Scattering/Created is %.3e, should be in the interval[%.3e, %.3e] \033[0m\n", n, Ratio, InferiorAcceptance, SuperiorAcceptance);
						if(Ratio == 0.){
							Ratio = 2e-1; // To avoid division by zero, multiply bias factor by 2 times the target ratio
						}
						*local_bias_guess *= params.targetRatio/Ratio;
						fprintf(log_file, "\033[1;31mTrying new BiasTuning parameter %.3e \033[0m\n", *local_bias_guess);
						gpuErrchk(cudaMemcpyToSymbol(d_bias_guess, local_bias_guess, sizeof(double), n * sizeof(double)));
						// Turn scattering_counter to zero, since we are going to retrack the same photons with a different bias parameter
						unsigned long long reset = 0;
						gpuErrchk(cudaMemcpyToSymbol(scattering_counter, &reset, sizeof(unsigned long long)));
						transferPhotonDataDevtoDev(CurrentLayerScattering, ScatteredPhotonStateCheckPoint, num_scat_phs[n-1]);
					}else{
						RedoTuning = 0;
						if(RelativeImprovement <= 0.1){
							fprintf(log_file, "\033[1;33mNo improvement found by enhancing the biasguess, medium is too optically thin \033[0m\n");
						}else if(BiasTuning_index < MAXITER_BIASTUNING){
							fprintf(log_file, "\033[1;32mBias Found! Ratio of Scattering/Created is %.3e, Relative Improvement: %.3e\033[0m\n",  Ratio, RelativeImprovement);
						}else{
							fprintf(log_file, "\033[1;33mBias Tuning limit reached! Latest Ratio is going to be considered.\033[0m\n");
						}
					}
					PreviousRatio = Ratio;
				}
			}while(params.fitBias && (RedoTuning && BiasTuning_index < MAXITER_BIASTUNING));
			
			if(params.fitBias) freePhotonData(&ScatteredPhotonStateCheckPoint);

			record_scattering<<<min(ideal_nblocks, max_block_number),N_THREADS, 0, local_stream>>>(CurrentLayerScattering, d_spect, instant_photon_number, min(ideal_nblocks, max_block_number), n);		
			Flag("the recording_scattering kernel");


			// Copy from NextLayerScattering to CurrentLayerScattering for the next round
			freePhotonData(&CurrentLayerScattering);
			//Copy data from NextLayerScattering to CurrentLayerScattering
			allocatePhotonData(&CurrentLayerScattering,  num_scat_phs[n]);
			transferPhotonDataDevtoDev(CurrentLayerScattering, NextLayerScattering, num_scat_phs[n]);
			freePhotonData(&NextLayerScattering);
			
			cudaMemcpyFromSymbol(&scatterings_performed, scattering_counter, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
            if(num_scat_phs[n] == 0){
				fprintf(log_file, "Quit flag reached in round %d!\n", n);
				quit_flag_sca = true;
			}
            unsigned long long reset = 0;
			cudaMemcpyToSymbol(scattering_counter, &reset, sizeof(unsigned long long));
			n++;
		}

		if(n >= MAX_LAYER_SCA && num_scat_phs[n - 1] > 0){
			fprintf(log_file, "WARNING: Maximum number of scattering layers reached. Some photons may not have been accounted for.\n");
		}

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(log_file, "in scattering kernerls %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
		if(!current_layer_freed)
			freePhotonData(&CurrentLayerScattering);
		
        memset(num_scat_phs, 0, MAX_LAYER_SCA * sizeof(unsigned long long));
}