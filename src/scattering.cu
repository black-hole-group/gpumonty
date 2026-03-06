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

// __global__ void check(struct of_photonSOA ph_old, struct of_photonSOA ph_new){
// 	unsigned long long test = 39283;
// 	printf("ph_old.K1[photon_index], ph_new.K1[photon_index] = %le, %le\n", ph_old.K1[test], ph_new.K1[test]);
// }
__host__ void scattering_flow_control(unsigned long long num_scat_phs[MAX_LAYER_SCA], struct of_photonSOA scat_ofphoton, struct of_spectrum *d_spect, unsigned long long instant_photon_number, int max_block_number, cudaTextureObject_t besselTexObj, double *d_table_ptr, double * d_p){
	/*Perform scattering loop*/
		int n = 1;
		bool quit_flag_sca = false;
		unsigned long long scatterings_performed = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

		struct of_photonSOA ScatteredPhotonStateCheckPoint;
		if(params.fitBias){
			allocatePhotonData(&ScatteredPhotonStateCheckPoint, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * instant_photon_number);
			transferPhotonDataDevtoDev(ScatteredPhotonStateCheckPoint, scat_ofphoton, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON *instant_photon_number);
		}

		struct of_photonSOA CurrentLayerScattering;
		allocatePhotonData(&CurrentLayerScattering, num_scat_phs[n-1]);
		// Transfer how much was scattered in the first round to CurrentLayerScattering
		transferPhotonDataDevtoDev(CurrentLayerScattering, scat_ofphoton, num_scat_phs[n-1]);
		freePhotonData(&scat_ofphoton);
		
		while(quit_flag_sca == false && n < MAX_LAYER_SCA){
			struct of_photonSOA NextLayerScattering;
			allocatePhotonData(&NextLayerScattering, SCATTERINGS_PER_PHOTON *num_scat_phs[n-1]);

			if(!params.scattering){
				break;
			}
			printf("\nStarting round %d\n", n);
			int ideal_nblocks = (int)ceil((double) num_scat_phs[n-1] / (double) N_THREADS);

			int RedoTuning = 1;
			double InferiorAcceptance = 0.8 * params.targetRatio;
			double SuperiorAcceptance = 1.2 * params.targetRatio;
			int BiasTuning_index = 0;
			do{
				cudaEventRecord(start, 0);
				if(ideal_nblocks > max_block_number){
					#ifdef DO_NOT_USE_TEXTURE_MEMORY
						track_scat<<<max_block_number,N_THREADS>>>(CurrentLayerScattering, d_p, d_table_ptr, NextLayerScattering, n, besselTexObj, 0, num_scat_phs[n-1]);
					#else
						track_scat<<<max_block_number,N_THREADS>>>(CurrentLayerScattering, dPTableTexObj, d_table_ptr, NextLayerScattering, n, besselTexObj, 0, num_scat_phs[n-1]);
					#endif
				}else{
					if (ideal_nblocks == 0)
						ideal_nblocks = 1;
					#ifdef DO_NOT_USE_TEXTURE_MEMORY
						track_scat<<<ideal_nblocks,N_THREADS>>>(CurrentLayerScattering, d_p, d_table_ptr, NextLayerScattering, n, besselTexObj, 0, num_scat_phs[n-1]);
					#else
						track_scat<<<ideal_nblocks,N_THREADS>>>(CurrentLayerScattering, dPTableTexObj, d_table_ptr, NextLayerScattering, n, besselTexObj, 0, num_scat_phs[n-1]);
					#endif
				}
				cudaDeviceSynchronize();
				cudaEventRecord(stop);
				cudaEventSynchronize(stop); 
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);
				Flag("the track_scat kernel");

				printf("Scattering kernel, round %d, execution time: %f s\n", n,milliseconds/1000.);
				cudaMemcpyFromSymbol(&num_scat_phs[n], d_num_scat_phs, sizeof(unsigned long long), n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
				printf("number of scattered photons generated = %llu in round %d\n", num_scat_phs[n], n);

				if(params.fitBias){
					double Ratio = (double) num_scat_phs[n] / (double) num_scat_phs[n-1];
					BiasTuning_index++;
					if((Ratio < InferiorAcceptance || Ratio > SuperiorAcceptance) && BiasTuning_index < MAXITER_BIASTUNING){
						printf("\033[1;31mIn round %d, Ratio of Scattering/Created is %.3e, should be in the interval[%.3e, %.3e] \033[0m\n", n, Ratio, InferiorAcceptance, SuperiorAcceptance);
						if(Ratio == 0.){
							Ratio = 2e-1; // To avoid division by zero, multiply bias factor by 2 times the target ratio
						}
						params.biasTuning *= params.targetRatio/Ratio;
						printf("\033[1;31mTrying new BiasTuning parameter %.3e\n", params.biasTuning);
						//cudaMemcpyToSymbol(d_biastuning[n], &(params.biasTuning), MAX_LAYER_SCA * sizeof(double));
						cudaMemcpyToSymbol(d_biastuning, &(params.biasTuning), sizeof(double), n * sizeof(double));
						transferPhotonDataDevtoDev(ScatteredPhotonStateCheckPoint, CurrentLayerScattering, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * instant_photon_number);
					}else{
						RedoTuning = 0;
						if(BiasTuning_index < MAXITER_BIASTUNING){
							printf("\033[1;32mBias Found! Ratio of Scattering/Created is %.3e, should be in the interval[%.3e, %.3e]\033[0m\n",  Ratio, InferiorAcceptance, SuperiorAcceptance);
						}else{
							printf("\033[1;33mBias Tuning limit reached! Latest Ratio is going to be considered.\033[0m\n");
						}
					}
				}
			}while(params.fitBias && (RedoTuning && BiasTuning_index < MAXITER_BIASTUNING));
			
			if(params.fitBias) freePhotonData(&ScatteredPhotonStateCheckPoint);
			
			if(ideal_nblocks > max_block_number){
				record_scattering<<<max_block_number,N_THREADS>>>(CurrentLayerScattering, d_spect, instant_photon_number, max_block_number, n);
			}else{
				if (ideal_nblocks == 0)
				ideal_nblocks = 1;
				record_scattering<<<ideal_nblocks,N_THREADS>>>(CurrentLayerScattering, d_spect, instant_photon_number, ideal_nblocks, n);
			}			
			Flag("the recording_scattering kernel");


			// Copy from NextLayerScattering to CurrentLayerScattering for the next round
			freePhotonData(&CurrentLayerScattering);
			//Copy data from NextLayerScattering to CurrentLayerScattering
			allocatePhotonData(&CurrentLayerScattering,  num_scat_phs[n]);
			transferPhotonDataDevtoDev(CurrentLayerScattering, NextLayerScattering, num_scat_phs[n]);
			freePhotonData(&NextLayerScattering);
			
			cudaMemcpyFromSymbol(&scatterings_performed, scattering_counter, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
            if(num_scat_phs[n] == 0){
				printf("Quit flag reached in round %d!\n", n);
				quit_flag_sca = true;
			}
            unsigned long long reset = 0;
			cudaMemcpyToSymbol(scattering_counter, &reset, sizeof(unsigned long long));
			n++;
		}
		if(n >= MAX_LAYER_SCA && num_scat_phs[n] > 0){
			printf("WARNING: Maximum number of scattering layers reached. Some photons may not have been accounted for.\n");
		}

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in scattering kernerls %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
		freePhotonData(&CurrentLayerScattering);
        memset(num_scat_phs, 0, MAX_LAYER_SCA * sizeof(unsigned long long));
        cudaMemcpyToSymbol(d_num_scat_phs, num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
}