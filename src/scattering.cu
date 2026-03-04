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

__host__ void scattering_flow_control(unsigned long long num_scat_phs[MAX_LAYER_SCA], struct of_photonSOA scat_ofphoton, struct of_spectrum *d_spect, unsigned long long instant_photon_number, int max_block_number, cudaTextureObject_t besselTexObj, double *d_table_ptr, double * d_p){
	/*Perform scattering loop*/
		int n = 1;
		bool quit_flag_sca = false;
		unsigned long long scatterings_performed = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
		while(quit_flag_sca == false && n < MAX_LAYER_SCA){
			if(!params.scattering){
				break;
			}
			printf("\nStarting round %d\n", n);
			int ideal_nblocks = (int)ceil((double) num_scat_phs[n-1] / (double) N_THREADS);
			unsigned long long round_num_scat_init = 0;

			for (int cum_sum = 0; cum_sum < n -1; cum_sum++){
				round_num_scat_init += num_scat_phs[cum_sum]; 
			}
			unsigned long long round_num_scat_end = round_num_scat_init + num_scat_phs[n-1];
			
			cudaEventRecord(start, 0);
			if(ideal_nblocks > max_block_number){
				#ifdef DO_NOT_USE_TEXTURE_MEMORY
					track_scat<<<max_block_number,N_THREADS>>>(scat_ofphoton, d_p, d_table_ptr, scat_ofphoton, n, besselTexObj, round_num_scat_init, round_num_scat_end);
				#else
					track_scat<<<max_block_number,N_THREADS>>>(scat_ofphoton, dPTableTexObj, d_table_ptr, scat_ofphoton, n, besselTexObj, round_num_scat_init, round_num_scat_end);
				#endif
			}else{
				if (ideal_nblocks == 0)
					ideal_nblocks = 1;
				#ifdef DO_NOT_USE_TEXTURE_MEMORY
					track_scat<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, d_p, d_table_ptr, scat_ofphoton, n, besselTexObj, round_num_scat_init, round_num_scat_end);
				#else
					track_scat<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, dPTableTexObj, d_table_ptr, scat_ofphoton, n, besselTexObj, round_num_scat_init, round_num_scat_end);
				#endif
			}
			cudaDeviceSynchronize();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop); 
            float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Scattering kernel, round %d, execution time: %f s\n", n,milliseconds/1000.);
			cudaError_t cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "in track_scat %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}

			if(ideal_nblocks > max_block_number){
				record_scattering<<<max_block_number,N_THREADS>>>(scat_ofphoton, d_spect, instant_photon_number, max_block_number, n);
			}else{
				if (ideal_nblocks == 0)
				ideal_nblocks = 1;
				record_scattering<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, d_spect, instant_photon_number, ideal_nblocks, n);
			}			
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "in record_scattering %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}
			
			cudaMemcpyFromSymbol(&scatterings_performed, scattering_counter, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
			//cudaMemcpyFromSymbol(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA* sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
			cudaMemcpyFromSymbol(num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA* sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
            if(num_scat_phs[n] == 0){
				printf("Quit flag reached in round %d!\n", n);
				quit_flag_sca = true;
			}
            unsigned long long reset = 0;
			cudaMemcpyToSymbol(scattering_counter, &reset, sizeof(unsigned long long));
			printf("number of scattered photons generated = %llu in round %d\n", num_scat_phs[n], n);
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
        
		freePhotonData(&scat_ofphoton);
        memset(num_scat_phs, 0, MAX_LAYER_SCA * sizeof(unsigned long long));
		//cudaMemcpyToSymbol(d_num_scat_phs, &num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_num_scat_phs, num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
}