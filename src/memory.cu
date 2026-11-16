/*
 * GPUmonty - memory.cu
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

void Free3D_Contiguous(struct of_spectrum ***ptr, int dim1) {
    free(ptr[0][0]);
    for (int i = 0; i < dim1; i++) {
        free(ptr[i]);
    }
    free(ptr);
}

struct of_spectrum*** Malloc3D_Contiguous(int dim1, int dim2, int dim3) {
    struct of_spectrum* flat_data = (struct of_spectrum*)malloc(dim1 * dim2 * dim3 * sizeof(struct of_spectrum));
    memset(flat_data, 0, dim1 * dim2 * dim3 * sizeof(struct of_spectrum));

    struct of_spectrum*** ptr3D = (struct of_spectrum***)malloc(dim1 * sizeof(struct of_spectrum**));
    
    for (int i = 0; i < dim1; i++) {
        ptr3D[i] = (struct of_spectrum**)malloc(dim2 * sizeof(struct of_spectrum*));
        for (int j = 0; j < dim2; j++) {
            int offset = (i * dim2 * dim3) + (j * dim3);
            ptr3D[i][j] = &flat_data[offset];
        }
    }
    return ptr3D;
}


__host__ void createdPTextureObj(cudaTextureObject_t * texObj, double * dP, cudaArray_t * cuArray) {	
	const int nw = N1; //r
	const int nx = NPRIM; //Nvar
	const int ny = N3; //phi
	const int nz = N2; //th
	float *dPf = (float *)malloc(nx * ny * nz * nw * sizeof(float));
	for (int i = 0; i < nx * ny * nz * nw; i++) {
		dPf[i] = (float)dP[i];
	}
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(cuArray, &channelDesc, make_cudaExtent(nx * ny, nz, nw), 0);
	cudaMemcpy3DParms copyParams = {0};

	copyParams.srcPtr   = make_cudaPitchedPtr((void *) dPf, nx * ny * sizeof(float), nx * ny, nz);
	copyParams.dstArray = *cuArray;
	copyParams.extent   = make_cudaExtent(nx * ny, nz, nw);
	copyParams.kind     = cudaMemcpyHostToDevice;	
	CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));

	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(texRes));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array  = *cuArray;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(texDescr));
	texDescr.normalizedCoords = false ;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	CHECK_CUDA_ERROR(cudaCreateTextureObject(texObj, &texRes, &texDescr, NULL));
	return;
}

void symbolToDevice(const void* symbol, const void* src, size_t size, cudaStream_t stream) {
    void* ptr = NULL;
    gpuErrchk(cudaGetSymbolAddress(&ptr, symbol));
    gpuErrchk(cudaMemcpyAsync(ptr, src, size, cudaMemcpyHostToDevice, stream));
}

void symbolFromDevice(void* dst, const void* symbol, size_t size, cudaStream_t stream) {
    void* ptr = NULL;
    gpuErrchk(cudaGetSymbolAddress(&ptr, symbol));
    gpuErrchk(cudaMemcpyAsync(dst, ptr, size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

__host__ void transferParams(cudaStream_t stream) {
    int Ns_int = (int) params.Ns;
    symbolToDevice(&d_Ns, &Ns_int, sizeof(int), stream);
    symbolToDevice(&d_dx, &dx, NDIM * sizeof(double), stream);
    symbolToDevice(&d_L_unit, &L_unit, sizeof(double), stream);
    symbolToDevice(&d_MBH, &params.MBH_par, sizeof(double), stream);
    symbolToDevice(&d_B_unit, &B_unit, sizeof(double), stream);
    symbolToDevice(&d_Ne_unit, &Ne_unit, sizeof(double), stream);
    if(hslope > 0)
        symbolToDevice(&d_hslope, &hslope, sizeof(double), stream);

    symbolToDevice(&d_startx, &startx, NDIM * sizeof(double), stream);
    symbolToDevice(&d_stopx, &stopx, NDIM * sizeof(double), stream);
    symbolToDevice(&d_thetae_unit, &Thetae_unit, sizeof(double), stream);
    symbolToDevice(&d_wgt, &wgt, (N_ESAMP + 1) * sizeof(double), stream);
    if(params.kappa_synch || params.powerlaw_synch){
        #if VARIABLE_KAPPA
            symbolToDevice(&d_F_nth, &F_nth, (KAPPA_NSAMP) * (N_ESAMP + 1) * sizeof(double), stream);
            // With variable kappa, when kappa > kappa_max, it goes to the thermal emission, therefore we need the thermal table.
            symbolToDevice(&d_F, &F, (N_ESAMP + 1) * sizeof(double), stream);
        #else
            symbolToDevice(&d_F_nth, &F_nth, (N_ESAMP + 1) * sizeof(double), stream);
        #endif
    }else{
        symbolToDevice(&d_F, &F, (N_ESAMP + 1) * sizeof(double), stream);
    }
    symbolToDevice(&d_nint, &nint, (NINT + 1) * sizeof(double), stream);
    symbolToDevice(&d_dndlnu_max, &dndlnu_max, (NINT + 1) * sizeof(double), stream);
    symbolToDevice(&d_K2, &K2, (N_ESAMP + 1) * sizeof(double), stream);
    symbolToDevice(&d_bias_norm, &bias_norm, sizeof(double), stream);
    symbolToDevice(&d_max_tau_scatt, &max_tau_scatt, sizeof(double), stream);
    symbolToDevice(&d_Rh, &Rh, sizeof(double), stream);
    symbolToDevice(&d_N1, &N1, sizeof(int), stream);
    symbolToDevice(&d_N2, &N2, sizeof(int), stream);
    symbolToDevice(&d_N3, &N3, sizeof(int), stream);

	cudaMemcpyToSymbol(d_scattering, &(params.scattering), sizeof(int));
	cudaMemcpyToSymbol(d_bremsstrahlung, &(params.bremsstrahlung), sizeof(int));
	cudaMemcpyToSymbol(d_thermal_synch, &(params.thermal_synch), sizeof(int));
	cudaMemcpyToSymbol(d_kappa_synch, &(params.kappa_synch), sizeof(int));
	cudaMemcpyToSymbol(d_powerlaw_synch, &(params.powerlaw_synch), sizeof(int));
    symbolToDevice(&d_scattering, &(params.scattering), sizeof(int), stream);
    symbolToDevice(&d_bremsstrahlung, &(params.bremsstrahlung), sizeof(int), stream);

    double h_bias_guess[MAX_LAYER_SCA];
    for (int i = 0; i < MAX_LAYER_SCA; i++) {
        h_bias_guess[i] = params.bias_guess;
    }
    symbolToDevice(&d_bias_guess, h_bias_guess, MAX_LAYER_SCA * sizeof(double), stream);

    symbolToDevice(&d_trat_small, &(params.trat_small), sizeof(double), stream);
    symbolToDevice(&d_trat_large, &(params.trat_large), sizeof(double), stream);
    symbolToDevice(&d_beta_crit, &(params.beta_crit), sizeof(double), stream);
    symbolToDevice(&d_thetae_max, &(params.Thetae_max), sizeof(double), stream);
    symbolToDevice(&d_tp_over_te, &(params.tp_over_te), sizeof(double), stream);
    symbolToDevice(&d_bhspin, &bhspin, sizeof(double), stream);

    #ifdef IHARM
        symbolToDevice(&d_METRIC, &METRIC, sizeof(int), stream);
        symbolToDevice(&d_gam, &gam, sizeof(double), stream);
        symbolToDevice(&d_game, &game, sizeof(double), stream);
        symbolToDevice(&d_gamp, &gamp, sizeof(double), stream);
        symbolToDevice(&d_with_electrons, &with_electrons, sizeof(int), stream);

        if(METRIC == METRIC_MKS3){
            symbolToDevice(&d_mks3R0, &mks3R0, sizeof(double), stream);
            symbolToDevice(&d_mks3H0, &mks3H0, sizeof(double), stream);
            symbolToDevice(&d_mks3MY1, &mks3MY1, sizeof(double), stream);
            symbolToDevice(&d_mks3MY2, &mks3MY2, sizeof(double), stream);
            symbolToDevice(&d_mks3MP0, &mks3MP0, sizeof(double), stream);
        } else if(METRIC == METRIC_FMKS){
            symbolToDevice(&d_poly_norm, &poly_norm, sizeof(double), stream);
            symbolToDevice(&d_poly_xt, &poly_xt, sizeof(double), stream);
            symbolToDevice(&d_poly_alpha, &poly_alpha, sizeof(double), stream);
            symbolToDevice(&d_mks_smooth, &mks_smooth, sizeof(double), stream);
        }
    #endif

    cudaStreamSynchronize(stream);
}

__host__ int setMaxBlocks(){
	int device_id;
    cudaGetDevice(&device_id);  
	int maxBlocksPerMultiprocessor;
	int numSMs;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id); 
	maxBlocksPerMultiprocessor = prop.maxBlocksPerMultiProcessor;
	numSMs = prop.multiProcessorCount;
	int max_block_number = maxBlocksPerMultiprocessor * numSMs;
    // printf("Current GPU in use: %s\n", prop.name);  // Print the GPU name
	// printf("Max number of threads per block: %d\n", maxThreadsPerBlock);
	// printf("Max number of blocks per SM: %d\n",maxBlocksPerMultiprocessor);
	// printf("number of SMs: %d\n", numSMs);
	// printf("Max number of threads per multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);

	// printf("Therefore, total number of blocks:%d\n", max_block_number );
	if(fmod(prop.maxThreadsPerMultiProcessor, N_THREADS) != 0){
		printf("WARNING: fmod(maxThreadsPerBlock, N_THREADS) != 0\n");
		printf("The number of threads per block is not a multiple of the number of threads per multiprocessor\n");
		printf("Maximum performance is achieved when you can fit whole blocks inside of SMs\n");
		exit(1);
	}

	return max_block_number;
}

__host__ void cummulativePhotonsPerZone(unsigned long long * generated_photons_arr, unsigned long long * d_index_to_ijk)
{
	unsigned long long *h_index_to_ijk = (unsigned long long *)malloc(N1 * N2 * N3 * sizeof(unsigned long long));
	unsigned long long *h_generated_photon_arr = (unsigned long long *)malloc(N1 * N2 * N3 * sizeof(unsigned long long));

	gpuErrchk(cudaMemcpy(h_generated_photon_arr, generated_photons_arr, N1 * N2 * N3* sizeof(unsigned long long ), cudaMemcpyDeviceToHost));
	h_index_to_ijk[0] = h_generated_photon_arr[0];
	for (int i = 1; i < N1 * N2 * N3; i++) {
		h_index_to_ijk[i] = h_index_to_ijk[i - 1] + h_generated_photon_arr[i];
	}

	gpuErrchk(cudaMemcpy(d_index_to_ijk, h_index_to_ijk, N1 * N2 * N3* sizeof(unsigned long long), cudaMemcpyHostToDevice));
	free(h_index_to_ijk);
	free(h_generated_photon_arr);

}

__host__ size_t photonSOASize(unsigned long long n_photons) {
    return n_photons * (20 * sizeof(double) + sizeof(int)); // X0,X1,X2,X3, K0,K1,K2,K3, dKdlam0-3, w,E,X1i,X2i, tau_abs,tau_scatt, E0,E0s, ratio_brems, (int) nscatt
}
__host__ unsigned long long photonsPerBatch(unsigned long long tot_nph, int *batch_divisions)
{
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        printf("Failed to get GPU memory info: %s\n", cudaGetErrorString(err));
    }

    if (params.targetRatio > 5) {
        int device_id;
        cudaGetDevice(&device_id);
        if (device_id == 0)
            printf("\033[1;31mWARNING: Target ratio is set to %.1f, clamping to 5.\033[0m\n", params.targetRatio);
        params.targetRatio = 5;
    }

    double ScatteringDynamicalSize = params.fitBias
        ? max(2.0 * params.targetRatio, (double)SCATTERINGS_PER_PHOTON)
        : (double)SCATTERINGS_PER_PHOTON;

    *batch_divisions = 1;

    while (true) {
        unsigned long long superph_per_batch = tot_nph / (*batch_divisions);
        unsigned long long scat_buf_size = (unsigned long long)(ScatteringDynamicalSize * superph_per_batch);

        size_t required_mem = 0;

        //GPUWorker allocations
        required_mem += photonSOASize(superph_per_batch);          // initial_photon_states

        // CurrentLayerScattering and NextLayerScattering are alive simultaneously
        // before the old current is freed. Worst case is at layer n=1 where
        // both are sized off the initial scatter count.
        if(params.scattering){
            required_mem += photonSOASize(scat_buf_size);          // CurrentLayerScattering
            required_mem += photonSOASize(scat_buf_size);              // NextLayerScattering
            required_mem += photonSOASize(scat_buf_size);              // scat_ofphoton
        }
        if (params.fitBias){
            required_mem += photonSOASize(superph_per_batch);      // PhotonStateCheckPoint
            required_mem += photonSOASize(scat_buf_size);          // ScatteredPhotonStateCheckPoint

        }

        // Add a safety margin (e.g. 20%) for CUDA runtime overhead, texture memory, etc.
        size_t required_with_margin = (size_t)(required_mem * 1.2);

        if (*batch_divisions == 1 && required_with_margin > free_mem) {
            printf("Not enough memory for %.2f GB. Available: %.2f GB. Partitioning...\n",
                   required_with_margin / (1024.0*1024.0*1024.0),
                   free_mem / (1024.0*1024.0*1024.0));
        }

        if (required_with_margin <= free_mem) {
            int current_device;
            cudaGetDevice(&current_device);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, current_device);
            printf("\033[1;34mPartitions: %d\033[0m | GPU %d (%s) | Free: %.2f GB | "
                   "Photons/partition: %llu | Est. usage: %.2f GB\n",
                   *batch_divisions, current_device, prop.name,
                   free_mem / (1024.0*1024.0*1024.0),
                   superph_per_batch,
                   required_with_margin / (1024.0*1024.0*1024.0));
            fflush(stdout);
            return superph_per_batch;
        }

        (*batch_divisions)++;
    }
}

__host__ void allocatePhotonData(struct of_photonSOA *ph, unsigned long long size) {
    gpuErrchk(cudaMalloc(&(ph->X0), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->X1), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->X2), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->X3), size * sizeof(double)));
    
    gpuErrchk(cudaMalloc(&(ph->K0), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->K1), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->K2), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->K3), size * sizeof(double)));
    
    gpuErrchk(cudaMalloc(&(ph->dKdlam0), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->dKdlam1), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->dKdlam2), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->dKdlam3), size * sizeof(double)));

    gpuErrchk(cudaMalloc(&(ph->w), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->E), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->X1i), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->X2i), size * sizeof(double)));
    
    gpuErrchk(cudaMalloc(&(ph->tau_abs), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->tau_scatt), size * sizeof(double)));
    
    gpuErrchk(cudaMalloc(&(ph->E0), size * sizeof(double)));
    gpuErrchk(cudaMalloc(&(ph->E0s), size * sizeof(double)));
    
    gpuErrchk(cudaMalloc(&(ph->nscatt), size * sizeof(int)));
	gpuErrchk(cudaMalloc(&(ph->ratio_brems), size * sizeof(double)));
}

__host__ void transferPhotonDataDevtoDev(struct of_photonSOA to, struct of_photonSOA from, unsigned long long size, cudaStream_t stream){
    gpuErrchk(cudaMemcpyAsync((to.X0), (from.X0), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.X1), (from.X1), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.X2), (from.X2), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.X3), (from.X3), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    gpuErrchk(cudaMemcpyAsync((to.K0), (from.K0), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.K1), (from.K1), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.K2), (from.K2), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.K3), (from.K3), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    gpuErrchk(cudaMemcpyAsync((to.dKdlam0), (from.dKdlam0), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.dKdlam1), (from.dKdlam1), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.dKdlam2), (from.dKdlam2), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.dKdlam3), (from.dKdlam3), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    gpuErrchk(cudaMemcpyAsync((to.w), (from.w), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.E), (from.E), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.X1i), (from.X1i), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.X2i), (from.X2i), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.tau_abs), (from.tau_abs), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.tau_scatt), (from.tau_scatt), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    gpuErrchk(cudaMemcpyAsync((to.E0), (from.E0), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.E0s), (from.E0s), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync((to.nscatt), (from.nscatt), size * sizeof(int), cudaMemcpyDeviceToDevice, stream));
	gpuErrchk(cudaMemcpyAsync((to.ratio_brems), (from.ratio_brems), size * sizeof(double), cudaMemcpyDeviceToDevice, stream));
}

__host__ void freePhotonData(struct of_photonSOA * ph){
	gpuErrchk(cudaFree(ph->X0));
	gpuErrchk(cudaFree(ph->X1));
	gpuErrchk(cudaFree(ph->X2));
	gpuErrchk(cudaFree(ph->X3));
	
	gpuErrchk(cudaFree(ph->K0));
	gpuErrchk(cudaFree(ph->K1));
	gpuErrchk(cudaFree(ph->K2));
	gpuErrchk(cudaFree(ph->K3));
	
	gpuErrchk(cudaFree(ph->dKdlam0));
	gpuErrchk(cudaFree(ph->dKdlam1));
	gpuErrchk(cudaFree(ph->dKdlam2));
	gpuErrchk(cudaFree(ph->dKdlam3));

	gpuErrchk(cudaFree(ph->w));
	gpuErrchk(cudaFree(ph->E));
	gpuErrchk(cudaFree(ph->X1i));
	gpuErrchk(cudaFree(ph->X2i));
	
	gpuErrchk(cudaFree(ph->tau_abs));
	gpuErrchk(cudaFree(ph->tau_scatt));
	
	gpuErrchk(cudaFree(ph->E0));
	gpuErrchk(cudaFree(ph->E0s));
	
	gpuErrchk(cudaFree(ph->nscatt));
	gpuErrchk(cudaFree(ph->ratio_brems));
}