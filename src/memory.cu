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

__host__ void create1DTextureObj(cudaTextureObject_t * texObj, double * ptr, cudaArray_t * cudaArray){
	int width = N_ESAMP + 1;
	float float_table[width];
	for (int i = 0; i < width; i++) {
		float_table[i] = (float)ptr[i];
	}


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CHECK_CUDA_ERROR(cudaMallocArray(cudaArray, &channelDesc, width, 1));

	// Copy data from host to device
    size_t spitch = width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(*cudaArray, 0, 0, float_table, spitch, spitch, 1, cudaMemcpyHostToDevice));
	
	// Specify texture object parameters
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *cudaArray;
	 // Specify texture description parameters
	 struct cudaTextureDesc texDesc;
	 memset(&texDesc, 0, sizeof(texDesc));
	 texDesc.addressMode[0] = cudaAddressModeClamp;
	 texDesc.filterMode = cudaFilterModeLinear;
	 texDesc.readMode = cudaReadModeElementType;
	 texDesc.normalizedCoords = 0;  // Using non-normalized coordinates
 
	 // Create texture object
	 CHECK_CUDA_ERROR(cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL));
	 // Destroy texture object and free CUDA array
	 //CHECK_CUDA_ERROR(cudaFreeArray(cuArray));
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


__host__ void transferParams() {
	int Ns_int = (int) params.Ns;
    cudaMemcpyToSymbol(d_Ns, &Ns_int, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, NDIM * sizeof(double));

	if(hslope > 0)
	cudaMemcpyToSymbol(d_hslope, &hslope,sizeof(double));
	
	cudaMemcpyToSymbol(d_startx, &startx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_stopx, &stopx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_thetae_unit, &Thetae_unit, sizeof(double));
	cudaMemcpyToSymbol(d_wgt, &wgt, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_F, &F, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_nint, &nint, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_dndlnu_max, &dndlnu_max, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_K2, &K2, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_bias_norm, &bias_norm, sizeof(double));
	cudaMemcpyToSymbol(d_max_tau_scatt, &max_tau_scatt, sizeof(double));
	cudaMemcpyToSymbol(d_Rh, &Rh, sizeof(double));
	cudaMemcpyToSymbol(d_N1, &N1, sizeof(int));
	cudaMemcpyToSymbol(d_N2, &N2, sizeof(int));
	cudaMemcpyToSymbol(d_N3, &N3, sizeof(int));


	/*iharm variables*/
	cudaMemcpyToSymbol(d_scattering, &(params.scattering), sizeof(int));
	cudaMemcpyToSymbol(d_biastuning, &(params.biasTuning), sizeof(double));

	cudaMemcpyToSymbol(d_trat_small, &(params.trat_small), sizeof(double));
	cudaMemcpyToSymbol(d_trat_large, &(params.trat_large), sizeof(double));
	cudaMemcpyToSymbol(d_beta_crit, &(params.beta_crit), sizeof(double));
	cudaMemcpyToSymbol(d_thetae_max, &(params.Thetae_max), sizeof(double));
	cudaMemcpyToSymbol(d_tp_over_te, &(params.tp_over_te), sizeof(double));
	cudaMemcpyToSymbol(d_bhspin, &bhspin, sizeof(double));
}

__host__ int setMaxBlocks(){
	int device_id;
    cudaGetDevice(&device_id);  
	int maxThreadsPerBlock;
	int maxBlocksPerMultiprocessor;
	int numSMs;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id); 
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	maxBlocksPerMultiprocessor = prop.maxBlocksPerMultiProcessor;
	numSMs = prop.multiProcessorCount;
	int max_block_number = maxBlocksPerMultiprocessor * numSMs;
    printf("Current GPU in use: %s\n", prop.name);  // Print the GPU name
	printf("Max number of threads per block: %d\n", maxThreadsPerBlock);
	printf("Max number of blocks per SM: %d\n",maxBlocksPerMultiprocessor);
	printf("number of SMs: %d\n", numSMs);
	printf("Max number of threads per multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);

	printf("Therefore, total number of blocks:%d\n", max_block_number );
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

	cudaMemcpyErrorCheck(h_generated_photon_arr, generated_photons_arr, N1 * N2 * N3* sizeof(unsigned long long ), cudaMemcpyDeviceToHost);
	h_index_to_ijk[0] = h_generated_photon_arr[0];
	for (int i = 1; i < N1 * N2 * N3; i++) {
		h_index_to_ijk[i] = h_index_to_ijk[i - 1] + h_generated_photon_arr[i];
	}

	cudaMemcpyErrorCheck(d_index_to_ijk, h_index_to_ijk, N1 * N2 * N3* sizeof(unsigned long long), cudaMemcpyHostToDevice);
	free(h_index_to_ijk);
	free(h_generated_photon_arr);

}



__host__ unsigned long long photonsPerBatch(unsigned long long tot_nph, int * batch_divisions)
{
	size_t free_mem, total_mem;
	cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
	if (err != cudaSuccess) {
		printf("Failed to get GPU memory info: %s\n", cudaGetErrorString(err));
	}
    size_t required_mem ;
	required_mem = tot_nph * sizeof(struct of_photon);
	required_mem += MAX_LAYER_SCA *  tot_nph * SCATTERINGS_PER_PHOTON * sizeof(struct of_photon);
	if (required_mem > free_mem) {
		printf("Not enough memory to allocate %.2lf GB for photon states. Available memory: %.2lf GB\n", required_mem / 1e9, free_mem / 1e9);
		printf("Beginning equipartion of photons...\n");
    }
	unsigned long long superph_per_batch = tot_nph;
	*batch_divisions = 1;

	
	while (required_mem > free_mem) {
		superph_per_batch = tot_nph/(*batch_divisions);
		required_mem = superph_per_batch * sizeof(struct of_photon);
		required_mem += MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * superph_per_batch * sizeof(struct of_photon);
		*batch_divisions = *batch_divisions + 1;
	}
	printf("\033[1;34mRequired partitions: %d\033[0m. Number of photons per partition: %d\n", *batch_divisions, (int)(tot_nph/(*batch_divisions)));
	return (unsigned long long)(tot_nph/(*batch_divisions));
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
}

__host__ void transferPhotonDataDevtoDev(struct of_photonSOA to, struct of_photonSOA from, unsigned long long size){
	cudaMemcpyErrorCheck((from.X0), (to.X0), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.X1), (to.X1), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.X2), (to.X2), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.X3), (to.X3), size * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMemcpyErrorCheck((from.K0), (to.K0), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.K1), (to.K1), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.K2), (to.K2), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.K3), (to.K3), size * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMemcpyErrorCheck((from.dKdlam0), (to.dKdlam0), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.dKdlam1), (to.dKdlam1), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.dKdlam2), (to.dKdlam2), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.dKdlam3), (to.dKdlam3), size * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMemcpyErrorCheck((from.w), (to.w), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.E), (to.E), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.X1i), (to.X1i), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.X2i), (to.X2i), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.tau_abs), (to.tau_abs), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.tau_scatt), (to.tau_scatt), size * sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMemcpyErrorCheck((from.E0), (to.E0), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.E0s), (to.E0s), size * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpyErrorCheck((from.nscatt), (to.nscatt), size * sizeof(int), cudaMemcpyDeviceToDevice);
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
}