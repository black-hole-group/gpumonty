/*
Summary of the file: This file contain memory related functions that are used in the GPU code.
*/
#include "decs.h"
#include "kernels.h"
#include "memory.h"


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
	//cudaMemcpy3D(&copyParams);
	//Array creation End

	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(texRes));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array  = *cuArray;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(texDescr));
	texDescr.normalizedCoords = false ;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	CHECK_CUDA_ERROR(cudaCreateTextureObject(texObj, &texRes, &texDescr, NULL));
	return;
}
__host__ void createTableTextureObj(cudaTextureObject_t * texObj, double table[][NT + 1], const int width, const int height, cudaArray_t * cuArray) {	
	float float_table[height][width];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float_table[i][j] = (float)table[i][j];
		}
	}


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CHECK_CUDA_ERROR(cudaMallocArray(cuArray, &channelDesc, width, height));

	// Copy data from host to device
    size_t spitch = width * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(*cuArray, 0, 0, float_table, spitch, 
                                         width * sizeof(float), height, 
                                         cudaMemcpyHostToDevice));
	
	// Specify texture object parameters
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *cuArray;
	 // Specify texture description parameters
	 struct cudaTextureDesc texDesc;
	 memset(&texDesc, 0, sizeof(texDesc));
	 texDesc.addressMode[0] = cudaAddressModeClamp;
	 texDesc.addressMode[1] = cudaAddressModeClamp;
	 texDesc.filterMode = cudaFilterModeLinear;
	 texDesc.readMode = cudaReadModeElementType;
	 texDesc.normalizedCoords = 0;  // Using non-normalized coordinates
 
	 // Create texture object
	 CHECK_CUDA_ERROR(cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL));
	 // Destroy texture object and free CUDA array
	 //CHECK_CUDA_ERROR(cudaFreeArray(cuArray));
}


__host__ void transferGlobalVariables(){
	/*
	This function will transfer the global variables to the GPU.

	Variables:
    @Ns - number of superphotons parameter
    @N1 - number of zones in the r direction
    @N2 - number of zones in the ph direction
    @N3 - number of zones in the time direction
    @dx - size of the zones
    @hslope - slope of the theta to x2 conversion
    @startx - starting x value
    @stopx - stopping x value
    @a - black hole spin
    @Thetae_unit - unit of the dimensionless electron temperature
    @wgt - array of weights for the superphoton distribution
    @F - array of F values for the superphoton distribution
    @nint - array of nint values for the superphoton distribution
    @dndlnu_max - array of dndlnu_max values for the superphoton distribution
    @K2 - array of second kind modified bessel function K2 values for the superphoton distribution
    @bias_norm - normalization factor for the bias function
    @max_tau_scatt - maximum scattering optical depth
    @Rh - radius of theevent horizon

	*/
    cudaMemcpyToSymbol(d_Ns, &Ns, sizeof(int));
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
	//cudaDeviceSetLimit(cudaLimitStackSize, 8000);
}

__host__ int setMaxBlocks(){
	/*
	This function will output the specs of the GPU in use. It will return the maximum number of blocks that can be used in the GPU.

	Variables:
	@device_id - the id of the GPU in use in case of multiple GPU systems
	@maxThreadsPerBlock - the maximum number of threads per block
	@maxBlocksPerMultiprocessor - the maximum number of blocks per multiprocessor
	@numSMs - the number of SMs in the GPU
	@max_block_number - the maximum number of blocks that can be used in the GPU

	Observations:
	This calculation depends on the GPU.
	The first thing you wish to do is maximize the number of threads per block in a way that you can fit as many blocks as possible in the SMs.
	For example, in case you have a GPU that fits 1536 threads per SM, 1024 threads per block is not a good idea because you end up underutilizing the SM (since you will only fit one block).
	You should aim for 512 threads per block, so you can fit 3 blocks in the SM and utilize the maximum number of threads in the SM.

	Deciding how many blocks to fit the GPU is an art and you should test it out yourself. The more blocks you fit, the more you can parallelize the code, but you may end up with a lot of overhead.
	*/
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
	/*
	Create a cummulative sum array to be used in the sampling of photons

	Parameters:
	@generated_photons_arr - array containing the number of photons to be generated in each zone
	@d_index_to_ijk - array containing the cummulative sum of the number of photons to be generated in each zone

	Variables:
	@nph - number of photons to be generated in each

	*/

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
	/*
	Estimate the number of photons to be generated per batch

	Paremeters:
	@tot_nph - total number of photons to be generated (in all partitions)

	Variables:
	@free_mem - free memory on the GPU
	@required_mem - memory required to store the photons in the GPU
	@superph_per_batch - number of photons to be generated per batch
	@batch_divisions - number of batches to divide the photons into

	*/

	size_t free_mem, total_mem;
	cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
	if (err != cudaSuccess) {
		printf("Failed to get GPU memory info: %s\n", cudaGetErrorString(err));
	}
    size_t required_mem ;
	required_mem = tot_nph * sizeof(struct of_photon);
	required_mem += MAX_LAYER_SCA *  tot_nph * sizeof(struct of_photon);
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
	printf("Required partitions: %d. Number of photons per partition: %d\n", *batch_divisions, (int)(tot_nph/(*batch_divisions)));

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