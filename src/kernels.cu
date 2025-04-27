

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
// __global__ void compare_array_to_texture(cudaTextureObject_t texObj, const int size){
// 	for(int i = 0; i < size; i++){
// 		float var = tex1D<float>(texObj, (i + 0. + 0.5f));
// 		printf("Error in position %d, Og: %f, tex: %f\n", i, (d_K2[i] + d_K2[i + 1])/2, var);
// 	}
// }

__host__ void mainFlowControl(time_t time, double * p, const char * filename){
	/*
	Launches the kernels that will generate the photons and sample them. It will also track the photons along the geodesics and solve the scattered photons.

	Parameters:
	@time: This is the usual C function that returns the number of seconds since the epoch. It is used to seed the random number generator
	@p: Array of the primitive variables at each grid cell
	@filename: Name of the file where the spectrum will be saved

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

	transferGlobalVariables();

	struct of_spectrum spect[N_THBINS][N_EBINS] = { };
    struct of_spectrum* d_spect;
    gpuErrchk(cudaMalloc((void**)&d_spect, N_THBINS * N_EBINS * sizeof(struct of_spectrum)));
	double * d_p; 
    gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
    cudaMemcpyErrorCheck(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice);

	cudaTextureObject_t dPTableTexObj = 0;
	cudaArray_t dPTableCuArray;
	createdPTextureObj(&dPTableTexObj, p, &dPTableCuArray);
	free(p);

	
	double *d_table_ptr;
    gpuErrchk(cudaMalloc((void**)&d_table_ptr, (NW + 1) * (NT + 1) * sizeof(double)));
    cudaMemcpyErrorCheck(d_table_ptr, table, (NW + 1) * (NT + 1) * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaTextureObject_t besselTexObj = 0;
	cudaArray_t besselCuArray;
	create1DTextureObj(&besselTexObj, K2, &besselCuArray);
	// compare_array_to_texture<<<1,1>>>(besselTexObj, N_ESAMP + 1);
	// cudaDeviceSynchronize();
	// cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "in compare_array_to_texture %s\n", cudaGetErrorString(cudaStatus));
	// 	exit(1);
	// }
	// exit(1);
	// cudaTextureObject_t tableTexObj = 0;
	// cudaArray_t cuArray;
	// createTableTextureObj(&tableTexObj, table, NT + 1, NW + 1, &cuArray);
	


	struct of_geom *d_geom;
	gpuErrchk(cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom)));
    cudaMemcpyErrorCheck(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice);



	int max_block_number = setMaxBlocks();

	unsigned long long gen_superph = 0;
	unsigned long long * generated_photons_arr;
	gpuErrchk(cudaMalloc(&generated_photons_arr, N1 * N2 * N3 * sizeof(unsigned long long)));
	double * dnmax_arr;
	gpuErrchk(cudaMalloc(&dnmax_arr, N1 * N2 * N3 * sizeof(double)));

	fprintf(stderr, "Generating super photons!\n");
	cudaEventRecord(start, 0);
    GPU_generate_photons<<<N_BLOCKS,N_THREADS>>>(d_geom, d_p, time, generated_photons_arr, dnmax_arr, besselTexObj);	
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
	unsigned long long reset = 0;
	int ideal_nblocks;
	int batch_divisions = 1;
	instant_photon_number = photonsPerBatch(gen_superph, &batch_divisions);



	while(instant_partition <= batch_divisions){
		printf("Starting partition %d out of %d\n", instant_partition, batch_divisions);

		//If in the last partition and there is an offset, just do it;
		if(instant_partition == batch_divisions){
			offset = gen_superph % batch_divisions;
			printf("Last partition with an offset of =%d\n", offset);
		}
		instant_photon_number = (unsigned long long)((gen_superph/batch_divisions) + offset);
		printf("Superphotons processed so far %llu. Superphotons to be processed in this batch %llu\n", photons_processed, instant_photon_number);
		ideal_nblocks = ceil(instant_photon_number/N_THREADS);
		if(ideal_nblocks == 0){
			ideal_nblocks = 1;
		}

		allocatePhotonData(&initial_photon_states, instant_photon_number);
		allocatePhotonData(&scat_ofphoton, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON *instant_photon_number);

		//gpuErrchk(cudaMalloc(&initial_photon_states, instant_photon_number* sizeof(struct of_photon)));
		//gpuErrchk(cudaMalloc(&scat_ofphoton, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * instant_photon_number* sizeof(struct of_photon))); 

		fprintf(stderr, "Sampling the photons!\n");
		cudaEventRecord(start, 0);
		if(ideal_nblocks > max_block_number){
			GPU_sample_photons_batch<<<N_BLOCKS,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk, besselTexObj);
		}else{
			if (ideal_nblocks == 0)
			ideal_nblocks = 1;
			GPU_sample_photons_batch<<<N_BLOCKS,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk, besselTexObj);
		}
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Sampling kernel execution time: %f s\n", milliseconds/1000.);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in GPU_sample_photons_batch %s, partition (%d)\n", cudaGetErrorString(cudaStatus), instant_partition);
			fprintf(stderr, "If the error is invalid memory location, there is probably too much scattering photons, try changing the bias function.\n");
			exit(1);
		}

		cudaMemcpyToSymbol(tracking_counter_sampling, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
		fprintf(stderr, "Photon sampling process completed!\n");


		fprintf(stderr, "Tracking photons along the geodesics\n");
		cudaEventRecord(start, 0);
		if(ideal_nblocks > max_block_number){
			GPU_track<<<max_block_number,N_THREADS>>>(initial_photon_states, dPTableTexObj, d_table_ptr, scat_ofphoton, instant_photon_number, max_block_number, besselTexObj);
		}else{
			if (ideal_nblocks == 0)
			ideal_nblocks = 1;
			GPU_track<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, dPTableTexObj, d_table_ptr, scat_ofphoton, instant_photon_number, ideal_nblocks, besselTexObj);
		}		

		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Tracking kernel execution time: %f s\n", milliseconds/1000.);

		cudaMemcpyFromSymbol(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in GPU_track %s\n", cudaGetErrorString(cudaStatus));
			fprintf(stderr, "number of scattered photons: %llu out of %llu", num_scat_phs[0], MAX_LAYER_SCA * instant_photon_number);
			exit(1);
		}
		
		if(ideal_nblocks > max_block_number){
			GPU_record<<<max_block_number,N_THREADS>>>(initial_photon_states, d_spect, instant_photon_number, max_block_number);
		}else{
			if (ideal_nblocks == 0)
			ideal_nblocks = 1;
			GPU_record<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_spect, instant_photon_number, ideal_nblocks);
		}			
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in GPU_record %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
	 	freePhotonData(&initial_photon_states);
		//cudaFree(initial_photon_states);


		cudaMemcpyToSymbol(tracking_counter, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
		printf("number of scattered photons generated = %llu in round 0\n", num_scat_phs[0]);
		printf("Solving the scattered photons...\n");
		int n = 1;
		bool quit_flag_sca = false;
		unsigned long long scatterings_performed = 0;
		while(quit_flag_sca == false && n < MAX_LAYER_SCA){
			printf("Starting round %d\n", n);
			ideal_nblocks = ceil(num_scat_phs[n-1]/N_THREADS);
			cudaEventRecord(start, 0);
			if(ideal_nblocks > max_block_number){
				GPU_track_scat<<<max_block_number,N_THREADS>>>(scat_ofphoton, dPTableTexObj, d_table_ptr, scat_ofphoton, n, max_block_number * N_THREADS, besselTexObj);
			}else{
				if (ideal_nblocks == 0)
					ideal_nblocks = 1;

				GPU_track_scat<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, dPTableTexObj, d_table_ptr, scat_ofphoton, n, ideal_nblocks * N_THREADS, besselTexObj);
			}
			cudaDeviceSynchronize();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop); 
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Scattering kernel, round %d, execution time: %f s\n", n,milliseconds/1000.);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "in GPU_track_scat %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}

			if(ideal_nblocks > max_block_number){
				GPU_record_scattering<<<max_block_number,N_THREADS>>>(scat_ofphoton, d_spect, instant_photon_number, max_block_number, n);
			}else{
				if (ideal_nblocks == 0)
				ideal_nblocks = 1;
				GPU_record_scattering<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, d_spect, instant_photon_number, ideal_nblocks, n);
			}			
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "in GPU_record_scattering %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}
			
			cudaMemcpyFromSymbol(&scatterings_performed, scattering_counter, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
			if(scatterings_performed != num_scat_phs[n - 1]){
				printf("Not all the photons created in scatterings have been evaluated (%llu, %llu)\n", scatterings_performed, num_scat_phs[n - 1]);
				exit(1);
			}
			cudaMemcpyFromSymbol(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA* sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
			if(num_scat_phs[n] == 0){
				printf("Quit flag reached in round %d!\n", n);
				quit_flag_sca = true;
			}
			cudaMemcpyToSymbol(scattering_counter, &reset, sizeof(unsigned long long));
			printf("number of scattered photons generated = %llu in round %d\n", num_scat_phs[n], n);
			n++;
		}

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in scattering kernerls %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		freePhotonData(&scat_ofphoton);
		instant_partition +=1;
		photons_processed += instant_photon_number;
		memset(num_scat_phs, 0, sizeof(num_scat_phs));
		cudaMemcpyToSymbol(d_num_scat_phs, &num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);

	}

    cudaMemcpyErrorCheck(spect, d_spect, N_EBINS * N_THBINS * sizeof(of_spectrum), cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&N_superph_recorded, d_N_superph_recorded, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
	report_spectrum(gen_superph, spect, filename);
	cudaFree(d_spect);
	cudaFree(generated_photons_arr); 
	cudaFree(dnmax_arr);
	cudaFree(d_geom);
	cudaFree(d_table_ptr);
	//cudaFree(d_p);
    cudaFree(d_index_to_ijk);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
	cudaDestroyTextureObject(dPTableTexObj);
	cudaFreeArray(dPTableCuArray);
	//CHECK_CUDA_ERROR(cudaDestroyTextureObject(tableTexObj));

}

__global__ void GPU_generate_photons(const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const time_t time, unsigned long long * __restrict__  generated_photons_arr, double * __restrict__ dnmax_arr, cudaTextureObject_t besselTexObj){
	unsigned long long generated_photons;
	double dnmax;
	int i, j, k;
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int seed = 139;//139 * global_index + time;
	GPU_init_monty_rand(seed);
	curandState localState = my_curand_state[global_index]; 

	/*This is how we'll split things between blocks and threads*/
	/*We'll divide N1 * N2 * N3 between blocks*/
	for(int a = global_index; a < N1 * N2 * N3; (a += N_BLOCKS * N_THREADS)){
		k = a % N3;
		j = (a/N3) % N2;
		i = (a/(N2 * N3));

		/*This portion of the code will estimate the number of photons that are going to be generated in each zone (n2gen). It will also estimate the dnmax
		which will be used when sampling the photons*/
		GPU_init_zone(i,j,k, &generated_photons, &dnmax, d_geom, d_p, d_Ns, &localState, besselTexObj);
		//GPU_init_blackbody_photons(i,j,k, &generated_photons, &dnmax, d_geom, d_dx, d_Ns);
		generated_photons_arr[a] = generated_photons;


		dnmax_arr[a] = dnmax;
		atomicAdd(&photon_count, generated_photons);
	}
	my_curand_state[global_index] = localState;
	return;
}



__device__ void GPU_init_zone(const int i, const int j, const int k, unsigned long long * __restrict__  n2gen, double * __restrict__ dnmax, const struct of_geom * __restrict__  d_geom, const double * __restrict__ d_p, const int d_Ns_par, curandState * localState, cudaTextureObject_t besselTexObj)
{
	int l;
	double Ne, Thetae, Bmag, lbth;
	double dl, dn, ninterp, K2;
	double Ucon[NDIM], Bcon[NDIM];
	double lb_min = log(BTHSQMIN);
	double dlb = log(BTHSQMAX / BTHSQMIN) / NINT;
	double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);
	get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);

	if (Ne == 0. || Thetae < THETAE_MIN) {
		*n2gen = 0.;
		*dnmax = 0.;
		return;
	}

		lbth = log(Bmag * Thetae * Thetae);

		dl = (lbth - lb_min) / dlb;
		l = (int) dl;
		dl = dl - l;
		if (l < 0) {
			*dnmax = 0.;
			*n2gen = 0.;
			return;
		} else if (l >= NINT|| 1) {
			//printf( "Outside of range! Change Nint!. B * th**2 = %le, lbth = %le, lb_min = %le, dlb = %le l = %d, (i,j) = (%d, %d)\n", Bmag * Thetae * Thetae, lbth, lb_min, dlb, l,i, j);
			ninterp = 0.;
			*dnmax = 0.;
			for (l = 0; l <= N_ESAMP; l++) {
				dn = F_eval(Thetae, Bmag,
						exp(l * dlnu +
						lnu_min)) / (exp(d_wgt[l]) +
								1.e-100);
				if (dn > *dnmax)
					*dnmax = dn;
				ninterp += dlnu * dn;
			}
			ninterp *= d_dx[1] * d_dx[2] * d_dx[3] * L_UNIT * L_UNIT * L_UNIT
				* M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
				* 1. / HPL;

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

	#ifdef __CUDA_ARCH__
	K2 = K2_eval(Thetae, besselTexObj);
	#else
	K2 = K2_eval(Thetae);
	#endif

	if (K2 == 0.) {
		*n2gen = 0.;
		*dnmax = 0.;
		return;
	}
	
	double nz = d_geom[SPATIAL_INDEX2D(i,j)].g * Ne * Bmag * Thetae * Thetae * ninterp / K2;

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


__global__ void GPU_sample_photons_batch(struct of_photonSOA ph_init, const struct of_geom * __restrict__  d_geom, const double * __restrict__  d_p, const unsigned long long * __restrict__  generated_photons_arr, const double * __restrict__ dnmax_arr, const int max_partition_ph, 
	const unsigned long long photons_processed_sofar, const unsigned long long * __restrict__  index_to_ijk, cudaTextureObject_t besselTexObj){
		int i,j,k;
		unsigned long long photon_index = 0;
		const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
		int zone_index = 0;
		double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
		int past_zone = (N1 * N2 * N3);
		curandState localState = my_curand_state[global_index];
		while (true) {
			//curandState localState = my_curand_state[global_index];
			photon_index = (atomicAdd(&tracking_counter_sampling, 1)-1);
			//photon_index = global_index;
			if (photon_index >= max_partition_ph){
				break;
				//	return;
			}
	
			zone_index = findPhotonIndex(index_to_ijk, N1 * N2 * N3, photons_processed_sofar +photon_index);
			k = zone_index % N3;
			j = (zone_index/N3) % N2;
			i = (zone_index/(N2 * N3));


			/*Sample all the photons generated in GPU_init_zone*/
			GPU_sample_zone_photon(i,j,k, dnmax_arr[zone_index], ph_init, d_geom, d_p, (past_zone == zone_index? 0 : 1), photon_index, Econ, Ecov, &localState, besselTexObj);
			past_zone = zone_index;
			
		}
		my_curand_state[global_index] = localState;
}
	
__device__ void GPU_sample_zone_photon(const int i, const int j, const int k, const double dnmax, struct of_photonSOA ph, const struct of_geom * __restrict__ d_geom, const double * __restrict__ d_p, const int zone_flag, const unsigned long long ph_arr_index,
double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState *  localState, cudaTextureObject_t besselTexObj)
{
	/* Set all initial superphoton attributes */
	int l;
	int z = 0;
	double K_tetrad[NDIM], tmpK[NDIM], E;
	double th, cth, sth, phi, sphi, cphi, jmax, weight;
	double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM], bhat[NDIM];
	double Xarray[4] = {ph.X0[ph_arr_index], ph.X1[ph_arr_index], ph.X2[ph_arr_index], ph.X3[ph_arr_index]};
	double Karray[4] = {ph.K0[ph_arr_index], ph.K1[ph_arr_index], ph.K2[ph_arr_index], ph.K3[ph_arr_index]};
	coord(i, j, Xarray);
	double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double Nln = lnu_max - lnu_min;
	double nu;
	get_fluid_zone(i,j, z, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);

	/* Sample from superphoton distribution in current simulation zone */
	do {
		nu = exp(curand_uniform_double(localState) * Nln + lnu_min);
		weight = GPU_linear_interp_weight(nu);
	} while (curand_uniform_double(localState) >
		(F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / (dnmax));


	ph.w[ph_arr_index] = weight;


	bool do_condition;
	#ifdef __CUDA_ARCH__
		jmax = jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2., besselTexObj);
	#else
		jmax = jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);
	#endif

	do {
	cth = 2. * curand_uniform_double(localState)- 1.;
	th = acos(cth);
	#ifdef __CUDA_ARCH__
		do_condition = curand_uniform_double(localState) > jnu_synch(nu, Ne, Thetae, Bmag, th, besselTexObj) / jmax;
	#else
		do_condition = curand_uniform_double(localState) > jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax;
	#endif
	} while (do_condition);


	sth = sqrt(1. - cth * cth);
	phi = 2. * M_PI * curand_uniform_double(localState);

	cphi = cos(phi);
	sphi = sin(phi);



	E = nu * HPL / (ME * CL * CL);
	K_tetrad[0] = E;
	K_tetrad[1] = E * cth;
	K_tetrad[2] = E * cphi * sth;
	K_tetrad[3] = E * sphi * sth;

	if (zone_flag) {	/* first photon created in this zone, so make the tetrad */
		if (Bmag > 0.) {
			for (l = 0; l < NDIM; l++)
				bhat[l] = Bcon[l] * B_UNIT / Bmag;
		} else {
			for (l = 1; l < NDIM; l++)
				bhat[l] = 0.;
			bhat[1] = 1.;
		}


		GPU_make_tetrad(Ucon, bhat, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Econ, Ecov);
	}

	GPU_tetrad_to_coordinate(Econ, K_tetrad, Karray);
	K_tetrad[0] *= -1.;
	GPU_tetrad_to_coordinate(Ecov, K_tetrad, tmpK);

	ph.E[ph_arr_index] = ph.E0[ph_arr_index] = ph.E0s[ph_arr_index] = -tmpK[0];
	ph.tau_scatt[ph_arr_index] = 0.;
	ph.tau_abs[ph_arr_index] = 0.;
	ph.X1i[ph_arr_index] = Xarray[1];
	ph.X2i[ph_arr_index] = Xarray[2];
	ph.nscatt[ph_arr_index] = 0;
	ph.K0[ph_arr_index] = Karray[0];
	ph.K1[ph_arr_index] = Karray[1];
	ph.K2[ph_arr_index] = Karray[2];
	ph.K3[ph_arr_index] = Karray[3];
	ph.X0[ph_arr_index] = Xarray[0];
	ph.X1[ph_arr_index] = Xarray[1];
	ph.X2[ph_arr_index] = Xarray[2];
	ph.X3[ph_arr_index] = Xarray[3];




	return;
}
	
__global__ void GPU_track(struct of_photonSOA ph, cudaTextureObject_t  d_p, const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long max_partition_ph, const int nblocks, cudaTextureObject_t besselTexObj){
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long photon_index = 0;
	int n = 1;
	curandState localState = my_curand_state[global_index];
	/*track each photon we created along its geodesic*/

    while (true) {
        // Each thread grabs the next available photon index atomically
        photon_index = (atomicAdd(&tracking_counter, 1) - 1);

        // If all photons are processed, exit the loop
        if (photon_index >= max_partition_ph) break;
        // Track the photon
		
        GPU_track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, 0, photon_index, &localState, besselTexObj);

        // Progress indicator
        if (global_index == 0) {
            float percentage = 100 - ((max_partition_ph-  photon_index) * 100) / max_partition_ph;
            if (percentage >= n * 10) {
                printf("Progress: %llu%%\n", (unsigned long long)percentage);
                n++;
            }
        }
	}
	my_curand_state[global_index] = localState;
}
	
__global__ void GPU_record(struct of_photonSOA ph, struct of_spectrum * __restrict__  d_spect, const unsigned long long  max_partition_ph, const int nblocks)
{
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Number of photons processed %llu\n", tracking_counter);
	}
	for(unsigned long long a = global_index; a < max_partition_ph; (a += nblocks * N_THREADS)){
		if(GPU_record_criterion(ph.X1[a]))
		GPU_record_super_photon(ph, d_spect, a);
	}
}



__global__ void GPU_track_scat(struct of_photonSOA ph, cudaTextureObject_t d_p, const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int n, const int number_of_threads, cudaTextureObject_t besselTexObj){
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long round_num_scat_init = 0;
	curandState localState = my_curand_state[global_index];
	double Ne, Thetae, B;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM], Gcov[NDIM][NDIM];

	for (int cum_sum = 0; cum_sum < n -1; cum_sum++){
		round_num_scat_init += d_num_scat_phs[cum_sum]; 
	}
	unsigned long long round_num_scat_end = round_num_scat_init + d_num_scat_phs[n-1];
	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Interval going from %llu to %llu in round %d\n", round_num_scat_init, round_num_scat_end, n);
	}
	
	for(unsigned long long a = round_num_scat_init + global_index; a < round_num_scat_end; (a += number_of_threads)){
		double X[NDIM] = {ph.X0[a], ph.X1[a], ph.X2[a], ph.X3[a]};
		gcov_func(X, Gcov);
		#ifndef SPHERE_TEST
			GPU_get_fluid_params(X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, d_p);
		#else
			GPU_get_fluid_params(X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);
		#endif
		GPU_scatter_super_photon(ph, ph, Ne, Thetae, B, Ucon, Bcon, Gcov, &localState, a);
		if (ph.w[a] < 1.e-100) {	/* must have been a problem popping k back onto light cone */
			return;
		}
		GPU_track_super_photon(ph, d_p, d_table_ptr, scat_ofphoton, n, a, &localState, besselTexObj);
		atomicAdd(&scattering_counter, 1);
	}
	my_curand_state[global_index] = localState;
}


__global__ void GPU_record_scattering(struct of_photonSOA ph, struct of_spectrum * __restrict__ d_spect, const unsigned long long  max_partition_ph, const int nblocks, const int n)
{
	const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long round_num_scat_init = 0;
	/*track each photon we created along its geodesic*/
	for (int cum_sum = 0; cum_sum < n -1; cum_sum++){
		round_num_scat_init += d_num_scat_phs[cum_sum]; 
	}
	unsigned long long round_num_scat_end = round_num_scat_init + d_num_scat_phs[n-1];
	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Interval going from %llu to %llu in round %d\n", round_num_scat_init, round_num_scat_end, n);
	}
	
	for(unsigned long long a =  round_num_scat_init + global_index; a < round_num_scat_end; (a += nblocks * N_THREADS)){

		if(GPU_record_criterion(ph.X1[a]))
		GPU_record_super_photon(ph, d_spect, a);
	}
}
