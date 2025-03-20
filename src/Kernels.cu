

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


	struct of_spectrum spect[N_THBINS][N_EBINS] = { };
    struct of_spectrum* d_spect;
    gpuErrchk(cudaMalloc((void**)&d_spect, N_THBINS * N_EBINS * sizeof(struct of_spectrum)));
	double * d_p; 
    gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
    cudaMemcpyErrorCheck(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice);

    double *d_table_ptr;
    gpuErrchk(cudaMalloc((void**)&d_table_ptr, (NW + 1) * (NT + 1) * sizeof(double)));
    cudaMemcpyErrorCheck(d_table_ptr, table, (NW + 1) * (NT + 1) * sizeof(double), cudaMemcpyHostToDevice);

	struct of_geom *d_geom;
	gpuErrchk(cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom)));
    cudaMemcpyErrorCheck(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice);

	transferGlobalVariables();

	int max_block_number = setMaxBlocks();

	unsigned long long gen_superph = 0;
	unsigned long long * generated_photons_arr;
	gpuErrchk(cudaMalloc(&generated_photons_arr, N1 * N2 * N3 * sizeof(unsigned long long)));
	double * dnmax_arr;
	gpuErrchk(cudaMalloc(&dnmax_arr, N1 * N2 * N3 * sizeof(double)));

	fprintf(stderr, "Generating super photons!\n");
	cudaEventRecord(start, 0);
    GPU_generate_photons<<<N_BLOCKS,N_THREADS>>>(d_geom, d_p, time, generated_photons_arr, dnmax_arr);
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
	struct of_photon * initial_photon_states;
	struct of_photon * scat_ofphoton;
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


		gpuErrchk(cudaMalloc(&initial_photon_states, instant_photon_number* sizeof(struct of_photon)));
		gpuErrchk(cudaMalloc(&scat_ofphoton, MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * instant_photon_number* sizeof(struct of_photon))); 

		fprintf(stderr, "Sampling the photons!\n");
		cudaEventRecord(start, 0);
		if(ideal_nblocks > max_block_number){
			GPU_sample_photons_batch<<<max_block_number,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk);
		}else{
			GPU_sample_photons_batch<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_geom, d_p, generated_photons_arr, dnmax_arr, instant_photon_number, photons_processed, d_index_to_ijk);
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
			GPU_track<<<max_block_number,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, d_spect, scat_ofphoton, instant_photon_number, max_block_number);
		}else{
			GPU_track<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, d_spect, scat_ofphoton, instant_photon_number, ideal_nblocks);
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
			GPU_record<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_spect, instant_photon_number, ideal_nblocks);
		}			
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in GPU_record %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
		cudaFree(initial_photon_states);


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
				GPU_track_scat<<<max_block_number,N_THREADS>>>(scat_ofphoton, d_p, d_table_ptr, d_spect, scat_ofphoton, n, max_block_number * N_THREADS);
			}else{
				if (ideal_nblocks == 0)
					ideal_nblocks = 1;

				GPU_track_scat<<<ideal_nblocks,N_THREADS>>>(scat_ofphoton, d_p, d_table_ptr, d_spect, scat_ofphoton, n, ideal_nblocks * N_THREADS);
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

		cudaFree(scat_ofphoton);
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
    cudaFree(d_index_to_ijk);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void GPU_generate_photons(struct of_geom * d_geom, double * d_p, time_t time, unsigned long long * generated_photons_arr, double * dnmax_arr){
	unsigned long long generated_photons;
	double dnmax;
	int i, j, k;
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int seed = 139;//139 * global_index + time;
	GPU_init_monty_rand(seed);
	curandState localState = my_curand_state[global_index]; 
	/*This is how we'll split things between blocks and threads*/
	/*We'll divide d_N1 * d_N2 * d_N3 between blocks*/
	for(int a = global_index; a < d_N1 * d_N2 * d_N3; (a += N_BLOCKS * N_THREADS)){
		k = a % d_N3;
		j = (a/d_N3) % d_N2;
		i = (a/(d_N2 * d_N3));

		/*This portion of the code will estimate the number of photons that are going to be generated in each zone (n2gen). It will also estimate the dnmax
		which will be used when sampling the photons*/
		GPU_init_zone(i,j,k, &generated_photons, &dnmax, d_geom, d_p, d_Ns, localState);
		//GPU_init_blackbody_photons(i,j,k, &generated_photons, &dnmax, d_geom, d_dx, d_Ns);
		generated_photons_arr[a] = generated_photons;
		dnmax_arr[a] = dnmax;
		atomicAdd(&photon_count, generated_photons);
	}
	my_curand_state[global_index] = localState;
	return;
}



__device__ void GPU_init_zone(int i, int j, int k, unsigned long long * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par, curandState localState)
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
		} else if (l >= NINT) {
			printf( "Outside of range! Change Nint!. B * th**2 = %le, lbth = %le, lb_min = %le, dlb = %le l = %d, (i,j) = (%d, %d)\n", Bmag * Thetae * Thetae, lbth, lb_min, dlb, l,i, j);
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

	K2 = K2_eval(Thetae);
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
		if (fmod(nz, 1.) > curand_uniform_double(&localState)) {
			*n2gen = (int) (nz) + 1;
		} else {
			*n2gen = (int) (nz);
		}
	}

	return;
}


__global__ void GPU_track_scat(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, int n, int number_of_threads){
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
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
		gcov_func(ph[a].X, Gcov);
		GPU_get_fluid_params(ph[a].X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, d_p);
		GPU_scatter_super_photon(&ph[a], &ph[a], Ne, Thetae, B, Ucon, Bcon, Gcov, localState);
		GPU_track_super_photon(&ph[a], d_spect, d_p, d_table_ptr, scat_ofphoton, n, a,0, localState);
		atomicAdd(&scattering_counter, 1);
	}
	my_curand_state[global_index] = localState;
}


__global__ void GPU_record_scattering(struct of_photon * ph, struct of_spectrum * d_spect, unsigned long long  max_partition_ph, int nblocks, int n)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
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

		if(GPU_record_criterion(&ph[a]))
		GPU_record_super_photon(&ph[a], d_spect);
	}
}

__global__ void GPU_record(struct of_photon * ph, struct of_spectrum * d_spect, unsigned long long  max_partition_ph, int nblocks)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Number of photons processed %llu\n", tracking_counter);
	}
	for(unsigned long long a = global_index; a < max_partition_ph; (a += nblocks * N_THREADS)){

		if(GPU_record_criterion(&ph[a]))
		GPU_record_super_photon(&ph[a], d_spect);
	}
}



__global__ void GPU_track(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, unsigned long long max_partition_ph, int nblocks){
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	double percentage;
	unsigned long long photon_index = 0;
	int n = 1;
	curandState localState = my_curand_state[global_index];
	struct of_photon local_ph;
	/*track each photon we created along its geodesic*/

    while (true) {
        // Each thread grabs the next available photon index atomically
        photon_index = (atomicAdd(&tracking_counter, 1) - 1);

        // If all photons are processed, exit the loop
        if (photon_index >= max_partition_ph) break;
		local_ph = ph[photon_index];
        // Track the photon
        GPU_track_super_photon(&local_ph, d_spect, d_p, d_table_ptr, scat_ofphoton, 0, photon_index, 0, localState);

		ph[photon_index] = local_ph;
        // Progress indicator
        if (global_index == 0) {
            percentage = 100 - ((max_partition_ph-  photon_index) * 100) / max_partition_ph;
            if (percentage >= n * 10) {
                printf("Progress: %llu%%\n", (unsigned long long)percentage);
                n++;
            }
        }
	}
	my_curand_state[global_index] = localState;
}


__global__ void GPU_sample_photons_batch(struct of_photon *ph_init, struct of_geom * d_geom, double * d_p, unsigned long long * generated_photons_arr, double * dnmax_arr, int max_partition_ph, 
unsigned long long photons_processed_sofar, unsigned long long * index_to_ijk){
	int i,j,k;
	unsigned long long photon_index = 0;
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
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

		/*Sample all the photons generated in GPU_init_zone*/
		GPU_sample_zone_photon(i,j,k, dnmax_arr[zone_index], ph_init, d_geom, d_p, (past_zone == zone_index? 0 : 1), photon_index, Econ, Ecov, localState);
		past_zone = zone_index;
	}
	my_curand_state[global_index] = localState;
}

__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p, int zone_flag, unsigned long long ph_arr_index,
double (*Econ)[NDIM], double (*Ecov)[NDIM], curandState localState)
{
/* Set all initial superphoton attributes */
	int l;
	int z = 0;
	double K_tetrad[NDIM], tmpK[NDIM], E;
	double th, cth, sth, phi, sphi, cphi, jmax, weight;
	double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM], bhat[NDIM];
	coord(i, j, ph[ph_arr_index].X);
    double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double Nln = lnu_max - lnu_min;
	double nu;
	get_fluid_zone(i, j, z, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);

	/* Sample from superphoton distribution in current simulation zone */

	#ifdef SCATTERING_TEST
		double conditioner, test;
		double ThetaS = 1.e-8;
		double temperature = ThetaS * ME * CL * CL / KBOL;
		do {
			nu = exp(curand_uniform_double(&localState) * Nln + lnu_min);
			weight = GPU_linear_interp_weight(nu);
			conditioner = (M_PI) * 2.0 * nu * nu * nu / (CL * CL) * (1./(exp(HPL * nu/(KBOL * temperature)) - 1.0))/(weight+ 1e-100)/dnmax;
			test = curand_uniform_double(&localState);

		} while (test >  conditioner);
	#else
		do {
			nu = exp(curand_uniform_double(&localState) * Nln + lnu_min);
			weight = GPU_linear_interp_weight(nu);
		} while (curand_uniform_double(&localState) >
			(F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / dnmax);
	#endif

	ph[ph_arr_index].w = weight;


	bool do_condition;
	jmax = jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);
	do {
		cth = 2. * curand_uniform_double(&localState) - 1.;
		th = acos(cth);
		do_condition = curand_uniform_double(&localState) > jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax;
	} while (do_condition);

	sth = sqrt(1. - cth * cth);
	phi = 2. * M_PI * curand_uniform_double(&localState);
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
	GPU_tetrad_to_coordinate(Econ, K_tetrad, ph[ph_arr_index].K);
	K_tetrad[0] *= -1.;
	GPU_tetrad_to_coordinate(Ecov, K_tetrad, tmpK);

	ph[ph_arr_index].E = ph[ph_arr_index].E0 = ph[ph_arr_index].E0s = -tmpK[0];
	//ph[ph_arr_index].L = tmpK[3];
	ph[ph_arr_index].tau_scatt = 0.;
	ph[ph_arr_index].tau_abs = 0.;
	ph[ph_arr_index].X1i = ph[ph_arr_index].X[1];
	ph[ph_arr_index].X2i = ph[ph_arr_index].X[2];
	ph[ph_arr_index].nscatt = 0;
	// ph[ph_arr_index].ne0 = Ne;
	// ph[ph_arr_index].b0 = Bmag;
	// ph[ph_arr_index].thetae0 = Thetae;

	return;
}


