/*This should be passed to every file*/
#include "decs.h"

/*GLOBAL VARIABLES*/
/*We need to be carefull with global variables that are modified by multiple threads at a time. We can use global variables, but just
do not edit with multiple threads, unless we know what we are doing*/

struct of_scattering{
	int bound_flag;
	double dtau_scatt, dtau_abs, dtau;
	double bi, bf;
	double alpha_scatti, alpha_scattf;
	double alpha_absi, alpha_absf;
	double dl, x1;
	double nu, Thetae, Ne, B, theta;
	double dtauK, frac;
	double bias;
	double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
	double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
	int nstep;
};

/**********************************************************************************************************************************************************************************/


__host__ void launch_loop(struct of_photon ph, int quit_flag, time_t time, double * p, const char * filename){
    cudaEvent_t start, stop;
    float milliseconds = 0;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	/*Copying global variables*/
	struct of_spectrum spect[N_THBINS][N_EBINS] = { };
    struct of_spectrum* d_spect;
    gpuErrchk(cudaMalloc((void**)&d_spect, N_THBINS * N_EBINS * sizeof(struct of_spectrum)));
    cudaMemcpyToSymbol(d_N1, &N1, sizeof(int));
	cudaMemcpyToSymbol(d_Ns, &Ns, sizeof(int));
    cudaMemcpyToSymbol(d_N2, &N2, sizeof(int));
    cudaMemcpyToSymbol(d_N3, &N3, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, NDIM * sizeof(double));
	
	if(hslope > 0)
	cudaMemcpyToSymbol(d_hslope, &hslope,sizeof(double));
	
	cudaMemcpyToSymbol(d_startx, &startx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_stopx, &stopx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_a, &a, sizeof(double));
	cudaMemcpyToSymbol(d_thetae_unit, &Thetae_unit, sizeof(double));

	double * d_p; 
    gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
    cudaMemcpyErrorCheck(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_wgt, &wgt, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_F, &F, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_nint, &nint, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_dndlnu_max, &dndlnu_max, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_K2, &K2, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_bias_norm, &bias_norm, sizeof(double));
	cudaMemcpyToSymbol(d_max_tau_scatt, &max_tau_scatt, sizeof(double));
	cudaMemcpyToSymbol(d_Rh, &Rh, sizeof(double));
	cudaDeviceSetLimit(cudaLimitStackSize, 8000);




	// Identifying GPU:
 	int device_id;
    cudaGetDevice(&device_id);  // Get the current device ID
	int maxThreadsPerBlock;
	int maxBlocksPerMultiprocessor;
	int numSMs;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);  // Get properties of the current device
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	maxBlocksPerMultiprocessor = prop.maxBlocksPerMultiProcessor;
	numSMs = prop.multiProcessorCount;
	int max_block_number = maxBlocksPerMultiprocessor * numSMs;
    printf("Current GPU in use: %s\n", prop.name);  // Print the GPU name
	printf("Max number of threads per block: %d\n", maxThreadsPerBlock);
	printf("Max number of blocks per SM: %d\n",maxBlocksPerMultiprocessor);
	printf("number of SMs: %d\n", numSMs);
	printf("Therefore, total number of blocks:%d\n", max_block_number );
	// Set the printf FIFO size limit
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 5000);

	// Allocate device memory
    double *d_table_ptr;
    gpuErrchk(cudaMalloc((void**)&d_table_ptr, (NW + 1) * (NT + 1) * sizeof(double)));

    // Copy data from host to device
    cudaMemcpyErrorCheck(d_table_ptr, table, (NW + 1) * (NT + 1) * sizeof(double), cudaMemcpyHostToDevice);

	/*Transfering geom array*/
	struct of_geom *d_geom;
	gpuErrchk(cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom)));
    cudaMemcpyErrorCheck(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice);
	/*Done transfering geom*/

	/*Number of super photons generated*/
	unsigned long long gen_superph = 0;

	/*Creating array for generated_photons and dnmax*/
	unsigned long long * generated_photons_arr;
	gpuErrchk(cudaMalloc(&generated_photons_arr, N1 * N2 * N3 * sizeof(unsigned long long)));

	double * dnmax_arr;
	gpuErrchk(cudaMalloc(&dnmax_arr, N1 * N2 * N3 * sizeof(double)));

	/*Calling the function to generate photons and sample them*/
	fprintf(stderr, "Generating super photons!\n");
	cudaEventRecord(start, 0);
    GPU_generate_photons<<<N_BLOCKS,N_THREADS>>>(d_geom, d_p, time, generated_photons_arr, dnmax_arr);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // Wait for stop event to be recorded
	// Calculate the elapsed time in milliseconds
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Generation kernel, execution time: %f s\n",milliseconds/1000.);
	cudaMemcpyFromSymbol(&gen_superph, photon_count, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
	fprintf(stderr, "Number of generated photons: %llu\n", gen_superph);


	/*CREATING CUMULATIVE SUM ARRAY #################################################################################################*/
		unsigned long long * d_index_to_ijk;
		gpuErrchk(cudaMalloc(&d_index_to_ijk, N1 * N2 * N3 * sizeof(unsigned long long)));
		unsigned long long *h_index_to_ijk = (unsigned long long *)malloc(N1 * N2 * N3 * sizeof(unsigned long long));
		unsigned long long *h_generated_photon_arr = (unsigned long long *)malloc(N1 * N2 * N3 * sizeof(unsigned long long));

		cudaMemcpyErrorCheck(h_generated_photon_arr, generated_photons_arr, N1 * N2 * N3* sizeof(unsigned long long ), cudaMemcpyDeviceToHost);
		h_index_to_ijk[0] = h_generated_photon_arr[0];
		for (int i = 1; i < N1 * N2 * N3; i++) {
			h_index_to_ijk[i] = h_index_to_ijk[i - 1] + h_generated_photon_arr[i];
		}

		cudaMemcpyErrorCheck(d_index_to_ijk, h_index_to_ijk, N1 * N2 * N3* sizeof(unsigned long long), cudaMemcpyHostToDevice);
		free(h_index_to_ijk);

	/*ENDING CUMULATIVE SUM ARRAY #################################################################################################*/

	/*Now, things may get complicated memory wise. We need to partition it?*/
	size_t free_mem, total_mem;
	
	//Get GPU remaining memory.
	cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
	if (err != cudaSuccess) {
		printf("Failed to get GPU memory info: %s\n", cudaGetErrorString(err));
	}
    size_t required_mem ;
	required_mem = gen_superph * sizeof(struct of_photon);
	required_mem += MAX_LAYER_SCA *  gen_superph/1 * sizeof(struct of_photon);
	if (required_mem > free_mem) {
		printf("Not enough memory to allocate %.2lf GB for photon states. Available memory: %.2lf GB\n", required_mem / 1e9, free_mem / 1e9);
		printf("Beginning equipartion of photons...\n");
    }
	// Check how many times gen_superph needs to be divided
	unsigned long long superph_per_batch = gen_superph;
	int batch_divisions = 1;
	while (required_mem > free_mem) {
		// Divide gen_superph by 2 and recalculate required memory
		superph_per_batch = gen_superph/batch_divisions;
		required_mem = superph_per_batch * sizeof(struct of_photon);
		//CHANGE LATER THIS MEANS NO SCATTERING IS ALLOCATED
		required_mem += MAX_LAYER_SCA * SCATTERINGS_PER_PHOTON * superph_per_batch * sizeof(struct of_photon);
		// Track the number of divisions
		batch_divisions++;
	}
	printf("Required partitions: %d. Number of photons per partition: %d\n", batch_divisions, (int)(gen_superph/batch_divisions));
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
		//In here, we consider a maximum of 8 scattering layers and each photon can scatter.
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
		cudaEventSynchronize(stop); // Wait for stop event to be recorded
		// Calculate the elapsed time in milliseconds
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
			GPU_track<<<max_block_number,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, d_spect, scat_ofphoton, instant_photon_number, instant_partition);
		}else{
			GPU_track<<<ideal_nblocks,N_THREADS>>>(initial_photon_states, d_p, d_table_ptr, d_spect, scat_ofphoton, instant_photon_number, instant_partition);
		}		

		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop); // Wait for stop event to be recorded
		// Calculate the elapsed time in milliseconds
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Tracking kernel execution time: %f s\n", milliseconds/1000.);

		cudaMemcpyFromSymbol(&num_scat_phs, d_num_scat_phs, MAX_LAYER_SCA * sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "in GPU_track %s\n", cudaGetErrorString(cudaStatus));
			fprintf(stderr, "number of scattered photons: %llu out of %llu", num_scat_phs[0], MAX_LAYER_SCA * instant_photon_number);
			exit(1);
		}
		cudaFree(initial_photon_states);


		cudaMemcpyToSymbol(tracking_counter, &reset, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice);
		printf("number of scattered photons generated = %llu in round 0\n", num_scat_phs[0]);
		/*Now, I create the array that will withhold all the information about the scattered photons*/
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
			cudaEventSynchronize(stop); // Wait for stop event to be recorded
			// Calculate the elapsed time in milliseconds
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Scattering kernel, round %d, execution time: %f s\n", n,milliseconds/1000.);
			// Check for kernel launch errors
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "in GPU_track_scat %s\n", cudaGetErrorString(cudaStatus));
				exit(1);
			}
			cudaMemcpyFromSymbol(&scatterings_performed, scattering_counter, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
			if(scatterings_performed != num_scat_phs[n - 1]){
				printf("Not all the photons created in scatterings have been evaluated (%llu, %llu)\n", scatterings_performed, num_scat_phs[n - 1]);
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
	int seed = 139 * global_index + time;
	GPU_init_monty_rand(seed);
	/*This is how we'll split things between blocks and threads*/
	/*We'll divide d_N1 * d_N2 * d_N3 between blocks*/
	for(int a = global_index; a < d_N1 * d_N2 * d_N3; (a += N_BLOCKS * N_THREADS)){
		k = a % d_N3;
		j = (a/d_N3) % d_N2;
		i = (a/(d_N2 * d_N3));

		/*This portion of the code will estimate the number of photons that are going to be generated in each zone (n2gen). It will also estimate the dnmax
		which will be used when sampling the photons*/
		GPU_init_zone(i,j,k, &generated_photons, &dnmax, d_geom, d_p, d_Ns);
		//GPU_init_blackbody_photons(i,j,k, &generated_photons, &dnmax, d_geom, d_dx, d_Ns);
		generated_photons_arr[a] = generated_photons;
		dnmax_arr[a] = dnmax;
		atomicAdd(&photon_count, generated_photons);
	}
	return;
}



__device__ void GPU_init_zone(int i, int j, int k, unsigned long long * n2gen, double *dnmax, struct of_geom * d_geom, double * d_p, int d_Ns_par)
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
		if (fmod(nz, 1.) > GPU_monty_rand()) {
			*n2gen = (int) (nz) + 1;
		} else {
			*n2gen = (int) (nz);
		}
	}
	// if(i == 0)
	// printf("%d, %d, %d, (%le, %le, %le, %le, %le, %le)\n",i,j,*n2gen, d_geom[SPATIAL_INDEX2D(i,j)].g, Ne,  Bmag,  Thetae,  ninterp,  K2);

	return;
}

__device__ void GPU_init_blackbody_photons(int i, int j, int k, unsigned long long *n2gen, double *dnmax, 
                                          struct of_geom *d_geom, 
                                          double *d_dx, int d_Ns_par) 
{
    // Constants
    const double h = HPL;  // Planck constant
    const double c = CL;    // Speed of light
    const double kb = KBOL;   // Boltzmann constant
    
    // Frequency sampling parameters
    const double lnu_min = log(NUMIN);
    const double lnu_max = log(NUMAX);
    const double dlnu = (lnu_max - lnu_min) / (N_ESAMP);
    
    // Initialize variables
    double ninterp = 0.0;
    *dnmax = 0.0;
    double ThetaS = 1e-8;
	double temperature =  ThetaS* ME * CL * CL / kb;
    
	if (i != 200){
		*n2gen = 0;
		*dnmax = 0;
		return;
	}
	// //only emit photons at r equal 1 cm
	// double x1_sphere_radius = log(1./L_UNIT);
	// double x1 = d_startx[1] + i * d_dx[1];

	// //if x1 is between the sphere radius and the sphere radius - dx1, emit photons
	// //also if x1 is between the sphere radius and the sphere radius + dx1, emit photons
	// if (x1 < x1_sphere_radius - d_dx[1] || x1 > x1_sphere_radius + d_dx[1]){
	// 	*n2gen = 0;
	// 	*dnmax = 0;
	// 	return;
	// }
	// double proportional_factor = fabs(x1 - x1_sphere_radius)/(d_dx[1]);
    // Integrate over frequency to get total number of photons

    for (int l = 0; l <= N_ESAMP; l++) {
        double nu = exp(l * dlnu + lnu_min);
        
        // Planck function (photons per frequency per volume)
        double dn = (M_PI) * 2 * nu * nu * nu / (c * c) * (1./(exp(h * nu/(kb * temperature)) - 1.0))/(exp(d_wgt[l]) + 1.e-100);
        
        // Track maximum for importance sampling
        if (dn > *dnmax) {
            *dnmax = dn;
        }
        
        // Integrate over frequency
        ninterp += dlnu * dn;
    }
    i = d_N1 - 1;
    // Scale by volume element
	double gdet_area = sqrt(d_geom[SPATIAL_INDEX2D(i,j)].gcov[2][2] * d_geom[SPATIAL_INDEX2D(i,j)].gcov[3][3]);
    double area = d_dx[2] * d_dx[3] * L_UNIT * L_UNIT;
    double nz = gdet_area * ninterp * area;


    // // Safety check for unreasonably large photon numbers
    // if (nz > d_Ns_par * log(NUMAX/NUMIN)) {
    //     printf("Warning: Too many photons in zone (%d,%d): nz=%le, gdet = %le, T=%le K, NUMAX = %le, max = %le, dnmax = %le\n", 
    //            i, j, nz, d_geom[SPATIAL_INDEX2D(i,j)].g, temperature, NUMAX, d_Ns_par * log(NUMAX/NUMIN), *dnmax);
    //     *n2gen = 0.;
    //     *dnmax = 0.;
    //     return;
    // }
    
    // Probabilistic rounding for fractional photons
    if (fmod(nz, 1.0) > GPU_monty_rand()) {
        *n2gen = (unsigned long long)(nz) + 1;
    } else {
        *n2gen = (unsigned long long)(nz);
    }
}


__global__ void GPU_track_scat(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, int n, int number_of_threads){
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long round_num_scat_init = 0;


	for (int cum_sum = 0; cum_sum <= n -2; cum_sum++){
		round_num_scat_init += d_num_scat_phs[cum_sum]; 
	}
	unsigned long long round_num_scat_end = round_num_scat_init + d_num_scat_phs[n-1];
	/*track each photon we created along its geodesic*/
	if(global_index == 0){
		printf("Interval going from %llu to %llu in round %d\n", round_num_scat_init, round_num_scat_end, n);

	}
	
	for(unsigned long long a = round_num_scat_init + global_index; a < round_num_scat_end; (a += number_of_threads)){
		GPU_track_super_photon(&ph[a], d_spect, d_p, d_table_ptr, scat_ofphoton, n, a,0);
		atomicAdd(&scattering_counter, 1);
	}
}


__global__ void GPU_track(struct of_photon * ph, double * d_p, double * d_table_ptr, struct of_spectrum * d_spect, struct of_photon * scat_ofphoton, int max_partition_ph, int instant_partition){
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	double percentage;
	unsigned long long photon_index = 0;
	int n = 1;
	/*track each photon we created along its geodesic*/

    while (true) {
        // Each thread grabs the next available photon index atomically
        photon_index = (atomicAdd(&tracking_counter, 1) - 1);

        // If all photons are processed, exit the loop
        if (photon_index >= max_partition_ph) break;

        // Track the photon
        GPU_track_super_photon(&ph[photon_index], d_spect, d_p, d_table_ptr, scat_ofphoton, 0, photon_index, instant_partition);

        // Progress indicator
        if (global_index == 0) {
            percentage = 100 - ((max_partition_ph-  photon_index) * 100) / max_partition_ph;
            if (percentage >= n * 10) {
                printf("Progress: %llu%%\n", (unsigned long long)percentage);
                n++;
            }
        }
	}
}


__global__ void GPU_sample_photons_batch(struct of_photon *ph_init, struct of_geom * d_geom, double * d_p, unsigned long long * generated_photons_arr, double * dnmax_arr, int max_partition_ph, 
unsigned long long photons_processed_sofar, unsigned long long * index_to_ijk){
	int i,j,k;
	unsigned long long photon_index = 0;
	int zone_index = 0;
	double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
	int past_zone = (d_N1 * d_N2 * d_N3);

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
		GPU_sample_zone_photon(i,j,k, dnmax_arr[zone_index], ph_init, d_geom, d_p, (past_zone == zone_index? 0 : 1), photon_index, Econ, Ecov);
		past_zone = zone_index;
	}
}

__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p, int zone_flag, unsigned long long ph_arr_index,
double (*Econ)[NDIM], double (*Ecov)[NDIM])
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
			nu = exp(GPU_monty_rand() * Nln + lnu_min);
			weight = GPU_linear_interp_weight(nu);
			conditioner = (M_PI) * 2.0 * nu * nu * nu / (CL * CL) * (1./(exp(HPL * nu/(KBOL * temperature)) - 1.0))/(weight+ 1e-100)/dnmax;
			test = GPU_monty_rand();

		} while (test >  conditioner);
	#else
		do {
			nu = exp(GPU_monty_rand() * Nln + lnu_min);
			weight = GPU_linear_interp_weight(nu);
		} while (GPU_monty_rand() >
			(F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / dnmax);
	#endif

	ph[ph_arr_index].w = weight;


	bool do_condition;
	jmax = jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);
	do {
		cth = 2. * GPU_monty_rand() - 1.;
		th = acos(cth);
		do_condition = GPU_monty_rand() > jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax;
	} while (do_condition);

	sth = sqrt(1. - cth * cth);
	phi = 2. * M_PI * GPU_monty_rand();
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
	ph[ph_arr_index].L = tmpK[3];
	ph[ph_arr_index].tau_scatt = 0.;
	ph[ph_arr_index].tau_abs = 0.;
	ph[ph_arr_index].X1i = ph[ph_arr_index].X[1];
	ph[ph_arr_index].X2i = ph[ph_arr_index].X[2];
	ph[ph_arr_index].nscatt = 0;
	ph[ph_arr_index].ne0 = Ne;
	ph[ph_arr_index].b0 = Bmag;
	ph[ph_arr_index].thetae0 = Thetae;

	return;
}





__host__ __device__ void get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM], struct of_geom * d_geom, double * d_p)
{
	int l, m;
	double Ucov[NDIM], Bcov[NDIM];
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	#ifdef __CUDA_ARCH__
	double thetaeUnit = d_thetae_unit;
	#else
	double thetaeUnit = Thetae_unit;

	#endif

	*Ne = d_p[NPRIM_INDEX3D(KRHO, i, j, k)] * NE_UNIT;
	*Thetae = d_p[NPRIM_INDEX3D(UU, i, j, k)] / (*Ne) * NE_UNIT * thetaeUnit;

	Bp[1] = d_p[NPRIM_INDEX3D(B1, i, j, k)];
	Bp[2] = d_p[NPRIM_INDEX3D(B2, i, j, k)];
	Bp[3] = d_p[NPRIM_INDEX3D(B3, i, j, k)];

	Vcon[1] = d_p[NPRIM_INDEX3D(U1, i, j, k)];
	Vcon[2] = d_p[NPRIM_INDEX3D(U2, i, j, k)];
	Vcon[3] = d_p[NPRIM_INDEX3D(U3, i, j, k)];

	/* Get Ucov */
	VdotV = 0.;
	for (l = 1; l < NDIM; l++)
		for (m = 1; m < NDIM; m++)
			VdotV += d_geom[SPATIAL_INDEX2D(i,j)].gcov[l][m] * Vcon[l] * Vcon[m];
	Vfac = sqrt(-1. / d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][0];
	for (l = 1; l < NDIM; l++){
		Ucon[l] = Vcon[l] - Vfac * d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][l];
		//printf("Ucon[%d] = %le, Vcon[%d] = %le, Vfac = %le, geom[0][%d] = %le\n", l, Ucon[l], l, Vcon[l], Vfac, l, d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][l]);
	}
	lower(Ucon, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Ucov);
	/* Get B and Bcov */
	UdotBp = 0.;
	for (l = 1; l < NDIM; l++)
		UdotBp += Ucov[l] * Bp[l];
	Bcon[0] = UdotBp;
	for (l = 1; l < NDIM; l++){
		Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
	}
	lower(Bcon, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Bcov);
	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;


	#ifdef SCATTERING_TEST
		*Ne = 1.e-4/(SIGMA_THOMSON * (1.e5 - 1.));
		*Thetae = 4.;
		*B = 0;
		return;
	#endif

	if (isnan(*B)){
		printf("i = %d, j = %d, k = %d\n", i, j, k);
		printf( "VdotV = %le\n", VdotV);
		printf( "Vfac = %lf\n", Vfac);
		for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcon[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcon[a][b]);
		for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcov[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcov[a][b]);
		printf( "Thetae: %lf\n", *Thetae);
		printf( "Ne: %lf\n", *Ne);
		printf( "Bp: %lf, %lf, %lf\n", Bp[1], Bp[2], Bp[3]);
		printf( "Vcon: %lf, %lf, %lf\n", Vcon[1], Vcon[2], Vcon[3]);
		printf( "Bcon: %lf, %lf, %lf, %lf\n Bcov: %lf, %lf, %lf %lf\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3], Bcov[0], Bcov[1], Bcov[2], Bcov[3]);
		printf( "Ucon: %lf, %lf, %lf, %lf\n Ucov: %lf, %lf, %lf %lf\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3], Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
	}
}
__device__ static double GPU_linear_interp_weight(double nu)
{
	int i;
	double di, lnu;
	double lnu_max = log(NUMAX);
	double lnu_min = log(NUMIN);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);
	lnu = log(nu);

	di = (lnu - lnu_min) / dlnu;
	i = (int) di;
	di = di - i;
	return exp((1. - di) * d_wgt[i] + di * d_wgt[i + 1]);

}



__device__ void GPU_make_tetrad(double Ucon[NDIM], double trial[NDIM],
		 double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM],
		 double Ecov[NDIM][NDIM])
{
	int k, l;
	double norm;
	__device__ void GPU_normalize(double *vcon, double Gcov[4][4]);
	__device__ void GPU_project_out(double *vcona, double *vconb, double Gcov[4][4]);

	/* econ/ecov index explanation:
	   Econ[k][l]
	   k: index attached to tetrad basis
	   index down
	   l: index attached to coordinate basis 
	   index up
	   Ecov[k][l]
	   k: index attached to tetrad basis
	   index up
	   l: index attached to coordinate basis 
	   index down
	 */

	/* start w/ time component parallel to U */
	for (k = 0; k < 4; k++){
		Econ[0][k] = Ucon[k];
	}
	GPU_normalize(Econ[0], Gcov);

	/*** done w/ basis vector 0 ***/

	/* now use the trial vector in basis vector 1 */
	/* cast a suspicious eye on the trial vector... */
	norm = 0.;
	for (k = 0; k < 4; k++)
		for (l = 0; l < 4; l++)
			norm += trial[k] * trial[l] * Gcov[k][l];
	if (norm <= SMALL_VECTOR) {	/* bad trial vector; default to radial direction */
		for (k = 0; k < 4; k++)	/* trial vector */
			trial[k] = GPU_delta(k, 1);
	}

	for (k = 0; k < 4; k++)	/* trial vector */
		Econ[1][k] = trial[k];

	/* project out econ0 */
	GPU_project_out(Econ[1], Econ[0], Gcov);
	GPU_normalize(Econ[1], Gcov);

	/*** done w/ basis vector 1 ***/

	/* repeat for x2 unit basis vector */
	for (k = 0; k < 4; k++)	/* trial vector */
		Econ[2][k] = GPU_delta(k, 2);
	/* project out econ[0-1] */
	GPU_project_out(Econ[2], Econ[0], Gcov);
	GPU_project_out(Econ[2], Econ[1], Gcov);
	GPU_normalize(Econ[2], Gcov);

	/*** done w/ basis vector 2 ***/

	/* and repeat for x3 unit basis vector */
	for (k = 0; k < 4; k++)	/* trial vector */
		Econ[3][k] = GPU_delta(k, 3);
	/* project out econ[0-2] */
	GPU_project_out(Econ[3], Econ[0], Gcov);

	GPU_project_out(Econ[3], Econ[1], Gcov);

	GPU_project_out(Econ[3], Econ[2], Gcov);

	GPU_normalize(Econ[3], Gcov);


	/*** done w/ basis vector 3 ***/

	/* now make covariant version */
	for (k = 0; k < 4; k++) {

		/* lower coordinate basis index */
		lower(Econ[k], Gcov, Ecov[k]);
	}

	/* then raise tetrad basis index */
	for (l = 0; l < 4; l++) {
		Ecov[0][l] *= -1.;
	}

	/* paranoia: check orthonormality */
	/*
	   double sum ;
	   int m ;
	   printf("ortho check:\n") ;
	   for(k=0;k<NDIM;k++)
	   for(l=0;l<NDIM;l++) {
	   sum = 0. ;
	   for(m=0;m<NDIM;m++) {
	   sum += Econ[k][m]*Ecov[l][m] ;
	   }
	   printf("%d %d %g\n",k,l,sum) ;
	   }
	   printf("\n") ;
	   for(k=0;k<NDIM;k++)
	   for(l=0;l<NDIM;l++) {
	   printf("%d %d %g\n",k,l,Econ[k][l]) ;
	   }
	   printf("\n") ;
	 */


	/* done */

}
__device__ void GPU_tetrad_to_coordinate(double Econ[NDIM][NDIM], double K_tetrad[NDIM],
			  double K[NDIM])
{
	int l;
	for (l = 0; l < 4; l++) {
		K[l] = Econ[0][l] * K_tetrad[0] +
		    Econ[1][l] * K_tetrad[1] +
		    Econ[2][l] * K_tetrad[2] + Econ[3][l] * K_tetrad[3];
	}

	return;
}

__device__ double GPU_delta(int i, int j)
{
	if (i == j)
		return (1.);
	else
		return (0.);
}
__device__ void GPU_normalize(double *vcon, double Gcov[NDIM][NDIM])
{
	int k, l;
	double norm;

	norm = 0.;
	for (k = 0; k < 4; k++)
		for (l = 0; l < 4; l++)
			norm += vcon[k] * vcon[l] * Gcov[k][l];

	norm = sqrt(fabs(norm));
	for (k = 0; k < 4; k++)
		vcon[k] /= norm;

	return;
}
__device__ void GPU_project_out(double *vcona, double *vconb, double Gcov[NDIM][NDIM])
{

	double adotb, vconb_sq;
	int k, l;

	vconb_sq = 0.;
	for (k = 0; k < 4; k++)
		for (l = 0; l < 4; l++)
			vconb_sq += vconb[k] * vconb[l] * Gcov[k][l];
	adotb = 0.;
	for (k = 0; k < 4; k++)
		for (l = 0; l < 4; l++)
			adotb += vcona[k] * vconb[l] * Gcov[k][l];

	for (k = 0; k < 4; k++)
		vcona[k] -= vconb[k] * adotb / vconb_sq;
	return;
}




/*THIS SECTION HAS BEEN RESERVED FOR TRACK_SUPER_PHOTON FUNCTION AND ITS DEPENDENCIES	*/
__device__ void GPU_track_super_photon(struct of_photon *ph, struct of_spectrum * d_spect, double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, int round_scat, int photon_index, int instant_partition)
{
	int bound_flag;
	double dtau_scatt, dtau_abs, dtau;
	double bi, bf;
	double alpha_scatti, alpha_scattf;
	double alpha_absi, alpha_absf;
	double dl, x1;
	double nu, Thetae, Ne, B, theta;
	struct of_photon php;
	double dtauK, frac;
	double bias = 0.;
	double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
	double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM],
	    Bcov[NDIM];
	int nstep = 0;
	/* quality control */
	if (isnan(ph->X[0]) ||
	    isnan(ph->X[1]) ||
	    isnan(ph->X[2]) ||
	    isnan(ph->X[3]) ||
	    isnan(ph->K[0]) ||
	    isnan(ph->K[1]) ||
	    isnan(ph->K[2]) || isnan(ph->K[3]) || ph->w == 0.) {
		printf("track_super_photon: bad input photon.\n");
		printf(
			"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
			ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
		return;
	}

	dtauK = L_UNIT / (ME * CL * CL / HPL);

	

	/* Initialize opacities */
	gcov_func(ph->X, Gcov);
	GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
			 Bcov, d_p);
	theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
	nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
	alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
	alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
	bi = GPU_bias_func(Thetae, ph->w, round_scat);
	/* Initialize dK/dlam */
	GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
	
	//while(0){

	while (!GPU_stop_criterion(ph)) {
		/* Save initial position/wave vector */
		Xi[0] = ph->X[0];
		Xi[1] = ph->X[1];
		Xi[2] = ph->X[2];
		Xi[3] = ph->X[3];
		Ki[0] = ph->K[0];
		Ki[1] = ph->K[1];
		Ki[2] = ph->K[2];
		Ki[3] = ph->K[3];
		dKi[0] = ph->dKdlam[0];
		dKi[1] = ph->dKdlam[1];
		dKi[2] = ph->dKdlam[2];
		dKi[3] = ph->dKdlam[3];
		E0 = ph->E0s;

		/* evaluate stepsize */
		dl = GPU_stepsize(ph->X, ph->K);

		/* step the geodesic */
		GPU_push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s), 0);
		

		if (GPU_stop_criterion(ph)){
			break;
		}

		/* allow photon to interact with matter, */
		gcov_func(ph->X, Gcov);
		GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov,
				 Bcon, Bcov, d_p);

		
				 
		if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {

			bound_flag = 0;
			if (Ne == 0.)
				bound_flag = 1;
			if (!bound_flag) {
				theta =
				    GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov,
						 B);
				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
				if (isnan(nu)) {
					printf(
						"isnan nu: track_super_photon dl,E0 %g %g\n",
						dl, E0);
					printf(
						"Xi, %g %g %g %g\n", Xi[0],
						Xi[1], Xi[2], Xi[3]);
					printf(
						"Ki, %g %g %g %g\n", Ki[0],
						Ki[1], Ki[2], Ki[3]);
					printf(
						"dKi, %g %g %g %g\n",
						dKi[0], dKi[1], dKi[2],
						dKi[3]);
				}
			}

			/* scattering optical depth along step */
			if (bound_flag || nu < 0.) {
				dtau_scatt =
					0.5 * alpha_scatti * dtauK * dl;
				dtau_abs = 0.5 * alpha_absi * dtauK * dl;
				alpha_scatti = alpha_absi = 0.;
				bias = 0.;
				bi = 0.;
			} else {
				alpha_scattf =
				    GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
				dtau_scatt =
				    0.5 * (alpha_scatti +
					   alpha_scattf) * dtauK * dl;
				alpha_scatti = alpha_scattf;
				/* absorption optical depth along step */
				alpha_absf =
				    GPU_alpha_inv_abs(nu, Thetae, Ne, B,
						  theta);
				dtau_abs =
				    0.5 * (alpha_absi +
					   alpha_absf) * dtauK * dl;

				alpha_absi = alpha_absf;

				bf = GPU_bias_func(Thetae, ph->w, round_scat);
				bias = 0.5 * (bi + bf);
				bi = bf;

			}

			x1 = -log(GPU_monty_rand());
			php.w =  ph->w / bias;
			//if(0){
			if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
				if (isnan(php.w) || isinf(php.w)) {
					printf(
						"w isnan in track_super_photon: Ne, bias, ph->w, php.w  %g, %g, %g, %g\n",
						Ne, bias, ph->w, php.w);
				}
				frac = x1 / (bias * dtau_scatt);

				/* Apply absorption until scattering event */
				dtau_abs *= frac;
				if (dtau_abs > 100){
					return;	/* This photon has been absorbed before scattering */
				}
				dtau_scatt *= frac;
				dtau = dtau_abs + dtau_scatt;

				//Do not include absorption in the scattering test
				#ifndef SCATTERING_TEST
					if (dtau_abs < 1.e-3){
						ph->w *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
					}
					else{
						ph->w *= exp(-dtau); 
					}
				#endif
				/* Interpolate position and wave vector to scattering event */

				GPU_push_photon(Xi, Ki, dKi, dl * frac, &E0,
					    0);
				ph->X[0] = Xi[0];
				ph->X[1] = Xi[1];
				ph->X[2] = Xi[2];
				ph->X[3] = Xi[3];
				ph->K[0] = Ki[0];
				ph->K[1] = Ki[1];
				ph->K[2] = Ki[2];
				ph->K[3] = Ki[3];
				ph->dKdlam[0] = dKi[0];
				ph->dKdlam[1] = dKi[1];
				ph->dKdlam[2] = dKi[2];
				ph->dKdlam[3] = dKi[3];
				ph->E0s = E0;
				

				/* Get plasma parameters at new position */
				gcov_func(ph->X, Gcov);
				GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae,
						 &B, Ucon, Ucov, Bcon,
						 Bcov, d_p);
				if (Ne > 0.) {
					GPU_scatter_super_photon(ph, &php, Ne,
							     Thetae, B,
							     Ucon, Bcon,
							     Gcov);
					if (ph->w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
						return;
					}
					if(php.w > 0){
						atomicAdd(&d_N_scatt, (round_scat + 1));
						int my_local_index = 0;
						if(round_scat > 0){
							for(int cum_sum = 0; cum_sum <= round_scat -1; cum_sum++){
								my_local_index += d_num_scat_phs[cum_sum];
							}
								my_local_index += atomicAdd(&d_num_scat_phs[round_scat], 1);
						}
						else{
							my_local_index = atomicAdd(&d_num_scat_phs[0], 1);
						}
						memcpy(&scat_ofphoton[my_local_index], &php, sizeof(struct of_photon));

						if(scat_ofphoton[my_local_index].w != php.w){
							printf("In GPU_track_super_photon, both weights should be the same! (%le, %le), %d\n", scat_ofphoton[my_local_index].w, php.w, my_local_index);
						}
					}
				}
				theta =
				    GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov,
						 B);
				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
				if (nu < 0.) {
					alpha_scatti = alpha_absi = 0.;
				} else {
					alpha_scatti =
					    GPU_alpha_inv_scatt(nu, Thetae,
							    Ne, d_table_ptr);
					alpha_absi =
					    GPU_alpha_inv_abs(nu, Thetae, Ne,
							  B, theta);
				}
				bi = GPU_bias_func(Thetae, ph->w, round_scat);

				ph->tau_abs += dtau_abs;
				ph->tau_scatt += dtau_scatt;

			} else {
				if (dtau_abs > 100){
					return;	/* This photon has been absorbed */
				}
				ph->tau_abs += dtau_abs;
				ph->tau_scatt += dtau_scatt;
				dtau = dtau_abs + dtau_scatt;

				//Do not include absorption in the scattering test
				#ifndef SCATTERING_TEST
					if (dtau < 1.e-3){
							ph->w *= (1. -dtau / 24. * (24. -dtau * (12. - dtau *(4. -dtau)))); //taylor expansion
					}else{
							ph->w *= exp(-dtau); 
					}
				#endif
			}
		}

		nstep++;

		/* signs that something's wrong w/ the integration */
		if (nstep > MAXNSTEP) {
			printf(
				"X1,X2,K1,K2, nu, bias,: %g, %g, %g, %g, %g, %g\n",
				ph->X[1], ph->X[2], ph->K[1], ph->K[2], nu, bias);
			break;
		}
	}

// 	/* accumulate result in spectrum on escape */
	if(1){
	//if ( GPU_record_criterion(ph) && nstep < MAXNSTEP){
		 GPU_record_super_photon(ph, d_spect);
	}
	/* done! */
	return;
}


__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM], double * d_p)
{
	int i, j, k;
	double del[NDIM];
	double rho, uu;
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	double gcon[NDIM][NDIM], coeff[8];

	//checks if it's within the grid
	if (X[1] < d_startx[1] ||
	    X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {

		*Ne = 0.;

		return;
	}

	// Finds out i and j index as well as fraction displacement del from the coordinates X[1], X[2], X[3]
	//Xtoij(X, &i, &j, del);
	GPU_Xtoijk(X, &i, &j, &k, del);
	//Xtoijk(X, &i, &j, &k, del);

	//Calculate the coeficient of displacement
	coeff[0] = (1. - del[1]) * (1. - del[2]) * (1. - del[3]);
	coeff[1] = (1. - del[1]) * (1. - del[2]) * del[3];
	coeff[2] = (1. - del[1]) * del[2] * del[3];
	coeff[3] = del[1] * del[2] * del[3];
	coeff[4] = (1. - del[1]) * del[2] * (1. - del[3]);
	coeff[5] = del[1] * (1. - del[2]) * (1. - del[3]);
	coeff[6] = del[1] * (1. - del[2]) * del[3];
	coeff[7] = del[1] * del[2] * (1. - del[3]);



	//interpolate based on the displacement
	rho = GPU_interp_scalar(d_p, KRHO, i, j, k, coeff);
	uu = GPU_interp_scalar(d_p, UU, i, j, k, coeff);
	*Ne = rho * NE_UNIT;
	*Thetae = uu / rho * d_thetae_unit;

	Bp[1] = GPU_interp_scalar(d_p, B1, i, j, k, coeff);
	Bp[2] = GPU_interp_scalar(d_p, B2, i, j, k, coeff);
	Bp[3] = GPU_interp_scalar(d_p, B3, i, j, k, coeff);

	Vcon[1] = GPU_interp_scalar(d_p, U1, i, j, k, coeff);
	Vcon[2] = GPU_interp_scalar(d_p, U2, i, j, k, coeff);
	Vcon[3] = GPU_interp_scalar(d_p, U3, i, j, k, coeff);

	gcon_func(X, gcov, gcon);
	
	/* Get Ucov */
	VdotV = 0.;
	for (i = 1; i < NDIM; i++)
		for (j = 1; j < NDIM; j++)
			VdotV += gcov[i][j] * Vcon[i] * Vcon[j];
	Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * gcon[0][0];
	for (i = 1; i < NDIM; i++){
		Ucon[i] = Vcon[i] - Vfac * gcon[0][i];
	}
	lower(Ucon, gcov, Ucov);

	/* Get B and Bcov */
	UdotBp = 0.;
	for (i = 1; i < NDIM; i++)
		UdotBp += Ucov[i] * Bp[i];
	Bcon[0] = UdotBp;
	for (i = 1; i < NDIM; i++)
		Bcon[i] = (Bp[i] + Ucon[i] * UdotBp) / Ucon[0];
	lower(Bcon, gcov, Bcov);

	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;

	#ifdef SCATTERING_TEST
		*Ne = 1.e-4/(SIGMA_THOMSON * (1.e5 - 1.));
		*Thetae = 4.;
		*B = 0;
		return;
	#endif

}



__device__ double GPU_bias_func(double Te, double w, int round_scatt)
{
	double bias, max, avg_num_scatt;
	#if(0)
		max = 0.5 * w / WEIGHT_MIN;
		//bias = Te * Te /(5. *d_max_tau_scatt);
		bias = fmax(1., d_bias_norm * Te * Te/d_max_tau_scatt);

		if (bias > max){
			bias = max;
		}

		return bias;
	#else
		return 1;
		max = 0.5 * w / WEIGHT_MIN;

		avg_num_scatt = d_N_scatt / (1. * d_N_superph_recorded + 1.);
		bias =
			100. * Te * Te / (d_bias_norm * d_max_tau_scatt *
					(avg_num_scatt + 2));

		//bias = Te * Te/(d_bias_norm * d_max_tau_scatt * 2.);

		if (bias < TP_OVER_TE)
			bias = TP_OVER_TE;
		if (bias > max)
			bias = max;
		//printf("bias = %le, max = %le, avg_num_scatt = %le\n", bias, max, avg_num_scatt);
		return bias / TP_OVER_TE;
	#endif
}
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[])
{
	int k;
	double lconn[NDIM][NDIM][NDIM];

	GPU_get_connection(X, lconn);

	for (k = 0; k < 4; k++) {

		dK[k] =
		    -2. * (Kcon[0] *
			   (lconn[k][0][1] * Kcon[1] +
			    lconn[k][0][2] * Kcon[2] +
			    lconn[k][0][3] * Kcon[3])
			   + Kcon[1] * (lconn[k][1][2] * Kcon[2] +
					lconn[k][1][3] * Kcon[3])
			   + lconn[k][2][3] * Kcon[2] * Kcon[3]
		    );

		dK[k] -=
		    (lconn[k][0][0] * Kcon[0] * Kcon[0] +
		     lconn[k][1][1] * Kcon[1] * Kcon[1] +
		     lconn[k][2][2] * Kcon[2] * Kcon[2] +
		     lconn[k][3][3] * Kcon[3] * Kcon[3]
		    );
	}


	return;
}

__device__ double GPU_stepsize(double X[NDIM], double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;
	#ifdef HAMR
		double x2_normal, stopx2_normal;
		x2_normal = (1. + X[2])/2.;
		stopx2_normal = 1.; 
		dlx2 = EPS * GSL_MIN(x2_normal, stopx2_normal - x2_normal) / (fabs(K[2]) + SMALL);
	#else
		dlx2 = EPS * GSL_MIN(X[2], d_stopx[2] - X[2]) / (fabs(K[2]) + SMALL);
	#endif

	dlx1 = EPS * X[1] / (fabs(K[1]) + SMALL);
	dlx3 = EPS / (fabs(K[3]) + SMALL);

	idlx1 = 1. / (fabs(dlx1) + SMALL);
	idlx2 = 1. / (fabs(dlx2) + SMALL);
	idlx3 = 1. / (fabs(dlx3) + SMALL);

	dl = 1. / (idlx1 + idlx2 + idlx3);

	return (dl);
}



// __device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
// 		 double dl, double *E0, int n)
// {
// 	double lconn[NDIM][NDIM][NDIM];
// 	double Kcont[NDIM], K[NDIM], dK;
// 	double Xcpy[NDIM], Kcpy[NDIM], dKcpy[NDIM];
// 	double Gcov[NDIM][NDIM], E1;
// 	double dl_2, err, errE;
// 	int i, k, iter;

// 	if (X[1] < d_startx[1])
// 		return;

// 	FAST_CPY(X, Xcpy);
// 	FAST_CPY(Kcon, Kcpy);
// 	FAST_CPY(dKcon, dKcpy);

// 	dl_2 = 0.5 * dl;
// 	/* Step the position and estimate new wave vector */
// 	// fprintf(stderr, "iteration = %d\n", n);
// 	// fprintf(stderr, "Inside Push Photon before X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", X[0], X[1], X[2], X[3]);
// 	// fprintf(stderr, "Inside Push Photon before k[0] = %lf, k[1] = %lf, k[2] = %lf, k[3] = %lf\n", K[0], K[1], K[2], K[3]);
// 	// fprintf(stderr, "dl = %lf, dKcon[0] = %lf, dKcon[1] = %lf, dKcon[2] = %lf, dKcon[3] = %lf\n", dl, dKcon[0], dKcon[1], dKcon[2], dKcon[3]);
// 	for (i = 0; i < NDIM; i++) {
// 		dK = dKcon[i] * dl_2;
// 		Kcon[i] += dK;
// 		K[i] = Kcon[i] + dK;
// 		X[i] += Kcon[i] * dl;
// 	}
// 	// fprintf(stderr, "Inside Push Photon after X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", X[0], X[1], X[2], X[3]);
// 	// fprintf(stderr, "Inside Push Photon after k[0] = %lf, k[1] = %lf, k[2] = %lf, k[3] = %lf\n", K[0], K[1], K[2], K[3]);

// 	// if(omp_get_thread_num() == 0){
// 	// 	fprintf(stderr, "X1 after = %le, K1 after = %le\n", X[1], K[1]);
// 	// }
// 	GPU_get_connection(X, lconn);

// 	/* We're in a coordinate basis so take advantage of symmetry in the connection */
// 	iter = 0;
// 	do {
// 		iter++;
// 		FAST_CPY(K, Kcont);

// 		err = 0.;
// 		for (k = 0; k < 4; k++) {
// 			dKcon[k] =
// 			    -2. * (Kcont[0] *
// 				   (lconn[k][0][1] * Kcont[1] +
// 				    lconn[k][0][2] * Kcont[2] +
// 				    lconn[k][0][3] * Kcont[3])
// 				   +
// 				   Kcont[1] * (lconn[k][1][2] * Kcont[2] +
// 					       lconn[k][1][3] * Kcont[3])
// 				   + lconn[k][2][3] * Kcont[2] * Kcont[3]
// 			    );

// 			dKcon[k] -=
// 			    (lconn[k][0][0] * Kcont[0] * Kcont[0] +
// 			     lconn[k][1][1] * Kcont[1] * Kcont[1] +
// 			     lconn[k][2][2] * Kcont[2] * Kcont[2] +
// 			     lconn[k][3][3] * Kcont[3] * Kcont[3]
// 			    );

// 			K[k] = Kcon[k] + dl_2 * dKcon[k];
// 			err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
// 		}
// 	} while (err > ETOL && iter < MAX_ITER);

// 	FAST_CPY(K, Kcon);

// 	gcov_func(X, Gcov);
// 	E1 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
// 	       Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);
// 	errE = fabs((E1 - (*E0)) / (*E0));

// 	if (n < 7
// 	    && (errE > 1.e-4 || err > ETOL || isnan(err) || isinf(err))) {
// 		FAST_CPY(Xcpy, X);
// 		FAST_CPY(Kcpy, Kcon);
// 		FAST_CPY(dKcpy, dKcon);
// 		GPU_push_photon(X, Kcon, dKcon, 0.5 * dl, E0, n + 1);
// 		GPU_push_photon(X, Kcon, dKcon, 0.5 * dl, E0, n + 1);
// 		E1 = *E0;
// 	}

// 	*E0 = E1;

// 	/* done! */
// }


/*Runge kutta 4th order*/
// __device__ void GPU_push_photon(double X[], double K[], double dK[], double dl, double * E0, int n)
// {

// 	int k;
// 	double lconn[NDIM][NDIM][NDIM];
// 	double Kt[NDIM], Xt[NDIM];
// 	double f1x[NDIM], f2x[NDIM], f3x[NDIM], f4x[NDIM];
// 	double f1k[NDIM], f2k[NDIM], f3k[NDIM], f4k[NDIM];
// 	double dl_2 = 0.5 * dl;

// 	for (k = 0; k < NDIM; k++)
// 		f1x[k] = K[k];

// 	GPU_get_connection(X, lconn);


// 	for (k = 0; k < NDIM; k++) {
// 		f1k[k] =
// 		    -2. * (K[0] *
// 			   (lconn[k][0][1] * K[1] + lconn[k][0][2] * K[2] +
// 			    lconn[k][0][3] * K[3]) +
// 			   K[1] * (lconn[k][1][2] * K[2] +
// 				   lconn[k][1][3] * K[3]) +
// 			   lconn[k][2][3] * K[2] * K[3]
// 		    );

// 		f1k[k] -=
// 		    (lconn[k][0][0] * K[0] * K[0] +
// 		     lconn[k][1][1] * K[1] * K[1] +
// 		     lconn[k][2][2] * K[2] * K[2] +
// 		     lconn[k][3][3] * K[3] * K[3]
// 		    );
// 	}

// 	for (k = 0; k < NDIM; k++) {
// 		Kt[k] = K[k] + dl_2 * f1k[k];
// 		f2x[k] = Kt[k];
// 		Xt[k] = X[k] + dl_2 * f1x[k];
// 	}

// 	GPU_get_connection(Xt, lconn);

// 	for (k = 0; k < NDIM; k++) {
// 		f2k[k] =
// 		    -2. * (Kt[0] *
// 			   (lconn[k][0][1] * Kt[1] +
// 			    lconn[k][0][2] * Kt[2] +
// 			    lconn[k][0][3] * Kt[3]) +
// 			   Kt[1] * (lconn[k][1][2] * Kt[2] +
// 				    lconn[k][1][3] * Kt[3]) +
// 			   lconn[k][2][3] * Kt[2] * Kt[3]
// 		    );

// 		f2k[k] -=
// 		    (lconn[k][0][0] * Kt[0] * Kt[0] +
// 		     lconn[k][1][1] * Kt[1] * Kt[1] +
// 		     lconn[k][2][2] * Kt[2] * Kt[2] +
// 		     lconn[k][3][3] * Kt[3] * Kt[3]
// 		    );
// 	}

// 	for (k = 0; k < NDIM; k++) {
// 		Kt[k] = K[k] + dl_2 * f2k[k];
// 		f3x[k] = Kt[k];
// 		Xt[k] = X[k] + dl_2 * f2x[k];
// 	}

// 	GPU_get_connection(Xt, lconn);

// 	for (k = 0; k < NDIM; k++) {
// 		f3k[k] =
// 		    -2. * (Kt[0] *
// 			   (lconn[k][0][1] * Kt[1] +
// 			    lconn[k][0][2] * Kt[2] +
// 			    lconn[k][0][3] * Kt[3]) +
// 			   Kt[1] * (lconn[k][1][2] * Kt[2] +
// 				    lconn[k][1][3] * Kt[3]) +
// 			   lconn[k][2][3] * Kt[2] * Kt[3]
// 		    );

// 		f3k[k] -=
// 		    (lconn[k][0][0] * Kt[0] * Kt[0] +
// 		     lconn[k][1][1] * Kt[1] * Kt[1] +
// 		     lconn[k][2][2] * Kt[2] * Kt[2] +
// 		     lconn[k][3][3] * Kt[3] * Kt[3]
// 		    );
// 	}

// 	for (k = 0; k < NDIM; k++) {
// 		Kt[k] = K[k] + dl * f3k[k];
// 		f4x[k] = Kt[k];
// 		Xt[k] = X[k] + dl * f3x[k];
// 	}

// 	GPU_get_connection(Xt, lconn);

// 	for (k = 0; k < NDIM; k++) {
// 		f4k[k] =
// 		    -2. * (Kt[0] *
// 			   (lconn[k][0][1] * Kt[1] +
// 			    lconn[k][0][2] * Kt[2] +
// 			    lconn[k][0][3] * Kt[3]) +
// 			   Kt[1] * (lconn[k][1][2] * Kt[2] +
// 				    lconn[k][1][3] * Kt[3]) +
// 			   lconn[k][2][3] * Kt[2] * Kt[3]
// 		    );

// 		f4k[k] -=
// 		    (lconn[k][0][0] * Kt[0] * Kt[0] +
// 		     lconn[k][1][1] * Kt[1] * Kt[1] +
// 		     lconn[k][2][2] * Kt[2] * Kt[2] +
// 		     lconn[k][3][3] * Kt[3] * Kt[3]
// 		    );
// 	}

// 	for (k = 0; k < NDIM; k++) {
// 		X[k] +=
// 		    0.166666666666667 * dl * (f1x[k] +
// 					      2. * (f2x[k] + f3x[k]) +
// 					      f4x[k]);
// 		K[k] +=
// 		    0.166666666666667 * dl * (f1k[k] +
// 					      2. * (f2k[k] + f3k[k]) +
// 					      f4k[k]);
// 	}

// 	GPU_init_dKdlam(X, K, dK);

// 	/* done */
// }




// // //This one below is from gpu_monty
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],  double dl,
	double *E0, int n)
{
        double lconn[NDIM][NDIM][NDIM];
        double Kcont[NDIM], K[NDIM], dK;
        double Gcov[NDIM][NDIM];
        double dl_2, err;
        int i, k, iter;

        if (X[1] < d_startx[1]) return;

        dl_2 = 0.5 * dl;

        /* Step the position and estimate new wave vector */
        for (i = 0; i < NDIM; i++) {
                dK = dKcon[i] * dl_2;
                Kcon[i] += dK;
                K[i] = Kcon[i] + dK;
                X[i] += Kcon[i] * dl;
        }

        GPU_get_connection(X, lconn);

        /* We're in a coordinate basis so take advantage of symmetry in the connection */
        iter = 0;
        do {
                iter++;
                FAST_CPY(K, Kcont);

                err = 0.;
                for (k = 0; k < 4; k++) {
                        dKcon[k] =
                            -2. * (Kcont[0] *
                                   (lconn[k][0][1] * Kcont[1] +
                                    lconn[k][0][2] * Kcont[2] +
                                    lconn[k][0][3] * Kcont[3])
                                   +
                                   Kcont[1] * (lconn[k][1][2] * Kcont[2] +
                                               lconn[k][1][3] * Kcont[3])
                                   + lconn[k][2][3] * Kcont[2] * Kcont[3]
                            );

                        dKcon[k] -=
                            (lconn[k][0][0] * Kcont[0] * Kcont[0] +
                             lconn[k][1][1] * Kcont[1] * Kcont[1] +
                             lconn[k][2][2] * Kcont[2] * Kcont[2] +
                             lconn[k][3][3] * Kcont[3] * Kcont[3]
                            );

                        K[k] = Kcon[k] + dl_2 * dKcon[k];
                        err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
                }
        } while ((err > ETOL || isinf(err) || isnan(err)) && iter < MAX_ITER);
        FAST_CPY(K, Kcon);

		gcov_func(X, Gcov);
        *E0 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
               Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);

		/* done! */
}

// __device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],  double dl,
// 	double *E0, int n)
// {
//         double lconn[NDIM][NDIM][NDIM];
//         double Kcont[NDIM], K[NDIM], dK;
// 		double Xcpy[NDIM], Kcpy[NDIM], dKcpy[NDIM];
//         double Gcov[NDIM][NDIM], E1, errE;
//         double dl_2, err;
//         int i, k, iter;
// 		bool condition;
//         if (X[1] < d_startx[1]) return;
// 		FAST_CPY(X, Xcpy);
// 		FAST_CPY(Kcon, Kcpy);
// 		FAST_CPY(dKcon, dKcpy);
// 		do{
// 			dl_2 = 0.5 * dl;

// 			/* Step the position and estimate new wave vector */
// 			for (i = 0; i < NDIM; i++) {
// 					dK = dKcon[i] * dl_2;
// 					Kcon[i] += dK;
// 					K[i] = Kcon[i] + dK;
// 					X[i] += Kcon[i] * dl;
// 			}

// 			GPU_get_connection(X, lconn);

// 			/* We're in a coordinate basis so take advantage of symmetry in the connection */
// 			iter = 0;
// 			do {
// 					iter++;
// 					FAST_CPY(K, Kcont);

// 					err = 0.;
// 					for (k = 0; k < 4; k++) {
// 							dKcon[k] =
// 								-2. * (Kcont[0] *
// 									(lconn[k][0][1] * Kcont[1] +
// 										lconn[k][0][2] * Kcont[2] +
// 										lconn[k][0][3] * Kcont[3])
// 									+
// 									Kcont[1] * (lconn[k][1][2] * Kcont[2] +
// 												lconn[k][1][3] * Kcont[3])
// 									+ lconn[k][2][3] * Kcont[2] * Kcont[3]
// 								);

// 							dKcon[k] -=
// 								(lconn[k][0][0] * Kcont[0] * Kcont[0] +
// 								lconn[k][1][1] * Kcont[1] * Kcont[1] +
// 								lconn[k][2][2] * Kcont[2] * Kcont[2] +
// 								lconn[k][3][3] * Kcont[3] * Kcont[3]
// 								);

// 							K[k] = Kcon[k] + dl_2 * dKcon[k];
// 							err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
// 					}
// 			} while ((err > ETOL || isinf(err) || isnan(err)) && iter < MAX_ITER);

// 			FAST_CPY(K, Kcon);

// 			gcov_func(X, Gcov);
// 			E1 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
// 			Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);
// 			errE = fabs((E1 - (*E0)) / (*E0));
// 			condition =  (n < 7) && (errE > 1.e-4 || err > ETOL || isinf(err) || isnan(err));
// 			if(condition){
// 				FAST_CPY(Xcpy, X);
// 				FAST_CPY(Kcpy, Kcon);
// 				FAST_CPY(dKcpy, dKcon);
// 				dl = 0.5 * dl;
// 				n += 1;
// 			}
// 		}while(condition);
// 		*E0 = E1;
//         *E0 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
//                Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);

// 		/* done! */
// }


__device__ void GPU_scatter_super_photon(struct of_photon *ph, struct of_photon *php,double Ne, double Thetae, double B, double Ucon[NDIM], double Bcon[NDIM], double Gcov[NDIM][NDIM])
{
	double P[NDIM], Econ[NDIM][NDIM], Ecov[NDIM][NDIM], K_tetrad[NDIM], K_tetrad_p[NDIM], Bhatcon[NDIM], tmpK[NDIM];
	int k;

	/* quality control */
	if (isnan(ph->K[1])) {
		printf("scatter: bad input photon, the program should exit itself\n");
		//exit(0);
	}

	/* quality control */
	if (ph->K[0] > 1.e5 || ph->K[0] < 0. || isnan(ph->K[1])
	    || isnan(ph->K[0]) || isnan(ph->K[3])) {
		printf(
			"normalization problem, killing superphoton: %g \n",
			ph->K[0]);
		ph->K[0] = fabs(ph->K[0]);
		printf("X1,X2: %g %g\n", ph->X[1], ph->X[2]);
		ph->w = 0.;
		return;
	}

	/* make trial vector for Gram-Schmidt orthogonalization in make_tetrad */
	/* note that B is in cgs but Bcon is in code units */
	if (B > 0.) {
		for (k = 0; k < NDIM; k++)
			Bhatcon[k] = Bcon[k] / (B / B_UNIT);
	} else {
		for (k = 0; k < NDIM; k++)
			Bhatcon[k] = 0.;
		Bhatcon[1] = 1.;
	}

	/* make local tetrad */
	GPU_make_tetrad(Ucon, Bhatcon, Gcov, Econ, Ecov);

	/* transform to tetrad frame */
	GPU_coordinate_to_tetrad(Ecov, ph->K, K_tetrad);

	/* quality control */
	if (K_tetrad[0] > 1.e5 || K_tetrad[0] < 0. || isnan(K_tetrad[1])) {
		printf(
			"conversion to tetrad frame problem: %g %g\n",
			ph->K[0], K_tetrad[0]);
		printf("%g %g %g\n",ph->K[1], ph->K[2], ph->K[3]);
		printf("%g %g %g\n",K_tetrad[1], K_tetrad[2], K_tetrad[3]);
		printf("%g %g %g %g\n",Ucon[0], Ucon[1], Ucon[2], Ucon[3]);
		printf("%g %g %g %g\n",Bhatcon[0], Bhatcon[1], Bhatcon[2], Bhatcon[3]);
		printf("%g %g %g %g\n", Gcov[0][0], Gcov[0][1], Gcov[0][2], Gcov[0][3]) ;
		printf("%g %g %g %g\n", Gcov[1][0], Gcov[1][1], Gcov[1][2], Gcov[1][3]) ;
		printf("%g %g %g %g\n", Gcov[2][0], Gcov[2][1], Gcov[2][2], Gcov[2][3]) ;
		printf("%g %g %g %g\n", Gcov[3][0], Gcov[3][1], Gcov[3][2], Gcov[3][3]) ;
		printf("%g %g %g %g\n", Ecov[0][0], Ecov[0][1], Ecov[0][2], Ecov[0][3]) ;
		printf("%g %g %g %g\n", Ecov[1][0], Ecov[1][1], Ecov[1][2], Ecov[1][3]) ;
		printf("%g %g %g %g\n", Ecov[2][0], Ecov[2][1], Ecov[2][2], Ecov[2][3]) ;
		printf("%g %g %g %g\n", Ecov[3][0], Ecov[3][1], Ecov[3][2], Ecov[3][3]) ;
		printf("X1,X2: %g %g\n",ph->X[1],ph->X[2]) ;
		ph->w = 0.;
		return;
	}

	/* find the electron that we collided with */
	GPU_sample_electron_distr_p( K_tetrad, P, Thetae);

	/* given electron momentum P, find the new
	   photon momentum Kp */
	GPU_sample_scattered_photon( K_tetrad, P, K_tetrad_p);


	/* transform back to coordinate frame */
	GPU_tetrad_to_coordinate(Econ, K_tetrad_p, php->K);

	/* quality control */
	if (isnan(php->K[1])) {
		printf(
			"problem with conversion to coordinate frame\n");
		printf("%g %g %g %g\n", Econ[0][0], Econ[0][1],
			Econ[0][2], Econ[0][3]);
		printf("%g %g %g %g\n", Econ[1][0], Econ[1][1],
			Econ[1][2], Econ[1][3]);
		printf("%g %g %g %g\n", Econ[2][0], Econ[2][1],
			Econ[2][2], Econ[2][3]);
		printf("%g %g %g %g\n", Econ[3][0], Econ[3][1],
			Econ[3][2], Econ[3][3]);
		printf("%g %g %g %g\n", K_tetrad_p[0],
			K_tetrad_p[1], K_tetrad_p[2], K_tetrad_p[3]);
		php->w = 0;
		return;
	}

	if (php->K[0] < 0) {
		// printf("K0, K0p, Kp, P[0]: %g %g %g %g\n",
		// 	K_tetrad[0], K_tetrad_p[0], php->K[0], P[0]);
		php->w = 0.;
		return;
	}

	/* bookkeeping */
	K_tetrad_p[0] *= -1.;
	GPU_tetrad_to_coordinate(Ecov, K_tetrad_p, tmpK);

	php->E = php->E0s = -tmpK[0];

	php->L = tmpK[3];
	php->tau_abs = 0.;
	php->tau_scatt = 0.;
	php->b0 = B;

	php->X1i = ph->X[1];
	php->X2i = ph->X[2];
	php->X[0] = ph->X[0];
	php->X[1] = ph->X[1];
	php->X[2] = ph->X[2];
	php->X[3] = ph->X[3];
	php->ne0 = Ne;
	php->thetae0 = Thetae;
	php->E0 = ph->E;
	php->nscatt = ph->nscatt + 1;

	return;
}
/* input and vectors are contravariant (index up) */
__device__ void GPU_coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM], double K_tetrad[NDIM])
{
	int k;

	for (k = 0; k < 4; k++) {
		K_tetrad[k] = Ecov[k][0] * K[0] + Ecov[k][1] * K[1] +Ecov[k][2] * K[2] + Ecov[k][3] * K[3];
	}
}
__device__ void GPU_sample_electron_distr_p(double k[4], double p[4], double Thetae)
{
	double beta_e, mu, phi, cphi, sphi, gamma_e, sigma_KN;
	double K, sth, cth, x1, n0dotv0, v0, v1;
	double n0x, n0y, n0z;
	double v0x, v0y, v0z;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	int sample_cnt = 0;
	do {
		GPU_sample_beta_distr( Thetae, &gamma_e, &beta_e);
		mu = GPU_sample_mu_distr( beta_e);
		/* sometimes |mu| > 1 from roundoff error, fix it */
		if (mu > 1.)
			mu = 1.;
		else if (mu < -1.)
			mu = -1;

		/* frequency in electron rest frame */
		K = gamma_e * (1. - beta_e * mu) * k[0];

		/* Avoid problems at small K */
		if (K < 1.e-3) {
			sigma_KN = 1. - 2. * K;
		} else {

			/* Klein-Nishina cross-section / Thomson */
			sigma_KN =
			    (3. / (4. * K * K)) * (2. +
						   K * K * (1. +
							    K) / ((1. +
								   2. *
								   K) *
								  (1. +
								   2. *
								   K)) +
						   (K * K - 2. * K -
						    2.) / (2. * K) *
						   log(1. + 2. * K));
		}

		x1 = GPU_monty_rand();

		sample_cnt++;

		if (sample_cnt > 10000000) {
			printf(
				"in sample_electron mu, gamma_e, K, sigma_KN, x1: %g %g %g %g %g %g\n",
				Thetae, mu, gamma_e, K, sigma_KN, x1);
			/* This is a kluge to prevent stalling for large values of \Theta_e */
			Thetae *= 0.5;
			sample_cnt = 0;
		}

	} while (x1 >= sigma_KN);

	/* first unit vector for coordinate system */
	v0x = k[1];
	v0y = k[2];
	v0z = k[3];
	v0 = sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
	v0x /= v0;
	v0y /= v0;
	v0z /= v0;

	/* pick zero-angle for coordinate system */
	//gsl_ran_dir_3d(r, &n0x, &n0y, &n0z);
	generate_random_direction( &n0x, &n0y, &n0z);
	n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

	/* second unit vector */
	v1x = n0x - (n0dotv0) * v0x;
	v1y = n0y - (n0dotv0) * v0y;
	v1z = n0z - (n0dotv0) * v0z;

	/* normalize */
	v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
	v1x /= v1;
	v1y /= v1;
	v1z /= v1;

	/* find one more unit vector using cross product;
	   this guy is automatically normalized */
	v2x = v0y * v1z - v0z * v1y;
	v2y = v0z * v1x - v0x * v1z;
	v2z = v0x * v1y - v0y * v1x;

	/* now resolve new momentum vector along unit vectors 
	   and create a four-vector $p$ */
	phi = GPU_monty_rand() * 2. * M_PI;	/* orient uniformly */  
	sphi = sin(phi);
	cphi = cos(phi);
	cth = mu;
	sth = sqrt(1. - mu * mu);

	p[0] = gamma_e;
	p[1] =
	    gamma_e * beta_e * (cth * v0x +
				sth * (cphi * v1x + sphi * v2x));
	p[2] =
	    gamma_e * beta_e * (cth * v0y +
				sth * (cphi * v1y + sphi * v2y));
	p[3] =
	    gamma_e * beta_e * (cth * v0z +
				sth * (cphi * v1z + sphi * v2z));

	if (beta_e < 0) {
		printf("betae error: %g %g %g %g\n",
			p[0], p[1], p[2], p[3]);
	}

	return;
}
__device__ void GPU_sample_beta_distr(double Thetae, double *gamma_e, double *beta_e)
{
	double y;

	/* checked */
	y = GPU_sample_y_distr( Thetae);

	/* checked */
	*gamma_e = y * y * Thetae + 1.;
	*beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));

	return;

}
__device__ double GPU_sample_y_distr(double Thetae)
{

	double S_3, pi_3, pi_4, pi_5, pi_6, y, x1, x2, x, prob;
	double num, den;

	pi_3 = sqrt(M_PI) / 4.;
	pi_4 = sqrt(0.5 * Thetae) / 2.;
	pi_5 = 3. * sqrt(M_PI) * Thetae / 8.;
	pi_6 = Thetae * sqrt(0.5 * Thetae);

	S_3 = pi_3 + pi_4 + pi_5 + pi_6;

	pi_3 /= S_3;
	pi_4 /= S_3;
	pi_5 /= S_3;
	pi_6 /= S_3;
	do {
		x1 = GPU_monty_rand();
		if (x1 < pi_3) {
			x = chi_square(3);
		} else if (x1 < pi_3 + pi_4) {
			x = chi_square(4);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			x = chi_square(5);
		} else {
			x = chi_square(6);
		}

		/* this translates between defn of distr in
		   Canfield et al. and standard chisq distr */
		y = sqrt(x / 2);

		x2 = GPU_monty_rand();
		num = sqrt(1. + 0.5 * Thetae * y * y);
		den = (1. + y * sqrt(0.5 * Thetae));

		prob = num / den;

	} while (x2 >= prob);
	return (y);
}

__device__ double GPU_sample_mu_distr(double beta_e)
{
	double mu, x1, det;

	x1 = GPU_monty_rand();
	det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * x1;
	if (det < 0.)
		printf("det < 0  %g %g\n\n", beta_e, x1);
	mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}
__device__ void GPU_sample_scattered_photon(double k[4], double p[4], double kp[4])
{
	double ke[4], kpe[4];
	double k0p;
	double n0x, n0y, n0z, n0dotv0, v0x, v0y, v0z, v1x, v1y, v1z, v2x,
	    v2y, v2z, v1, dir1, dir2, dir3;
	double cth, sth, phi, cphi, sphi;

	/* boost into the electron frame
	   ke == photon momentum in elecron frame */
	GPU_boost(k, p, ke);
	if (ke[0] > 1.e-4) {
		k0p = GPU_sample_klein_nishina( ke[0]);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = GPU_sample_thomson();
	}
	sth = sqrt(fabs(1. - cth * cth));

	/* unit vector 1 for scattering coordinate system is
	   oriented along initial photon wavevector */
	// v0x = ke[1] / ke[0];
	// v0y = ke[2] / ke[0];
	// v0z = ke[3] / ke[0];

	// Explicitly compute kemag instead of using ke[0] to ensure that photon
  	// is created normalized and doesn't inherit light cone errors from the
  	// original superphoton
	double kemag = sqrt(ke[1]*ke[1] + ke[2]*ke[2] + ke[3]*ke[3]);
	v0x = ke[1]/kemag;
	v0y = ke[2]/kemag;
	v0z = ke[3]/kemag;
	
	/* randomly pick zero-angle for scattering coordinate system.
	   There's undoubtedly a better way to do this. */
	//gsl_ran_dir_3d(r, &n0x, &n0y, &n0z);
	generate_random_direction(&n0x, &n0y, &n0z); /*This currently matches gsl function used*/
	n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

	/* unit vector 2 */
	v1x = n0x - (n0dotv0) * v0x;
	v1y = n0y - (n0dotv0) * v0y;
	v1z = n0z - (n0dotv0) * v0z;
	v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
	v1x /= v1;
	v1y /= v1;
	v1z /= v1;

	/* find one more unit vector using cross product;
	   this guy is automatically normalized */
	v2x = v0y * v1z - v0z * v1y;
	v2y = v0z * v1x - v0x * v1z;
	v2z = v0x * v1y - v0y * v1x;

	/* now resolve new momentum vector along unit vectors */
	/* create a four-vector $p$ */
	/* solve for orientation of scattered photon */

	/* find phi for new photon */
	phi = 2. * M_PI * GPU_monty_rand();	
	sphi = sin(phi);
	cphi = cos(phi);

	p[1] *= -1.;
	p[2] *= -1.;
	p[3] *= -1.;

	dir1 = cth * v0x + sth * (cphi * v1x + sphi * v2x);
	dir2 = cth * v0y + sth * (cphi * v1y + sphi * v2y);
	dir3 = cth * v0z + sth * (cphi * v1z + sphi * v2z);

	kpe[0] = k0p;
	kpe[1] = k0p * dir1;
	kpe[2] = k0p * dir2;
	kpe[3] = k0p * dir3;
	
	/* transform k back to lab frame */
	GPU_boost(kpe, p, kp);

	/* quality control */
	if (kp[0] < 0 || isnan(kp[0])) {
		printf("in sample_scattered_photon:\n");
		printf("k0p[0] = %g\n", k0p);
		printf("kp[0], kpe[0]: %g %g\n", kp[0], kpe[0]);
		printf("kpe: %g %g %g %g\n", kpe[0], kpe[1],
			kpe[2], kpe[3]);
		printf("k:  %g %g %g %g\n", k[0], k[1], k[2],
			k[3]);
		printf("ke: %g %g %g %g\n", ke[0], ke[1], ke[2],
			ke[3]);
		printf("p:   %g %g %g %g\n", p[0], p[1], p[2],
			p[3]);
		printf("kp:  %g %g %g %g\n", kp[0], kp[1], kp[2],
			kp[3]);
		printf("phi = %g, cphi = %g, sphi = %g\n", phi, cphi, sphi);
		printf("cth = %g, sth = %g\n", cth, sth);
	}
	/* done! */
}
__device__ void GPU_boost(double v[4], double u[4], double vp[4])
{
	double g, V, n1, n2, n3, gm1;

	g = u[0];
	V = sqrt(fabs(1. - 1. / (g * g)));
	n1 = u[1] / (g * V + SMALL);
	n2 = u[2] / (g * V + SMALL);
	n3 = u[3] / (g * V + SMALL);
	gm1 = g - 1.;

	/* general Lorentz boost into frame u from lab frame */
	vp[0] = u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3];
	vp[1] =
	    -u[1] * v[0] + (1. + n1 * n1 * gm1) * v[1] +
	    n1 * n2 * gm1 * v[2] + n1 * n3 * gm1 * v[3];
	vp[2] =
	    -u[2] * v[0] + n2 * n1 * gm1 * v[1] + (1. +
						   n2 * n2 * gm1) * v[2] +
	    n2 * n3 * gm1 * v[3];
	vp[3] =
	    -u[3] * v[0] + n3 * n1 * gm1 * v[1] + n3 * n2 * gm1 * v[2] +
	    (1. + n3 * n3 * gm1) * v[3];

}
__device__ double GPU_sample_thomson()
{
	double x1, x2;

	do {

		x1 = 2. * GPU_monty_rand() - 1.;
		x2 = (3. / 4.) * GPU_monty_rand();

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}
__device__ double GPU_sample_klein_nishina(double k0)
{
	double k0pmin, k0pmax, k0p_tent, x1;
	int n = 0;

	/* a low efficiency sampling algorithm, particularly for large k0;
	   limiting efficiency is log(2 k0)/(2 k0) */
	k0pmin = k0 / (1. + 2. * k0);	/* at theta = Pi */
	k0pmax = k0;		/* at theta = 0 */
	do {

		/* tentative value */
		k0p_tent = k0pmin + (k0pmax - k0pmin) * GPU_monty_rand();

		/* rejection sample in box of height = kn(kmin) */
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= GPU_monty_rand();

		n++;

	} while (x1 >= GPU_klein_nishina(k0, k0p_tent));

	return (k0p_tent);
}
__device__ double GPU_klein_nishina(double a, double ap)
{
	double ch, kn;

	ch = 1. + 1. / a - 1. / ap;
	kn = (a / ap + ap / a - 1. + ch * ch) / (a * a);

	return (kn);
}

__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]){
	double interp;

	interp = coeff[0] * var[NPRIM_INDEX3D(mmenemonics, i, j, k)] + coeff[5] * var[NPRIM_INDEX3D(mmenemonics, i+1, j, k)] +
	coeff[4] * var[NPRIM_INDEX3D(mmenemonics, i, j + 1, k)] + coeff[7]  * var[NPRIM_INDEX3D(mmenemonics, i+1, j+1, k)] +
	coeff[1] * var[NPRIM_INDEX3D(mmenemonics, i, j, k+1)] + coeff[6] * var[NPRIM_INDEX3D(mmenemonics, i+1, j, k+1)] +
	coeff[2] * var[NPRIM_INDEX3D(mmenemonics, i, j+1, k+1)] + coeff[3] * var[NPRIM_INDEX3D(mmenemonics, i+1, j+1, k+1)];

	return interp;
}



__device__ void GPU_record_super_photon(struct of_photon *ph , struct of_spectrum* d_spect) {
    double lE, dx2;
    int iE, ix2;

    if (isnan(ph->w) || isnan(ph->E)) {
        printf("record isnan: %g %g\n", ph->w, ph->E);
        return;
    }

	/*TODO: FIX RACE CONDITION BY USING https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf slide 7
	taken from https://stackoverflow.com/questions/16785263/cuda-multiple-threads-writing-to-a-shared-variable*/
    // if (ph->tau_scatt > d_max_tau_scatt) {
    //    d_max_tau_scatt = ph->tau_scatt;
    // }

	d_max_tau_scatt = atomicMaxdouble(&d_max_tau_scatt, ph->tau_scatt);
    // Bin in x2 coordinate
	#ifdef HAMR
		dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
		ix2 = ((ph->X[2]) < 0) ? (int)((1 +ph->X[2]) / dx2) : (int)((d_stopx[2] - ph->X[2]) / dx2);
	#else
	    dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
    	ix2 = (ph->X[2] < 0.5 * (d_startx[2] + d_stopx[2])) ? (int)(ph->X[2] / dx2) : (int)((d_stopx[2] - ph->X[2]) / dx2);
	#endif
    if (ix2 < 0 || ix2 >= N_THBINS){
        return;
	}

    // Get energy bin
    lE = log(ph->E);
    iE = (int)((lE - lE0) / dlE + 2.5) - 2;

    if (iE < 0 || iE >= N_EBINS){
	    return;
	}

    atomicAdd(&d_N_superph_recorded, 1);
    //atomicAdd(&d_N_scatt, ph->nscatt);

	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].dNdlE), ph->w);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].dEdlE), ph->w * ph->E);
    atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].tau_abs), ph->w * ph->tau_abs);
    atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].tau_scatt), ph->w * ph->tau_scatt);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X1iav), ph->w * ph->X1i);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X2isq), ph->w * (ph->X2i * ph->X2i));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X3fsq), ph->w * (ph->X[3] * ph->X[3]));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].ne0),  ph->w * (ph->ne0));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].b0), ph->w * (ph->b0));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].thetae0),ph->w * (ph->thetae0));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].nscatt),  ph->nscatt);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].nph), 1);
}

__device__ __forceinline__ double atomicMaxdouble(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}




__device__ int findPhotonIndex(const unsigned long long *cumulativeArray, int arraySize, unsigned long long photon_index) {
    int left = 0;
    int right = arraySize - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (cumulativeArray[mid] > photon_index) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left; // `left` is the smallest index where cumulativeArray[left] > photon_index
}