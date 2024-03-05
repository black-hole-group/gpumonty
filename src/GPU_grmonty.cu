#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

extern "C"
{
#include "decs.h"
}

#include "gpu_header.h"

#include "defs_CUDA.h"



/*TODO: PASSAR AS VARIAVEIS STRUCT OF_PHOTON para o device*/
/*TODO: Gotta check if all the random functions are working fine, GPU_montyrand, chisquared() and so on*/
// Define the device random number generator state
__device__ curandStateMtgp32 my_curand_state;

__device__ struct local_track_var{
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
	struct of_photon * ph;
	struct of_photon php;
};
__global__ void test_struct_data2(){
	//printf("startx = (%lf, %lf, %lf, %lf), dx = (%lf, %lf, %lf, %lf)\n", d_startx[0], d_startx[1], d_startx[2], d_startx[3], d_dx[0], d_dx[1], d_dx[2], d_dx[3]);
	/*Done transfering geom*/
	// for (int i = 0; i < (N_ESAMP +1); i++){
	// 	printf("d_wgt[%d] = %le\n", i, d_wgt[i]);
	// }
}

void launch_loop(struct of_photon ph, int quit_flag, time_t time, double * p){
	/*Copying global variables*/
	int seed = 139 + time;
	struct of_spectrum spect[N_THBINS][N_EBINS] = { };
    struct of_spectrum* d_spect;
    gpuErrchk(cudaMalloc((void**)&d_spect, N_THBINS * N_EBINS * sizeof(struct of_spectrum)));

    cudaMemcpyToSymbol(d_N1, &N1, sizeof(int));
	cudaMemcpyToSymbol(d_Ns, &Ns, sizeof(int));
    cudaMemcpyToSymbol(d_N2, &N2, sizeof(int));
    cudaMemcpyToSymbol(d_N3, &N3, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_startx, &startx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_stopx, &stopx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_a, &a, sizeof(double));
	cudaMemcpyToSymbol(d_thetae_unit, &Thetae_unit, sizeof(double));
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
	//size_t limit = 0;
	//cudaDeviceSetLimit(cudaLimitStackSize, 1024);
	//cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    //cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    //printf("cudaLimitStackSize: %u\n", (unsigned)limit);
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

	/*Allocating memory for array of local_track_variables*/
	struct local_track_var *local_track_vars;
	gpuErrchk(cudaMalloc(&local_track_vars, MAXNSTEP * sizeof(struct local_track_var)));

	/*Calling the main function*/
	int N_superph_made_cpu = 0;
	int *N_superph_made_gpu;
	gpuErrchk(cudaMalloc((void **)&N_superph_made_gpu, sizeof(int)));


	/*Initializing the Random number generator*/
	curandStateMtgp32 *devMTGPStates;
    mtgp32_kernel_params *devKernelParams;

	/* Allocate space for prng states on device */
    gpuErrchk(cudaMalloc((void **)&devMTGPStates, 64 *
              sizeof(curandStateMtgp32)));
	/* Allocate space for MTGP kernel parameters */
    gpuErrchk(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));

    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
    /* Initialize one state per thread block */
    curandMakeMTGP32KernelState(devMTGPStates,
                mtgp32dc_params_fast_11213, devKernelParams, N_THREADS, seed);

	/*Calling the main function*/
    GPU_mainloop<<<1,N_THREADS>>>(devMTGPStates, ph, time, d_geom, d_p, d_table_ptr, local_track_vars, N_superph_made_gpu, d_spect);
	cudaDeviceSynchronize();
	cudaMemcpyErrorCheck(&N_superph_made_cpu, N_superph_made_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&N_superph_made_cpu, N_superph_made_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyErrorCheck(spect, d_spect, N_EBINS * N_THBINS * sizeof(of_spectrum), cudaMemcpyDeviceToHost);

	report_spectrum(N_superph_made_cpu, spect);
	
	cudaDeviceSynchronize();
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	//printf(cudaStatus);

	if (cudaStatus != cudaSuccess) {
    	printf("CUDA Error: %s\n", cudaGetErrorString(cudaStatus));
    // Additional error handling code here, if needed
}
    cudaFree(devKernelParams);
    cudaFree(devMTGPStates);
	cudaFree(N_superph_made_gpu);
	cudaFree(d_table_ptr);
	cudaFree(local_track_vars);
}


__global__ void GPU_mainloop(curandStateMtgp32 *state, struct of_photon ph, time_t time, struct of_geom * d_geom, double * d_p, double * d_table_ptr, struct local_track_var * local_track_vars, int *super_photon_made, struct of_spectrum* d_spect)
{

	struct of_photon d_ph = ph;
	int quit_flag = 0;
	// for(int i = 0; i < (NT+1); i++)for(int j = 0; j < (NW+1); j++){
	// 	printf("Table[%d][%d] = %le\n", i, j, d_table_ptr[j + (NT+1) * i]);
	// }
	/* spectral bin parameters */
	d_dlE = 0.25;		/* bin width */
	d_lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */
	d_N_scatt = 0;
	d_N_superph_recorded = 0;
	time_t starttime = time;
	time_t currtime;
	int zi = 0;
    while(1){
        /*First thing we should do is make super photon*/
        if (!quit_flag){
            GPU_make_super_photon(state, &d_ph, &quit_flag, d_geom, d_p, &zi);
        }

			//printf("quit_flag after= %d", quit_flag);
			if (quit_flag){
				break;
            }

			/* push them around */
			//printf("it is doing its thing\n");
 			
			//GPU_track_super_photon(state, &d_ph, d_p, local_track_vars, 0, d_table_ptr, d_spect);

			/* step */
		 	//atomicAdd(super_photon_made, 1);
			}

}
__device__ void GPU_make_super_photon(curandStateMtgp32 *state, struct of_photon *ph, int *quit_flag, struct of_geom * d_geom, double * d_p, int * zi)
{
    int n2gen = -1;
    double dnmax;
    int zone_i, zone_j, zone_k;

	/*if the number of photons is not negative, e.g there are super photons in the zone
	then, continue the program, e.g, sample the zone photon and then pushes, this function is only checking the need to generate this zone photon
	*/
	while (n2gen <= 0) {
		n2gen = GPU_get_zone(state, &zone_i, &zone_j, &zone_k, &dnmax, d_geom, d_p, zi);
	}

	n2gen--;
	/*Before continue sampleing the zone photon, check if we reached the final radial zone
	if so, just leave the program.*/
	if (zone_i == d_N1)
		*quit_flag = 1;
	else
		*quit_flag = 0;
	if (*quit_flag != 1) {
		/* Initialize the superphoton energy, direction, weight, etc. */
		GPU_sample_zone_photon(state, zone_i, zone_j, zone_k, dnmax, ph, d_geom, d_p);
	}

	return;
}
__device__ int GPU_get_zone(curandStateMtgp32 *state, int *i, int *j, int *k, double *dnmax, struct of_geom * d_geom, double * d_p, int * zi)
{
/* Return the next zone and the number of superphotons that need to be		*
 * generated in it.								*/
	int in2gen;
	double n2gen;
	int tid = threadIdx.x;
	int offset = 0;
	// The fact this is static int means it is only set to 0 when function is called first time
	// meanwhile, the value is updated and kept in memory
	//TODO: Gotta check if max_threads is N_THREADS -1
	if (tid == N_THREADS - 1){
		offset = d_N1%N_THREADS;
	}	
	//static int zi = -1;
	if (*zi == 0){
		*zi = (int)(d_N1/N_THREADS) * threadIdx.x;
	}
	static int zj = 0;
	static int zk = -1;
	zone_flag = 1;
	zk++;
	if(zk >= d_N3){
		zk = 0;
		zj++;
		if (zj >= d_N2) {
			zj = 0;
			*zi = *zi + 1;
			if (*zi >= ((d_N1/N_THREADS * tid) + d_N1/N_THREADS) + offset) {
				in2gen = 1;
				*i = d_N1;
				return 1;
			}
		}
	}
	GPU_init_zone(*zi, zj, zk, &n2gen, dnmax, d_geom, d_p);
	/*in2gen is the number of photons that need to be generated in the next zone*/
	if (fmod(n2gen, 1.) > GPU_monty_rand(state)) {
		in2gen = (int) n2gen + 1;
	} else {
		in2gen = (int) n2gen;
	}

	*i = *zi;
	*j = zj;
	*k = zk;

	return in2gen;
}
__device__ void GPU_sample_zone_photon(curandStateMtgp32 *state, int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p)
{
/* Set all initial superphoton attributes */
	int l;
	int z = 0;
	double K_tetrad[NDIM], tmpK[NDIM], E, Nln;
	double nu, th, cth, sth, phi, sphi, cphi, jmax, weight;
	double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM], bhat[NDIM];
	static double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
	#if(HAMR)
	GPU_coord_hamr(i, j, z, CENT, ph -> X);
	#else
	coord(i, j, ph->X);
	#endif
    double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	Nln = lnu_max - lnu_min;

	GPU_get_fluid_zone(i, j, z, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);

	/* Sample from superphoton distribution in current simulation zone */

	do {
		nu = exp(GPU_monty_rand(state) * Nln + lnu_min);
		weight = GPU_linear_interp_weight(nu);
	} while (GPU_monty_rand(state) >
		 (GPU_F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / dnmax);
	ph->w = weight;
	jmax = GPU_jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);

	do {
		cth = 2. * GPU_monty_rand(state) - 1.;
		th = acos(cth);

	} while (GPU_monty_rand(state) >
		 GPU_jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax);

	sth = sqrt(1. - cth * cth);
	phi = 2. * M_PI * GPU_monty_rand(state);
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
		GPU_make_tetrad(Ucon, bhat, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcov, Econ, Ecov);
		zone_flag = 0;
	}

	GPU_tetrad_to_coordinate(Econ, K_tetrad, ph->K);

	K_tetrad[0] *= -1.;
	GPU_tetrad_to_coordinate(Ecov, K_tetrad, tmpK);

	ph->E = ph->E0 = ph->E0s = -tmpK[0];
	ph->L = tmpK[3];
	ph->tau_scatt = 0.;
	ph->tau_abs = 0.;
	ph->X1i = ph->X[1];
	ph->X2i = ph->X[2];
	ph->nscatt = 0;
	ph->ne0 = Ne;
	ph->b0 = Bmag;
	ph->thetae0 = Thetae;

	return;
}

__device__ double GPU_monty_rand(curandStateMtgp32 *state) {
	int tid = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	return curand_uniform_double(&state[tid]);
}



__device__ void GPU_coord_hamr(int i, int j, int z, int loc, double * X)
{
	X[0] = 0.0;
	int j_local = j;
	if (j < 0) j_local = -j - 1;
	//if (j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL2])) j_local = 2 * N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j;
	//if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && loc == FACE2) j_local = j;
	if (j >= d_N2*pow(1 + REF_2, 0)) j_local = 2 * d_N2*pow(1 + REF_2, 0) - 1 - j;
	if (j == d_N2*pow(1 + REF_2, 0) && loc == FACE2) j_local = j;
	if (loc == FACE1) {
		X[1] = d_startx[1] + i*d_dx[1];
		X[2] = d_startx[2] + (j_local + 0.5)*d_dx[2];
		X[3] = d_startx[3] + (z + 0.5)*d_dx[3];
	}
	else if (loc == FACE2) {
		X[1] = d_startx[1] + (i + 0.5)*d_dx[1];
		X[2] = d_startx[2] + j_local*d_dx[2];
		X[3] = d_startx[3] + (z + 0.5)*d_dx[3];
	}
	else if (loc == FACE3) {
		X[1] = d_startx[1] + (i + 0.5)*d_dx[1];
		X[2] = d_startx[2] + (j_local + 0.5)*d_dx[2];
		X[3] = d_startx[3] + z*d_dx[3];
	}
	else if (loc == CENT) {
		X[1] = d_startx[1] + (i + 0.5)*d_dx[1];
		X[2] = d_startx[2] + (j_local + 0.5)*d_dx[2];
		X[3] = d_startx[3] + (z + 0.5)*d_dx[3]; 
		//X[3] = 0;
	}
	else {
		X[1] = d_startx[1] + i*d_dx[1];
		X[2] = d_startx[2] + j_local*d_dx[2];
		X[3] = d_startx[3] + z*d_dx[3];
	}

	#if(!CARTESIAN)
	if (j < 0){
		X[2] = X[2] + 1;
		X[2] = -X[2];
		X[2] = X[2] - 1;
	}
	if (j == d_N2*pow(1 + REF_2, 0) && loc == FACE2){
	}
	else if (j >= d_N2*pow(1 + REF_2, 0)){
		X[2] = X[2] + 1;
		X[2] = 4. - X[2];
		X[2] = X[2] - 1;
	}
	#endif
	//if (j == N2 * pow(1 + REF_2, block[n][AMR_LEVEL2]))printf( "test: %f %d \n" ,X[2], loc==FACE2);
	return;
}
__device__ void GPU_get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM], struct of_geom * d_geom, double * d_p)
{
	int l, m;
	double Ucov[NDIM], Bcov[NDIM];
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	*Ne = d_p[DEVICE_NPRIM_INDEX3D(KRHO, i, j, k)] * NE_UNIT;
	*Thetae = d_p[DEVICE_NPRIM_INDEX3D(UU, i, j, k)] / (*Ne) * NE_UNIT * d_thetae_unit;

	Bp[1] = d_p[DEVICE_NPRIM_INDEX3D(B1, i, j, k)];
	Bp[2] = d_p[DEVICE_NPRIM_INDEX3D(B2, i, j, k)];
	Bp[3] = d_p[DEVICE_NPRIM_INDEX3D(B3, i, j, k)];

	Vcon[1] = d_p[DEVICE_NPRIM_INDEX3D(U1, i, j, k)];
	Vcon[2] = d_p[DEVICE_NPRIM_INDEX3D(U2, i, j, k)];
	Vcon[3] = d_p[DEVICE_NPRIM_INDEX3D(U3, i, j, k)];

	/* Get Ucov */
	VdotV = 0.;
	for (l = 1; l < NDIM; l++)
		for (m = 1; m < NDIM; m++)
			VdotV += d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcov[l][m] * Vcon[l] * Vcon[m];
	Vfac = sqrt(-1. / d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcon[0][0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcon[0][0];
	for (l = 1; l < NDIM; l++){
		Ucon[l] = Vcon[l] - Vfac * d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcon[0][l];
		//printf("Ucon[%d] = %le, Vcon[%d] = %le, Vfac = %le, geom[0][%d] = %le\n", l, Ucon[l], l, Vcon[l], Vfac, l, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcon[0][l]);
	}
	GPU_lower(Ucon, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcov, Ucov);
	/* Get B and Bcov */
	UdotBp = 0.;
	for (l = 1; l < NDIM; l++)
		UdotBp += Ucov[l] * Bp[l];
	Bcon[0] = UdotBp;
	for (l = 1; l < NDIM; l++){
		Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
	}
	GPU_lower(Bcon, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcov, Bcov);
	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;
	if (isnan(*B)){
		printf("i = %d, j = %d, k = %d\n", i, j, k);
		printf( "VdotV = %le\n", VdotV);
		printf( "Vfac = %lf\n", Vfac);
		for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcon[%d][%d]: %lf\n", a, b, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcon[a][b]);
		for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcov[%d][%d]: %lf\n", a, b, d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].gcov[a][b]);
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
__device__ double GPU_F_eval(double Thetae, double Bmag, double nu)
{

	double K, x;
	__device__ double GPU_linear_interp_F(double);

	K = KFAC * nu / (Bmag * Thetae * Thetae);
	if (K > KMAX) {
		return 0.;
	} else if (K < KMIN) {
		/* use a good approximation */
		x = pow(K, 0.333333333333333333);
		return (x * (37.67503800178 + 2.240274341836 * x));
	} else {
		return GPU_linear_interp_F(K);
	}
}
__device__ double GPU_K2_eval(double Thetae)
{

	__device__ double GPU_linear_interp_K2(double);

	if (Thetae < THETAE_MIN)
		return 0.;
	if (Thetae > TMAX)
		return 2. * Thetae * Thetae;

	return GPU_linear_interp_K2(Thetae);
}
__device__ double GPU_linear_interp_F(double K)
{
	double lK_min = log(KMIN);
    double dlK = log(KMAX / KMIN) / (N_ESAMP);
	double result;
	int i;
	double di, lK;
	lK = log(K);
	di = (lK - lK_min) * dlK;
	i = (int) di;
	di = di - i;
	result = exp((1. - di) * d_F[i] + di * d_F[i + 1]);
	return result;
}
__device__ double GPU_jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta)
{
	double K2, nuc, nus, x, f, j, sth, xp1, xx;
	__device__ double GPU_K2_eval(double Thetae);

	if (Thetae < THETAE_MIN)
		return 0.;

	K2 = GPU_K2_eval(Thetae);

	nuc = EE * B / (2. * M_PI * ME * CL);
	sth = sin(theta);
	nus = (2. / 9.) * nuc * Thetae * Thetae * sth;
	if (nu > 1.e12 * nus)
		return (0.);
	x = nu / nus;
	xp1 = pow(x, 1. / 3.);
	xx = sqrt(x) + CST * sqrt(xp1);
	f = xx * xx;
	j = (M_SQRT2 * M_PI * EE * EE * Ne * nus / (3. * CL * K2)) * f *
	    exp(-xp1);

	return (j);
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
		GPU_lower(Econ[k], Gcov, Ecov[k]);
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
	for (int i = 0; i < NDIM; i++){
		//fprintf(stderr, "k[%d] = %le\n", i, K[i]);
		//printf("Econ[0][%d] = %le,Econ[1][%d] = %le,Econ[2][%d] = %le,Econ[3][%d] = %le\n", i, Econ[0][i], i, Econ[1][i], i, Econ[2][i], i, Econ[3][i]);
	}
	return;
}
__device__ void GPU_lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov)
{

	ucov[0] = Gcov[0][0] * ucon[0]
	    + Gcov[0][1] * ucon[1]
	    + Gcov[0][2] * ucon[2]
	    + Gcov[0][3] * ucon[3];
	ucov[1] = Gcov[1][0] * ucon[0]
	    + Gcov[1][1] * ucon[1]
	    + Gcov[1][2] * ucon[2]
	    + Gcov[1][3] * ucon[3];
	ucov[2] = Gcov[2][0] * ucon[0]
	    + Gcov[2][1] * ucon[1]
	    + Gcov[2][2] * ucon[2]
	    + Gcov[2][3] * ucon[3];
	ucov[3] = Gcov[3][0] * ucon[0]
	    + Gcov[3][1] * ucon[1]
	    + Gcov[3][2] * ucon[2]
	    + Gcov[3][3] * ucon[3];

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
__device__ static void GPU_init_zone(int i, int j, int k, double *nz, double *dnmax, struct of_geom * d_geom, double * d_p)
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
	GPU_get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, d_geom, d_p);

	if (Ne == 0. || Thetae < THETAE_MIN) {
		*nz = 0.;
		*dnmax = 0.;
		return;
	}

	lbth = log(Bmag * Thetae * Thetae);

	dl = (lbth - lb_min) / dlb;
	l = (int) dl;
	dl = dl - l;
	if (l < 0) {
		*dnmax = 0.;
		*nz = 0.;
		return;
	} else if (l >= NINT) {
		printf(
			"warning: outside of nint table range %g...change in harm_utils.c\n",
			Bmag * Thetae * Thetae);
		printf( "lbth = %le, lb_min = %le, dlb = %le l = %d\n", lbth, lb_min, dlb, l);
		ninterp = 0.;
		*dnmax = 0.;
		for (l = 0; l <= N_ESAMP; l++) {
			dn = GPU_F_eval(Thetae, Bmag,
				    exp(j * dlnu +
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

	K2 = GPU_K2_eval(Thetae);
	if (K2 == 0.) {
		*nz = 0.;
		*dnmax = 0.;
		return;
	}

	*nz = d_geom[DEVICE_SPATIAL_INDEX2D(i,j)].g * Ne * Bmag * Thetae * Thetae * ninterp / K2;
	if (*nz > d_Ns * log(NUMAX / NUMIN)) {
		printf(
			"Something very wrong in zone %d %d: \nB=%g  Thetae=%g  K2=%g  ninterp=%g\n\n",
			i, j, Bmag, Thetae, K2, ninterp);
		*nz = 0.;
		*dnmax = 0.;
	}

	return;
}
__device__ double GPU_linear_interp_K2(double Thetae)
{

	int i;
	double di, lT;
	
	lT = log(Thetae);

	di = (lT - d_lT_min) * d_dlT;
	i = (int) di;
	di = di - i;

	return exp((1. - di) * d_K2[i] + di * d_K2[i + 1]);
}






/*THIS SECTION HAS BEEN RESERVED FOR TRACK_SUPER_PHOTON FUNCTION AND ITS DEPENDENCIES	*/
/*This is the main function that is working right now*/
// __device__ void GPU_track_super_photon(curandStateMtgp32 *state, struct of_photon *ph, double * d_p, struct local_track_var * local_track_vars, int recursive_index, double * d_table_ptr, struct of_spectrum* d_spect)
// {
// 	int max_recursions = MAXNSTEP;
// 	bool starting_recursion = false;
// 	bool jump_to_ending = false;
// 	int bound_flag;
// 	double dtau_scatt, dtau_abs, dtau;
// 	double bi, bf;
// 	double alpha_scatti, alpha_scattf;
// 	double alpha_absi, alpha_absf;
// 	double dl, x1;
// 	double nu, Thetae, Ne, B, theta;
// 	struct of_photon php;
// 	double dtauK, frac;
// 	double bias = 0.;
// 	double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
// 	double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM],
// 		Bcov[NDIM];
// 	int nstep = 0;
// 	/* quality control */
// 	if (isnan(ph->X[0]) ||
// 		isnan(ph->X[1]) ||
// 		isnan(ph->X[2]) ||
// 		isnan(ph->X[3]) ||
// 		isnan(ph->K[0]) ||
// 		isnan(ph->K[1]) ||
// 		isnan(ph->K[2]) || isnan(ph->K[3]) || ph->w == 0.) {
// 		printf( "track_super_photon: bad input photon.\n");
// 		printf(
// 			"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
// 			ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
// 			ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
// 		return;
// 	}

// 	dtauK = 2. * M_PI * L_UNIT / (ME * CL * CL / HBAR);

// 	/* Initialize opacities */
// 	GPU_gcov_func_hamr(ph->X, Gcov);
// 	GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
// 			Bcov, d_p);

// 	theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
// 	nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
// 	alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
// 	alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
// 	bi = GPU_bias_func(Thetae, ph->w);

// 	/* Initialize dK/dlam */
// 	GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
// 	recursive_round:
// 		while (!GPU_stop_criterion(state, ph) || jump_to_ending) {
// 			if(!jump_to_ending){
// 				/*****************************condition******************************/
// 				if(starting_recursion){
// 				starting_recursion = false;
// 				int bound_flag;
// 				double dtau_scatt, dtau_abs, dtau;
// 				double bi, bf;
// 				double alpha_scatti, alpha_scattf;
// 				double alpha_absi, alpha_absf;
// 				double dl, x1;
// 				double nu, Thetae, Ne, B, theta;
// 				struct of_photon php;
// 				double dtauK, frac;
// 				double bias = 0.;
// 				double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
// 				double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM],
// 					Bcov[NDIM];
// 				int nstep = 0;

// 				/* quality control */
// 				if (isnan(ph->X[0]) ||
// 					isnan(ph->X[1]) ||
// 					isnan(ph->X[2]) ||
// 					isnan(ph->X[3]) ||
// 					isnan(ph->K[0]) ||
// 					isnan(ph->K[1]) ||
// 					isnan(ph->K[2]) || isnan(ph->K[3]) || ph->w == 0.) {
// 					printf( "track_super_photon: bad input photon.\n");
// 					printf(
// 						"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
// 						ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
// 						ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
// 					return;
// 				}

// 				dtauK = 2. * M_PI * L_UNIT / (ME * CL * CL / HBAR);

// 				/* Initialize opacities */
// 				GPU_gcov_func_hamr(ph->X, Gcov);
// 				GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
// 						Bcov, d_p);

// 				theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
// 				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
// 				alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
// 				alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
// 				bi = GPU_bias_func(Thetae, ph->w);

// 				/* Initialize dK/dlam */
// 				GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
// 				}

// 				/*****************************************************end of condition******************************************/

// 				/* Save initial position/wave vector */
// 				Xi[0] = ph->X[0];
// 				Xi[1] = ph->X[1];
// 				Xi[2] = ph->X[2];
// 				Xi[3] = ph->X[3];
// 				Ki[0] = ph->K[0];
// 				Ki[1] = ph->K[1];
// 				Ki[2] = ph->K[2];
// 				Ki[3] = ph->K[3];
// 				dKi[0] = ph->dKdlam[0];
// 				dKi[1] = ph->dKdlam[1];
// 				dKi[2] = ph->dKdlam[2];
// 				dKi[3] = ph->dKdlam[3];
// 				E0 = ph->E0s;

// 				/* evaluate stepsize */
// 				dl = GPU_stepsize(ph->X, ph->K);
// 				//printf("it just evaluated stepsize\n");
// 				/* step the geodesic */
// 				GPU_push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s),0);
// 				//printf("First push photon\n");
// 				if (GPU_stop_criterion(state,ph))
// 					break;

// 				/* allow photon to interact with matter, */
// 				GPU_gcov_func_hamr(ph->X, Gcov);
// 				GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov,
// 						Bcon, Bcov, d_p);
// 				if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {

// 					bound_flag = 0;
// 					if (Ne == 0.)
// 						bound_flag = 1;
// 					if (!bound_flag) {
// 						theta =
// 							GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov,
// 								B);
// 						nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
// 						if (isnan(nu)) {
// 							printf(
// 								"isnan nu: track_super_photon dl,E0 %g %g\n",
// 								dl, E0);
// 							printf(
// 								"Xi, %g %g %g %g\n", Xi[0],
// 								Xi[1], Xi[2], Xi[3]);
// 							printf(
// 								"Ki, %g %g %g %g\n", Ki[0],
// 								Ki[1], Ki[2], Ki[3]);
// 							printf(
// 								"dKi, %g %g %g %g\n",
// 								dKi[0], dKi[1], dKi[2],
// 								dKi[3]);
// 							//exit(1);
// 							printf("The function should exit the code!");
// 							return;
// 						}
// 					}

// 					/* scattering optical depth along step */
// 					if (bound_flag || nu < 0.) {
// 						dtau_scatt =
// 							0.5 * alpha_scatti * dtauK * dl;
// 						dtau_abs = 0.5 * alpha_absi * dtauK * dl;
// 						alpha_scatti = alpha_absi = 0.;
// 						bias = 0.;
// 						bi = 0.;
// 					} else {
// 						alpha_scattf =
// 							GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
// 						dtau_scatt =
// 							0.5 * (alpha_scatti +
// 							alpha_scattf) * dtauK * dl;
// 						alpha_scatti = alpha_scattf;

// 						/* absorption optical depth along step */
// 						alpha_absf =
// 							GPU_alpha_inv_abs(nu, Thetae, Ne, B,
// 								theta);
// 						dtau_abs =
// 							0.5 * (alpha_absi +
// 							alpha_absf) * dtauK * dl;
// 						alpha_absi = alpha_absf;

// 						bf = GPU_bias_func(Thetae, ph->w);
// 						bias = 0.5 * (bi + bf);
// 						bi = bf;
// 					}

// 					x1 = -log(GPU_monty_rand(state));
// 					php.w = ph->w / bias;
// 					if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
// 						if (isnan(php.w) || isinf(php.w)) {
// 							printf(
// 								"w isnan in track_super_photon: Ne, bias, ph->w, php.w  %g, %g, %g, %g\n",
// 								Ne, bias, ph->w, php.w);
// 						}

// 						frac = x1 / (bias * dtau_scatt);

// 						/* Apply absorption until scattering event */
// 						dtau_abs *= frac;
// 						if (dtau_abs > 100)
// 							return;	/* This photon has been absorbed before scattering */

// 						dtau_scatt *= frac;
// 						dtau = dtau_abs + dtau_scatt;
// 						if (dtau_abs < 1.e-3)
// 							ph->w *=
// 								(1. -
// 								dtau / 24. * (24. -
// 									dtau * (12. -
// 										dtau *
// 										(4. -
// 											dtau))));
// 						else
// 							ph->w *= exp(-dtau);

// 						/* Interpolate position and wave vector to scattering event */
// 						GPU_push_photon(Xi, Ki, dKi, dl * frac, &E0,0);
// 						ph->X[0] = Xi[0];
// 						ph->X[1] = Xi[1];
// 						ph->X[2] = Xi[2];
// 						ph->X[3] = Xi[3];
// 						ph->K[0] = Ki[0];
// 						ph->K[1] = Ki[1];
// 						ph->K[2] = Ki[2];
// 						ph->K[3] = Ki[3];
// 						ph->dKdlam[0] = dKi[0];
// 						ph->dKdlam[1] = dKi[1];
// 						ph->dKdlam[2] = dKi[2];
// 						ph->dKdlam[3] = dKi[3];
// 						ph->E0s = E0;

// 						/* Get plasma parameters at new position */
// 						GPU_gcov_func_hamr(ph->X, Gcov);
// 						GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae,
// 								&B, Ucon, Ucov, Bcon,
// 								Bcov, d_p);
// 						if (Ne > 0.) {							
// 							GPU_scatter_super_photon(state, ph, &php, Ne,
// 										Thetae, B,
// 										Ucon, Bcon,
// 										Gcov);
// 							if (ph->w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
// 								return;
// 							}
// 							/*Modifying condition to true*/
// 							//printf("It got recursive! (%d)\n", recursive_index);
// 							//printf("Nstep = %d\n", nstep);
// 							starting_recursion = true;
// 							/*Saving local variables to a certain index*/
// 							local_track_vars[recursive_index].bound_flag = bound_flag;
// 							local_track_vars[recursive_index].dtau_scatt = dtau_scatt;
// 							local_track_vars[recursive_index].dtau_abs = dtau_abs;
// 							local_track_vars[recursive_index].dtau = dtau;
// 							local_track_vars[recursive_index].bi = bi;
// 							local_track_vars[recursive_index].bf = bf;
// 							local_track_vars[recursive_index].alpha_scatti = alpha_scatti;
// 							local_track_vars[recursive_index].alpha_scattf = alpha_scattf;
// 							local_track_vars[recursive_index].alpha_absi = alpha_absi;
// 							local_track_vars[recursive_index].alpha_absf = alpha_absf;
// 							local_track_vars[recursive_index].dl = dl;
// 							local_track_vars[recursive_index].x1 = x1;
// 							local_track_vars[recursive_index].nu = nu;
// 							local_track_vars[recursive_index].Thetae = Thetae;
// 							local_track_vars[recursive_index].Ne = Ne;
// 							local_track_vars[recursive_index].B = B;
// 							local_track_vars[recursive_index].theta = theta;
// 							local_track_vars[recursive_index].dtauK = dtauK;
// 							local_track_vars[recursive_index].frac = frac;
// 							local_track_vars[recursive_index].bias = bias;
// 							local_track_vars[recursive_index].E0 = E0;
// 							for (int i = 0; i < NDIM; i++) {
// 								local_track_vars[recursive_index].Xi[i] = Xi[i];
// 								local_track_vars[recursive_index].Ki[i] = Ki[i];
// 								local_track_vars[recursive_index].dKi[i] = dKi[i];
// 								local_track_vars[recursive_index].Ucon[i] = Ucon[i];
// 								local_track_vars[recursive_index].Ucov[i] = Ucov[i];
// 								local_track_vars[recursive_index].Bcon[i] = Bcon[i];
// 								local_track_vars[recursive_index].Bcov[i] = Bcov[i];
// 							}
// 							for (int i = 0; i < NDIM; i++) for (int j= 0; j<NDIM; j++){
// 								local_track_vars[recursive_index].Gcov[i][j] = Gcov[i][j];
// 							}

// 							local_track_vars[recursive_index].nstep = nstep;
// 							local_track_vars[recursive_index].php = php;
// 							local_track_vars[recursive_index].ph = ph;
// 							/*transfer content from php to ph*/
// 							ph = &php;
// 							/*Increasing the place in the list*/
// 							//printf("Recursive_index = %d\n", recursive_index);
// 							recursive_index++;
// 							//printf("all the value for ph and php are updated\n");
// 							//printf("recursive_index = %d\n", recursive_index);

// 							continue;
// 							//track_super_photon(&php);
// 						}

// 					}
// 				}
// 			}else{
// 				jump_to_ending = false;
// 				recursive_index--;
// 				//printf("It's leaving recursion!(%d)\n", recursive_index);
// 				//printf("Recursive index = %d\n", recursive_index);
// 				bound_flag = local_track_vars[recursive_index].bound_flag;
// 				dtau_scatt = local_track_vars[recursive_index].dtau_scatt;
// 				dtau_abs = local_track_vars[recursive_index].dtau_abs;
// 				dtau = local_track_vars[recursive_index].dtau;
// 				bi = local_track_vars[recursive_index].bi;
// 				bf = local_track_vars[recursive_index].bf;
// 				alpha_scatti = local_track_vars[recursive_index].alpha_scatti;
// 				alpha_scattf = local_track_vars[recursive_index].alpha_scattf;
// 				alpha_absi = local_track_vars[recursive_index].alpha_absi;
// 				alpha_absf = local_track_vars[recursive_index].alpha_absf;
// 				dl = local_track_vars[recursive_index].dl;
// 				x1 = local_track_vars[recursive_index].x1;
// 				nu = local_track_vars[recursive_index].nu;
// 				Thetae = local_track_vars[recursive_index].Thetae;
// 				Ne = local_track_vars[recursive_index].Ne;
// 				B = local_track_vars[recursive_index].B;
// 				theta = local_track_vars[recursive_index].theta;
// 				dtauK = local_track_vars[recursive_index].dtauK;
// 				frac = local_track_vars[recursive_index].frac;
// 				bias = local_track_vars[recursive_index].bias;
// 				E0 = local_track_vars[recursive_index].E0;

// 				for (int i = 0; i < NDIM; i++) {
// 					Xi[i] = local_track_vars[recursive_index].Xi[i];
// 					Ki[i] = local_track_vars[recursive_index].Ki[i];
// 					dKi[i] = local_track_vars[recursive_index].dKi[i];
// 					Ucon[i] = local_track_vars[recursive_index].Ucon[i];
// 					Ucov[i] = local_track_vars[recursive_index].Ucov[i];
// 					Bcon[i] = local_track_vars[recursive_index].Bcon[i];
// 					Bcov[i] = local_track_vars[recursive_index].Bcov[i];
// 				}
// 				for (int i = 0; i < NDIM; i++) for (int j= 0; j<NDIM; j++){
// 						Gcov[i][j]= local_track_vars[recursive_index].Gcov[i][j];
// 				}
// 				nstep = local_track_vars[recursive_index].nstep;
// 				ph = local_track_vars[recursive_index].ph;
// 				php = local_track_vars[recursive_index].php;

// 				//printf("Nstep = %d\n", nstep);
// 			}
// 			if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {
// 				if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
// 				theta =
// 					GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov,
// 						B);
// 				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
// 				if (nu < 0.) {
// 					alpha_scatti = alpha_absi = 0.;
// 				} else {
// 					alpha_scatti =
// 						GPU_alpha_inv_scatt(nu, Thetae,
// 								Ne, d_table_ptr);
// 					alpha_absi =
// 						GPU_alpha_inv_abs(nu, Thetae, Ne,
// 							B, theta);
// 				}
// 				bi = GPU_bias_func(Thetae, ph->w);

// 				ph->tau_abs += dtau_abs;
// 				ph->tau_scatt += dtau_scatt;

// 			} else {
// 				if (dtau_abs > 100)
// 					return;	/* This photon has been absorbed */
// 				ph->tau_abs += dtau_abs;
// 				ph->tau_scatt += dtau_scatt;
// 				dtau = dtau_abs + dtau_scatt;
// 				if (dtau < 1.e-3)
// 					ph->w *=
// 						(1. -
// 						dtau / 24. * (24. -
// 							dtau * (12. -
// 								dtau *
// 								(4. -
// 									dtau))));
// 				else
// 					ph->w *= exp(-dtau);
// 			}
// 		}

// 		nstep++;
// 		//printf("Nstep = %d\n", nstep);
// 		/* signs that something's wrong w/ the integration */
// 		if (nstep > MAXNSTEP) {
// 			printf(
// 				"X1,X2,K1,K2,bias: %g %g %g %g %g\n",
// 				ph->X[1], ph->X[2], ph->K[1], ph->K[2],
// 				bias);
// 			break;
// 		}
// 	}
// 	//printf("It left the while!\n");

// 	/* accumulate result in spectrum on escape */
// 	if (GPU_record_criterion(ph) && nstep < MAXNSTEP)
// 		GPU_record_super_photon(ph, d_spect);

// 	if(recursive_index > 0){
// 		jump_to_ending = true;
// 		goto recursive_round;
// 	}
// 	/* done! */
// 	return;
// }

__device__ void GPU_track_super_photon(curandStateMtgp32 *state, struct of_photon *ph, double * d_p, struct local_track_var * local_track_vars, int recursive_index, double * d_table_ptr, struct of_spectrum* d_spect)
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

	dtauK = 2. * M_PI * L_UNIT / (ME * CL * CL / HBAR);

	/* Initialize opacities */
	#if(HAMR)
	GPU_gcov_func_hamr(ph->X, Gcov);
	//gcov_func(ph->X, Gcov);
	#else
	gcov_func(ph->X, Gcov);
	#endif

	GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
			 Bcov, d_p);

	theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
	nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
	alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
	alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
	bi = GPU_bias_func(Thetae, ph->w);

	/* Initialize dK/dlam */
	GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
	// printf("Outside everything dKcon[0] = %lf, dKcon[1] = %lf, dKcon[2] = %lf, dKcon[3] = %lf\n", ph->dKdlam[0], ph->dKdlam[1], ph->dKdlam[2], ph->dKdlam[3]);
	// exit(1);
	while (!GPU_stop_criterion(state, ph)) {
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
		// printf("before pushing photon: X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", ph->X[0], ph->X[1], ph->X[2], ph->X[3]);
		// printf("Outside Push Photon function dl = %lf, dKcon[0] = %lf, dKcon[1] = %lf, dKcon[2] = %lf, dKcon[3] = %lf\n", dl, ph->dKdlam[0], ph->dKdlam[1], ph->dKdlam[2], ph->dKdlam[3]);

		GPU_push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s), 0);

		if (GPU_stop_criterion(state, ph))
			break;

		/* allow photon to interact with matter, */
		#if(HAMR)
		// for(int i = 0; i < NDIM; i++)
		// for(int j = 0; j < NDIM; j++){
		// printf("Gcov[%d][%d] = %lf\n", i, j, Gcov[i][j]);
		// }
		GPU_gcov_func_hamr(ph->X, Gcov);
		// printf("X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", ph->X[0], ph->X[1], ph->X[2], ph->X[3]);
		// for(int i = 0; i < NDIM; i++)
		// for(int j = 0; j < NDIM; j++){
		// printf("Gcov[%d][%d] = %lf\n", i, j, Gcov[i][j]);
		// }
		//gcov_func(ph->X, Gcov);
		#else
		gcov_func(ph->X, Gcov);
		#endif
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

				bf = GPU_bias_func(Thetae, ph->w);
				bias = 0.5 * (bi + bf);
				bi = bf;
			}

			x1 = -log(GPU_monty_rand(state));
			php.w = ph->w / bias;
			if(recursive_index == 0)
			//printf("bias = %le, dtau_scatt = %le, dl = %le, php.w = %le, x1 = %le\n", bias, dtau_scatt, dl, php.w, x1);
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
				if (dtau_abs < 1.e-3)
					ph->w *=
					    (1. -
					     dtau / 24. * (24. -
							   dtau * (12. -
								   dtau *
								   (4. -
								    dtau))));
				else
					ph->w *= exp(-dtau);

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
				#if(HAMR)
				GPU_gcov_func_hamr(ph->X, Gcov);
				//gcov_func(ph->X, Gcov);
				#else
				gcov_func(ph->X, Gcov);
				#endif
				GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae,
						 &B, Ucon, Ucov, Bcon,
						 Bcov, d_p);

				if (Ne > 0.) {
					GPU_scatter_super_photon(state, ph, &php, Ne,
							     Thetae, B,
							     Ucon, Bcon,
							     Gcov);
					if (ph->w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
						return;
					}
					recursive_index++;
					//printf("it got recursive (%d)\n", recursive_index);
					GPU_track_super_photon(state, &php, d_p, local_track_vars, recursive_index, d_table_ptr, d_spect);
				}
				recursive_index--;
				//printf("Leaving recursion (%d)\n", recursive_index);

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
				bi = GPU_bias_func(Thetae, ph->w);

				ph->tau_abs += dtau_abs;
				ph->tau_scatt += dtau_scatt;

			} else {
				if (dtau_abs > 100){
					return;	/* This photon has been absorbed */
				}
				ph->tau_abs += dtau_abs;
				ph->tau_scatt += dtau_scatt;
				dtau = dtau_abs + dtau_scatt;
				if (dtau < 1.e-3)
					ph->w *=
					    (1. -
					     dtau / 24. * (24. -
							   dtau * (12. -
								   dtau *
								   (4. -
								    dtau))));
				else
					ph->w *= exp(-dtau);
			}
		}

		nstep++;

		/* signs that something's wrong w/ the integration */
		if (nstep > MAXNSTEP) {
			printf(
				"X1,X2,K1,K2,bias: %g %g %g %g %g\n",
				ph->X[1], ph->X[2], ph->K[1], ph->K[2],
				bias);
			break;
		}

	}

	/* accumulate result in spectrum on escape */
	if (GPU_record_criterion(ph) && nstep < MAXNSTEP)
		GPU_record_super_photon(ph, d_spect);

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
	__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double del[8]);
	//printf( "d_startx[1] = %le, d_stopx[1] = %le, d_startx[2] = %le, d_stopx[2] = %le\n", d_startx[1], d_stopx[1], d_startx[2], d_stopx[2]);

	//checks if it's within the grid
	if (X[1] < d_startx[1] ||
	    X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2] || X[3] < d_startx[3] || X[3] > d_stopx[3]) {

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
	// if (fabs(rho - p[KRHO][i][j]) > 1e-2){
	// 	printf( "X[1] = %le, X[2] = %le, i = %d, j = %d\n", X[1], X[2], i, j);
	// 	printf( "rho = %le, interp_rho = %le\n", p[KRHO][i][j], rho);
	// }
	*Ne = rho * NE_UNIT;
	if (*Ne == 0){
		printf("Ne = 0!!\n");
	}
	*Thetae = uu / rho * d_thetae_unit;

	Bp[1] = GPU_interp_scalar(d_p, B1, i, j, k, coeff);
	Bp[2] = GPU_interp_scalar(d_p, B2, i, j, k, coeff);
	Bp[3] = GPU_interp_scalar(d_p, B3, i, j, k, coeff);

	Vcon[1] = GPU_interp_scalar(d_p, U1, i, j, k, coeff);
	Vcon[2] = GPU_interp_scalar(d_p, U2, i, j, k, coeff);
	Vcon[3] = GPU_interp_scalar(d_p, U3, i, j, k, coeff);

	#if(HAMR)
	GPU_gcon_func_hamr(gcov, gcon);
	//gcon_func(X, gcon);
	#else
	gcon_func(X, gcon);
	#endif
	
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
	GPU_lower(Ucon, gcov, Ucov);

	/* Get B and Bcov */
	UdotBp = 0.;
	for (i = 1; i < NDIM; i++)
		UdotBp += Ucov[i] * Bp[i];
	Bcon[0] = UdotBp;
	for (i = 1; i < NDIM; i++)
		Bcon[i] = (Bp[i] + Ucon[i] * UdotBp) / Ucon[0];
	GPU_lower(Bcon, gcov, Bcov);

	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;

}
__device__ double GPU_get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
		    double Bcov[NDIM], double B)
{

	double k, mu;

	if (B == 0.)
		return (M_PI / 2.);

	k = fabs(K[0] * Ucov[0] + K[1] * Ucov[1] + K[2] * Ucov[2] +
		 K[3] * Ucov[3]);

	/* B is in cgs but Bcov is in code units */
	mu = (K[0] * Bcov[0] + K[1] * Bcov[1] + K[2] * Bcov[2] +
	      K[3] * Bcov[3]) / (k * B / B_UNIT);

	if (fabs(mu) > 1.)
		mu /= fabs(mu);

	return (acos(mu));
}
__device__ void GPU_gcov_func_hamr(double *X, double gcovp[][NDIM])
{
	int i, j, k, l;
	double sth, cth, s2, rho2;
	double del[NDIM];
	double r, th, phi;
	double a = 0.9375;
	double tilt = TILT_ANGLE / 180.*M_PI;
	//printf( "X = %lf, %lf, %lf, %lf\n", X[0], X[1], X[2], X[3]);

	for(j=0;j<NDIM;j++) for(k=0;k<NDIM;k++) gcovp[j][k] = 0.;
	GPU_bl_coord_hamr(X, &r, &th, &phi);
	cth = cos(th);
	sth = sin(th);

	s2 = sth*sth;
	rho2 = r*r + d_a*d_a*cth*cth;

	//compute Jacobian x1,x2,x3 -> r,th,phi (dr/dx1)
	gcovp[0][0] = (-1. + 2.*r / rho2);
	gcovp[0][1] = (2.*r / rho2)*r;
	gcovp[0][3] = (-2.*d_a*r*s2 / rho2);

	gcovp[1][0] = gcovp[0][1];
	gcovp[1][1] = (1. + 2.*r / rho2)*r*r;
	gcovp[1][3] = (-d_a*s2*(1. + 2.*r / rho2)) * r;

	gcovp[2][2] = rho2 * (M_PI/2) * (M_PI/2);

	gcovp[3][0] = gcovp[0][3];
	gcovp[3][1] = gcovp[1][3];
	gcovp[3][3] = s2*(rho2 + d_a*d_a*s2*(1. + 2.*r / rho2));
	//for(j=0;j<NDIM;j++) for(k=0;k<NDIM;k++) printf( "gcovp[%d][%d], %lf\n",j,k,gcovp[j][k]);


}
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM])
{

	*i = (int) ((X[1] - d_startx[1]) / d_dx[1] - 0.5 + 1000) - 1000;
	*j = (int) ((X[2] - d_startx[2]) / d_dx[2] - 0.5 + 1000) - 1000;
	if (*i < 0) {
		*i = 0;
		del[1] = 0.;
	} else if (*i > d_N1 - 2) {
		*i = d_N1 - 2;
		del[1] = 1.;
	} else {
		del[1] = (X[1] - ((*i + 0.5) * d_dx[1] + d_startx[1])) / d_dx[1];
	}

	if (*j < 0) {
		*j = 0;
		del[2] = 0.;
	} else if (*j > d_N2 - 2) {
		*j = d_N2 - 2;
		del[2] = 1.;
	} else {
		del[2] = (X[2] - ((*j + 0.5) * d_dx[2] + d_startx[2])) / d_dx[2]; //fractional displacement of the center of the grid cel
	}
	*k = 0;
	del[3] = 0;
	#if(HAMR3D)
	*k= (int) ((X[3] - d_startx[3]) / d_dx[3] - 0.5 + 1000) - 1000;
	if (*k < 0) {
		*k = 0;
		del[3] = 0.;
	} else if (*k > d_N3 - 2) {
		*k = d_N3 - 2;
		del[3] = 1.;
	} else {
		del[3] = (X[3] - ((*k + 0.5) * d_dx[3] + d_startx[3])) / d_dx[3]; //fractional displacement of the center of the grid cel
	}
	#endif
	return;
}
__device__ void GPU_gcon_func_hamr(double gcov[][NDIM], double gcon[][NDIM])
{
  GPU_invert_matrix( gcov, gcon );
}
__device__ void GPU_vofx_matthewcoords(double *X, double *V){
	V[0] = X[0];
	double RTRANS =5000000.;
	double RB = 0.;
	double RADEXP = 1.0;
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	double BRAVO = 0.0;
	double TANGO = 1.0;
	double CHARLIE = 0.0;
	double DELTA = 3.0;
	if (X[1] < Xtrans){
		V[1] = exp(pow(X[1], RADEXP)) + RB;
	}
	else if (X[1] >= Xtrans && X[1]<1.01*Xtrans){
		V[1] = 10.*(X[1] / Xtrans - 1.)*((X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS) +
			(1. - 10.*(X[1] / Xtrans - 1.))*(exp(pow(X[1], RADEXP)) + RB);
	}
	else{
		V[1] = (X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS;
	}
	double A1 = 1. / (1. + pow(CHARLIE*(log(V[1]) / log(10.)), DELTA));
	double A2 = BRAVO*(log(V[1]) / log(10.)) + TANGO;
	double A3 = pow(0.5, 1. - A2);
	double sign = 1.;
	double X_2 =(X[2]+1.0)/2.0;
	double Xc = sqrt(pow(X_2, 2.));

	if (X_2 < 0.0){
		sign = -1.;
	}
	if (X_2 > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X_2 >= 0.5){
		Xc = 1. - Xc;
		V[2] = M_PI - sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	else{
		V[2] = sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	V[3] = X[3];
}
__device__ void GPU_bl_coord_hamr(double * X, double * r, double *th, double *phi)
{
	double V[4];
	double SINGSMALL = 1.e-20;
  	void (*vofx_function_pointer)(double*, double*);
    vofx_function_pointer = GPU_vofx_matthewcoords;
	vofx_function_pointer(X,V);
	// avoid singularity at polar axis
	if (fabs(V[2])<SINGSMALL){
		if (V[2] >= 0.0) V[2] = SINGSMALL;
		if (V[2]<0.0)  V[2] = -SINGSMALL;
	}
	if (fabs(M_PI - V[2]) <SINGSMALL){
		if (V[2] >= M_PI) V[2] = M_PI + SINGSMALL;
		if (V[2]<M_PI)  V[2] = M_PI -  SINGSMALL;
	}
	*r = V[1];
	*th = V[2];
	*phi = V[3];
	return ;
}
__device__ int GPU_LU_decompose( double A[][NDIM], int permute[] )
{
  double row_norm[NDIM];

  double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */

  double  absmax, maxtemp, mintemp;

  int i, j, k, max_row;
  int n = NDIM;


  max_row = 0;

  /* Find the maximum elements per row so that we can pretend later
     we have unit-normalized each equation: */

  for( i = 0; i < n; i++ ) { 
    absmax = 0.;
    
    for( j = 0; j < n ; j++ ) { 
      
      maxtemp = fabs( A[i][j] ); 

      if( maxtemp > absmax ) { 
	absmax = maxtemp; 
      }
    }

    /* Make sure that there is at least one non-zero element in this row: */
    if( absmax == 0. ) { 
     //printf( "LU_decompose(): row-wise singular matrix!\n");
      return(1);
    }

    row_norm[i] = 1. / absmax ;   /* Set the row's normalization factor. */
  }


  /* The following the calculates the matrix composed of the sum 
     of the lower (L) tridagonal matrix and the upper (U) tridagonal
     matrix that, when multiplied, form the original maxtrix.  
     This is what we call the LU decomposition of the maxtrix. 
     It does this by a recursive procedure, starting from the 
     upper-left, proceding down the column, and then to the next
     column to the right.  The decomposition can be done in place 
     since element {i,j} require only those elements with {<=i,<=j} 
     which have already been computed.
     See pg. 43-46 of "Num. Rec." for a more thorough description. 
  */

  /* For each of the columns, starting from the left ... */
  for( j = 0; j < n; j++ ) {

    /* For each of the rows starting from the top.... */

    /* Calculate the Upper part of the matrix:  i < j :   */
    for( i = 0; i < j; i++ ) {
      for( k = 0; k < i; k++ ) { 
	A[i][j] -= A[i][k] * A[k][j];
      }
    }

    absmax = 0.0;

    /* Calculate the Lower part of the matrix:  i <= j :   */

    for( i = j; i < n; i++ ) {

      for (k = 0; k < j; k++) { 
	A[i][j] -= A[i][k] * A[k][j];
      }

      /* Find the maximum element in the column given the implicit 
	 unit-normalization (represented by row_norm[i]) of each row: 
      */
      maxtemp = fabs(A[i][j]) * row_norm[i] ;

      if( maxtemp >= absmax ) {
	absmax = maxtemp;
	max_row = i;
      }

    }

    /* Swap the row with the largest element (of column j) with row_j.  absmax
       This is the partial pivoting procedure that ensures we don't divide
       by 0 (or a small number) when we solve the linear system.  
       Also, since the procedure starts from left-right/top-bottom, 
       the pivot values are chosen from a pool involving all the elements 
       of column_j  in rows beneath row_j.  This ensures that 
       a row  is not permuted twice, which would mess things up. 
    */
    if( max_row != j ) {

      /* Don't swap if it will send a 0 to the last diagonal position. 
	 Note that the last column cannot pivot with any other row, 
	 so this is the last chance to ensure that the last two 
	 columns have non-zero diagonal elements.
       */

      if( (j == (n-2)) && (A[j][j+1] == 0.) ) {
	max_row = j;
      }
      else { 
	for( k = 0; k < n; k++ ) { 

	  maxtemp       = A[   j   ][k] ; 
	  A[   j   ][k] = A[max_row][k] ;
	  A[max_row][k] = maxtemp; 

	}

	/* Don't forget to swap the normalization factors, too... 
	   but we don't need the jth element any longer since we 
	   only look at rows beneath j from here on out. 
	*/
	row_norm[max_row] = row_norm[j] ; 
      }
    }

    /* Set the permutation record s.t. the j^th element equals the 
       index of the row swapped with the j^th row.  Note that since 
       this is being done in successive columns, the permutation
       vector records the successive permutations and therefore
       index of permute[] also indexes the chronology of the 
       permutations.  E.g. permute[2] = {2,1} is an identity 
       permutation, which cannot happen here though. 
    */

    permute[j] = max_row;

    if( A[j][j] == 0. ) { 
      A[j][j] = absmin;
    }


  /* Normalize the columns of the Lower tridiagonal part by their respective 
     diagonal element.  This is not done in the Upper part because the 
     Lower part's diagonal elements were set to 1, which can be done w/o 
     any loss of generality.
  */
    if( j != (n-1) ) { 
      maxtemp = 1. / A[j][j]  ;
      
      for( i = (j+1) ; i < n; i++ ) {
	A[i][j] *= maxtemp;
      }
    }

  }

  return(0);

  /* End of LU_decompose() */

}
__device__ void GPU_LU_substitution( double A[][NDIM], double B[], int permute[] )
{
  int i, j ;
  int n = NDIM;
  double tmpvar,tmpvar2;

  
  /* Perform the forward substitution using the LU matrix. 
   */
  for(i = 0; i < n; i++) {

    /* Before doing the substitution, we must first permute the 
       B vector to match the permutation of the LU matrix. 
       Since only the rows above the currrent one matter for 
       this row, we can permute one at a time. 
    */
    tmpvar        = B[permute[i]];
    B[permute[i]] = B[    i     ];
    for( j = (i-1); j >= 0 ; j-- ) { 
      tmpvar -=  A[i][j] * B[j];
    }
    B[i] = tmpvar; 
  }
	   

  /* Perform the backward substitution using the LU matrix. 
   */
  for( i = (n-1); i >= 0; i-- ) { 
    for( j = (i+1); j < n ; j++ ) { 
      B[i] -=  A[i][j] * B[j];
    }
    B[i] /= A[i][i] ; 
  }

  /* End of LU_substitution() */

}
__device__ int GPU_invert_matrix( double Am[][NDIM], double Aminv[][NDIM] )  
{ 

  int i,j;
  int n = NDIM;
  int permute[NDIM]; 
  double dxm[NDIM], Amtmp[NDIM][NDIM];

  for( i = 0 ; i < NDIM*NDIM ; i++ ) {  Amtmp[0][i] = Am[0][i]; }

  // Get the LU matrix:
  if( GPU_LU_decompose( Amtmp,  permute ) != 0  ) { 
    printf("invert_matrix(): singular matrix encountered! \n");
    printf("This is probably due to a nan value somewhere rather than determinant = 0. Investigate!\n");
	return(1);
  }

  for( i = 0; i < n; i++ ) { 
    for( j = 0 ; j < n ; j++ ) { dxm[j] = 0. ; }
    dxm[i] = 1.; 
    
    /* Solve the linear system for the i^th column of the inverse matrix: :  */
    GPU_LU_substitution( Amtmp,  dxm, permute );

    for( j = 0 ; j < n ; j++ ) {  Aminv[j][i] = dxm[j]; }

  }

  return(0);
}
/* get frequency in fluid frame, in Hz */
__device__ double GPU_get_fluid_nu(double X[4], double K[4], double Ucov[NDIM])
{
	double ener, nu;

	/* this is the energy in electron rest-mass units */
	ener = -(K[0] * Ucov[0] +
		 K[1] * Ucov[1] + K[2] * Ucov[2] + K[3] * Ucov[3]);

	nu = ener * ME * CL * CL / HPL;

	if (isnan(ener)) {
		printf("isnan get_fluid_nu, K: %g %g %g %g\n",
			K[0], K[1], K[2], K[3]);
		printf("isnan get_fluid_nu, X: %g %g %g %g\n",
			X[0], X[1], X[2], X[3]);
		printf("isnan get_fluid_nu, U: %g %g %g %g\n",
			Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
	}

	return nu;

}
/* return Lorentz invariant scattering opacity */
__device__ double GPU_alpha_inv_scatt(double nu, double Thetae, double Ne, double * d_table_ptr)
{
	double kappa;

	kappa = GPU_kappa_es(nu, Thetae, d_table_ptr);
	//printf("nu = %le, Thetae = %le, kappa = %le, Ne = %le\n", nu, Thetae, kappa, Ne);
	return (nu * kappa * Ne * MP);
}
/* return Lorentz invariant absorption opacity */
__device__ double GPU_alpha_inv_abs(double nu, double Thetae, double Ne, double B,
		     double theta)
{
	double j, bnu;

	j = GPU_jnu_inv(nu, Thetae, Ne, B, theta);
	bnu = GPU_Bnu_inv(nu, Thetae);

	return (j / (bnu + 1.e-100));
}
__device__ double GPU_bias_func(double Te, double w)
{
	double bias, max, avg_num_scatt;

	max = 0.5 * w / WEIGHT_MIN;

	avg_num_scatt = d_N_scatt / (1. * d_N_superph_recorded + 1.);
	bias =
	    100. * Te * Te / (d_bias_norm * d_max_tau_scatt *
			      (avg_num_scatt + 2));

	if (bias < TP_OVER_TE)
		bias = TP_OVER_TE;
	if (bias > max)
		bias = max;

	return bias / TP_OVER_TE;
}
__device__ void GPU_init_dKdlam(double X[], double Kcon[], double dK[])
{
	int k;
	double lconn[NDIM][NDIM][NDIM];

	GPU_get_connection(X, lconn);
	//printf( "Inside INIT_DKDLAM after X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", X[0], X[1], X[2], X[3]);

	// for(int i = 0; i< 4; i++)
	// for(int j = 0; j < 4; j++)
	// for(int z = 0; z< 4; z++){
	// 	printf( "We are inside init_dKdlam function: lconn[%d][%d][%d] = %le\n", i, j, z, lconn[i][j][z]);
	// }
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
__device__ int GPU_stop_criterion(curandStateMtgp32 *state, struct of_photon *ph)
{
	double wmin, X1min, X1max;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */
	
	X1min = log(d_Rh);	/* this is coordinate-specific; stop
				   at event horizon */
	X1max = log(RMAX);	/* this is coordinate and simulation
				   specific: stop at large distance */

	//printf( "w is: %le, wmin is: %le\n", ph->w, wmin);
	//printf( "X[1] = %le, X1min = %le\n", ph ->X[1], X1min);
	if (ph->X[1] < X1min)
		//printf( "it's getting here 1\n");
		return 1;

	if (ph->X[1] > X1max) {
		if (ph->w < wmin) {
			if (GPU_monty_rand(state) <= 1. / ROULETTE) {
				//printf( "it's getting here 2\n");
				ph->w *= ROULETTE;
			} else
				//printf( "it's getting here 3\n");
				ph->w = 0.;
		}
		return 1;
	}

	if (ph->w < wmin) {
		if (GPU_monty_rand(state) <= 1. / ROULETTE) {
			ph->w *= ROULETTE;
			//printf( "it's getting here 4\n");
		} else {
			ph->w = 0.;
			//printf("it's getting here 5\n");
			return 1;
		}
	}

	return (0);
}
__device__ double GPU_stepsize(double X[NDIM], double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;
	#if(HAMR)
		double x2_normal, stopx2_normal;
		x2_normal = (1 + X[2])/2;
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
	// printf("dlx1 = %le, dlx2 = %le, dlx3 = %le, idlx1 = %le, idlx2 = %le, idlx3 = %le, dl = %le\n", dlx1, dlx2, dlx3, idlx1, idlx2, idlx3, dl);
	// for (int i = 0; i < NDIM; i++){
	// 	printf("X[%d] = %le, K[%d] = %le\n", i, X[i], i, K[i]);
	// 	printf("dl = %le\n", dl);
	// }
	return (dl);
}

// //This one below is from gpu_monty
// __device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],  double dl,
// 	double *E0, int n)
// {

//         double lconn[NDIM][NDIM][NDIM];
//         double Kcont[NDIM], K[NDIM], dK;
//         double Gcov[NDIM][NDIM];
//         double dl_2, err;
//         int i, k, iter;

//         if (X[1] < d_startx[1]) return;

//         dl_2 = 0.5 * dl;
//         /* Step the position and estimate new wave vector */
//         for (i = 0; i < NDIM; i++) {
//                 dK = dKcon[i] * dl_2;
//                 Kcon[i] += dK;
//                 K[i] = Kcon[i] + dK;
//                 X[i] += Kcon[i] * dl;
//         }

//         GPU_get_connection(X, lconn);

//         /* We're in a coordinate basis so take advantage of symmetry in the connection */
//         iter = 0;
//         do {
//                 iter++;
//                 FAST_CPY(K, Kcont);

//                 err = 0.;
//                 for (k = 0; k < 4; k++) {
//                         dKcon[k] =
//                             -2. * (Kcont[0] *
//                                    (lconn[k][0][1] * Kcont[1] +
//                                     lconn[k][0][2] * Kcont[2] +
//                                     lconn[k][0][3] * Kcont[3])
//                                    +
//                                    Kcont[1] * (lconn[k][1][2] * Kcont[2] +
//                                                lconn[k][1][3] * Kcont[3])
//                                    + lconn[k][2][3] * Kcont[2] * Kcont[3]
//                             );

//                         dKcon[k] -=
//                             (lconn[k][0][0] * Kcont[0] * Kcont[0] +
//                              lconn[k][1][1] * Kcont[1] * Kcont[1] +
//                              lconn[k][2][2] * Kcont[2] * Kcont[2] +
//                              lconn[k][3][3] * Kcont[3] * Kcont[3]
//                             );

//                         K[k] = Kcon[k] + dl_2 * dKcon[k];
//                         err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
//                 }
//         } while ((err > ETOL || isinf(err) || isnan(err)) && iter < MAX_ITER);

//         FAST_CPY(K, Kcon);

//         GPU_gcov_func_hamr(X, Gcov);
//         *E0 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
//                Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);

//         /* done! */
// }

// __device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
// 		 double dl, double *E0, int n)
// {
// 	const int max_n = 7;
// 	/*if it has already done first recursion for that specific n(recursion), it value will be true for first recursion*/
// 	bool second_recursion[max_n];
// 	//double * E0_list[max_n];
// 	int dl_original = dl;
// 	start_pushphoton_recursion:
// 		double lconn[NDIM][NDIM][NDIM];
// 		double Kcont[NDIM], K[NDIM], dK;
// 		double Xcpy[NDIM], Kcpy[NDIM], dKcpy[NDIM];
// 		double Gcov[NDIM][NDIM], E1;
// 		double dl_2, err, errE;
// 		int i, k, iter;
		

// 		if (X[1] < d_startx[1])
// 			return;

// 		FAST_CPY(X, Xcpy);
// 		FAST_CPY(Kcon, Kcpy);
// 		FAST_CPY(dKcon, dKcpy);
// 		dl_2 = 0.5 * dl;
// 		/* Step the position and estimate new wave vector */
// 		for (i = 0; i < NDIM; i++) {
// 			dK = dKcon[i] * dl_2;
// 			Kcon[i] += dK;
// 			K[i] = Kcon[i] + dK;
// 			X[i] += Kcon[i] * dl;
// 		}

// 		GPU_get_connection(X, lconn);

// 		/* We're in a coordinate basis so take advantage of symmetry in the connection */
// 		iter = 0;
// 		do {
// 			iter++;
// 			FAST_CPY(K, Kcont);

// 			err = 0.;
// 			for (k = 0; k < 4; k++) {
// 				dKcon[k] =
// 					-2. * (Kcont[0] *
// 					(lconn[k][0][1] * Kcont[1] +
// 						lconn[k][0][2] * Kcont[2] +
// 						lconn[k][0][3] * Kcont[3])
// 					+
// 					Kcont[1] * (lconn[k][1][2] * Kcont[2] +
// 							lconn[k][1][3] * Kcont[3])
// 					+ lconn[k][2][3] * Kcont[2] * Kcont[3]
// 					);

// 				dKcon[k] -=
// 					(lconn[k][0][0] * Kcont[0] * Kcont[0] +
// 					lconn[k][1][1] * Kcont[1] * Kcont[1] +
// 					lconn[k][2][2] * Kcont[2] * Kcont[2] +
// 					lconn[k][3][3] * Kcont[3] * Kcont[3]
// 					);

// 				K[k] = Kcon[k] + dl_2 * dKcon[k];
// 				err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
// 			}
// 		} while (err > ETOL && iter < MAX_ITER);

// 		FAST_CPY(K, Kcon);

// 		GPU_gcov_func_hamr(X, Gcov);

// 		E1 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
// 			Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);
// 		errE = fabs((E1 - (*E0)) / (*E0));

// 		/*start recursions for n state*/
// 		if (n < 7
// 			&& (errE > 1.e-4 || err > ETOL || isnan(err) || isinf(err))) {
// 			FAST_CPY(Xcpy, X);
// 			FAST_CPY(Kcpy, Kcon);
// 			FAST_CPY(dKcpy, dKcon);
// 			//E0_list[n] = E0;
// 			dl *= 0.5;
// 			second_recursion[n] = false;
// 			n++;
// 			goto start_pushphoton_recursion;
// 		}else{
// 			*E0 = E1;
// 			n--;
// 		}

// 		/*start second recursion*/
// 		if(n >= 0 && !second_recursion[n]){
// 			second_recursion[n] = true;
// 			n++;
// 			dl = dl_original/pow(2., n+1);
// 			goto start_pushphoton_recursion;
// 		}
// 		if(n >= 0 && second_recursion[n]){
// 			E1 = *E0;
// 			n--;
// 		}

// 		/*go to the caller of recursion n -> n-1*/
// 		if(n >= 0){
// 			dl= dl_original/pow(2., n+1);
// 			*E0 = E1;
// 			if (!second_recursion[n]){
// 				goto start_pushphoton_recursion;
// 			}
// 		}
		
// 		*E0 = E1;
// 		// if(n == 0){
// 		// for (i = 0; i < NDIM; i++)
// 		// printf("New X[%d] = %le\n", i, X[i]);
// 		// }
// 	/* done! */
//}

__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM],
		 double dl, double *E0, int n)
{
	double lconn[NDIM][NDIM][NDIM];
	double Kcont[NDIM], K[NDIM], dK;
	double Xcpy[NDIM], Kcpy[NDIM], dKcpy[NDIM];
	double Gcov[NDIM][NDIM], E1;
	double dl_2, err, errE;
	int i, k, iter;

	if (X[1] < d_startx[1])
		return;

	FAST_CPY(X, Xcpy);
	FAST_CPY(Kcon, Kcpy);
	FAST_CPY(dKcon, dKcpy);
	dl_2 = 0.5 * dl;
	/* Step the position and estimate new wave vector */
	// printf("iteration = %d\n", n);
	// printf("Inside Push Photon before X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", X[0], X[1], X[2], X[3]);
	// printf("Inside Push Photon before k[0] = %lf, k[1] = %lf, k[2] = %lf, k[3] = %lf\n", K[0], K[1], K[2], K[3]);
	// printf("dl = %lf, dKcon[0] = %lf, dKcon[1] = %lf, dKcon[2] = %lf, dKcon[3] = %lf\n", dl, dKcon[0], dKcon[1], dKcon[2], dKcon[3]);
	for (i = 0; i < NDIM; i++) {
		dK = dKcon[i] * dl_2;
		Kcon[i] += dK;
		K[i] = Kcon[i] + dK;
		X[i] += Kcon[i] * dl;
	}
	// printf("Inside Push Photon after X[0] = %lf, X[1] = %lf, X[2] = %lf, X[3] = %lf\n", X[0], X[1], X[2], X[3]);
	// printf("Inside Push Photon after k[0] = %lf, k[1] = %lf, k[2] = %lf, k[3] = %lf\n", K[0], K[1], K[2], K[3]);

	// if(omp_get_thread_num() == 0){
	// 	printf("X1 after = %le, K1 after = %le\n", X[1], K[1]);
	// }
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
	} while (err > ETOL && iter < MAX_ITER);

	FAST_CPY(K, Kcon);

	GPU_gcov_func_hamr(X, Gcov);

	E1 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
	       Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);
	errE = fabs((E1 - (*E0)) / (*E0));

	if (n < 7
	    && (errE > 1.e-4 || err > ETOL || isnan(err) || isinf(err))) {
		FAST_CPY(Xcpy, X);
		FAST_CPY(Kcpy, Kcon);
		FAST_CPY(dKcpy, dKcon);
		GPU_push_photon(X, Kcon, dKcon, 0.5 * dl, E0, n + 1);
		GPU_push_photon(X, Kcon, dKcon, 0.5 * dl, E0, n + 1);
		E1 = *E0;
	}

	*E0 = E1;
	// if(n == 0){
	// for (i = 0; i < NDIM; i++)
	// printf("New X[%d] = %le\n", i, X[i]);
	// }
	/* done! */
}

__device__ void GPU_scatter_super_photon(curandStateMtgp32 *state, struct of_photon *ph, struct of_photon *php,
			  double Ne, double Thetae, double B,
			  double Ucon[NDIM], double Bcon[NDIM],
			  double Gcov[NDIM][NDIM])
{
	double P[NDIM], Econ[NDIM][NDIM], Ecov[NDIM][NDIM],
	    K_tetrad[NDIM], K_tetrad_p[NDIM], Bhatcon[NDIM], tmpK[NDIM];
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
/*		printf("%g %g %g\n",ph->K[1], ph->K[2], ph->K[3]);
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
		printf("X1,X2: %g %g\n",ph->X[1],ph->X[2]) ;*/
		ph->w = 0.;
		return;
	}

	/* find the electron that we collided with */
	GPU_sample_electron_distr_p(state, K_tetrad, P, Thetae);

	/* given electron momentum P, find the new
	   photon momentum Kp */
	GPU_sample_scattered_photon(state, K_tetrad, P, K_tetrad_p);


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
		printf("K0, K0p, Kp, P[0]: %g %g %g %g\n",
			K_tetrad[0], K_tetrad_p[0], php->K[0], P[0]);
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
__device__ void GPU_coordinate_to_tetrad(double Ecov[NDIM][NDIM], double K[NDIM],
			  double K_tetrad[NDIM])
{
	int k;

	for (k = 0; k < 4; k++) {
		K_tetrad[k] =
		    Ecov[k][0] * K[0] +
		    Ecov[k][1] * K[1] +
		    Ecov[k][2] * K[2] + Ecov[k][3] * K[3];
	}
}
__device__ void GPU_sample_electron_distr_p(curandStateMtgp32 *state, double k[4], double p[4], double Thetae)
{
	double beta_e, mu, phi, cphi, sphi, gamma_e, sigma_KN;
	double K, sth, cth, x1, n0dotv0, v0, v1;
	double n0x, n0y, n0z;
	double v0x, v0y, v0z;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	int sample_cnt = 0;
	do {
		GPU_sample_beta_distr(state, Thetae, &gamma_e, &beta_e);
		mu = GPU_sample_mu_distr(state, beta_e);
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

		x1 = GPU_monty_rand(state);

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
	generate_random_direction(state, &n0x, &n0y, &n0z);
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
	phi = GPU_monty_rand(state) * 2. * M_PI;	/* orient uniformly */
	sphi = sinf(phi);
	cphi = cosf(phi);
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
__device__ void GPU_sample_beta_distr(curandStateMtgp32 *state, double Thetae, double *gamma_e, double *beta_e)
{
	double y;

	/* checked */
	y = GPU_sample_y_distr(state, Thetae);

	/* checked */
	*gamma_e = y * y * Thetae + 1.;
	*beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));

	return;

}
__device__ double GPU_sample_y_distr(curandStateMtgp32 *state, double Thetae)
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
		x1 = GPU_monty_rand(state);
		if (x1 < pi_3) {
			x = generate_chi_square(state, 3);
		} else if (x1 < pi_3 + pi_4) {
			x = generate_chi_square(state, 4);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			x = generate_chi_square(state, 5);
		} else {
			x = generate_chi_square(state, 6);
		}

		/* this translates between defn of distr in
		   Canfield et al. and standard chisq distr */
		y = sqrt(x / 2);

		x2 = GPU_monty_rand(state);
		num = sqrt(1. + 0.5 * Thetae * y * y);
		den = (1. + y * sqrt(0.5 * Thetae));

		prob = num / den;

	} while (x2 >= prob);
	return (y);
}
__device__ double generate_chi_square(curandStateMtgp32 *state, int df) {
        return chi_square(state, df);
}
__device__ double chi_square(curandStateMtgp32 *state, int df) {
    double sum = 0.0f;
    for (int i = 0; i < df; ++i) {
        double normal_variate = curand_normal(state);
        sum += normal_variate * normal_variate;
    }
    return sum;
}
__device__ double GPU_sample_mu_distr(curandStateMtgp32 *state, double beta_e)
{
	double mu, x1, det;

	x1 = GPU_monty_rand(state);
	det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * x1;
	if (det < 0.)
		printf("det < 0  %g %g\n\n", beta_e, x1);
	mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}
__device__ void GPU_sample_scattered_photon(curandStateMtgp32 *state, double k[4], double p[4], double kp[4])
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
		k0p = GPU_sample_klein_nishina(state, ke[0]);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = GPU_sample_thomson(state);
	}
	sth = sqrt(fabs(1. - cth * cth));

	/* unit vector 1 for scattering coordinate system is
	   oriented along initial photon wavevector */
	v0x = ke[1] / ke[0];
	v0y = ke[2] / ke[0];
	v0z = ke[3] / ke[0];

	/* randomly pick zero-angle for scattering coordinate system.
	   There's undoubtedly a better way to do this. */
	//gsl_ran_dir_3d(r, &n0x, &n0y, &n0z);
	generate_random_direction(state, &n0x, &n0y, &n0z);
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
	phi = 2. * M_PI * GPU_monty_rand(state);	
	sphi = sinf(phi);
	cphi = cosf(phi);

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
		//if (flag == 0) printf("Sampled as thomson\n");
		//else printf("sampled as klein nishina##########################################################################################################\n");
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
__device__ double GPU_sample_thomson(curandStateMtgp32 *state)
{
	double x1, x2;

	do {

		x1 = 2. * GPU_monty_rand(state) - 1.;
		x2 = (3. / 4.) * GPU_monty_rand(state);

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}
__device__ double GPU_sample_klein_nishina(curandStateMtgp32 *state,double k0)
{
	double k0pmin, k0pmax, k0p_tent, x1;
	int n = 0;

	/* a low efficiency sampling algorithm, particularly for large k0;
	   limiting efficiency is log(2 k0)/(2 k0) */
	k0pmin = k0 / (1. + 2. * k0);	/* at theta = Pi */
	k0pmax = k0;		/* at theta = 0 */
	do {

		/* tentative value */
		k0p_tent = k0pmin + (k0pmax - k0pmin) * GPU_monty_rand(state);

		/* rejection sample in box of height = kn(kmin) */
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= GPU_monty_rand(state);

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
__device__ void generate_random_direction(curandStateMtgp32 *state, double * x, double *y, double *z) {
    double u = curand_normal(state);
    double v = curand_normal(state);
    double w = curand_normal(state);
    double length = sqrt(u*u + v*v + w*w);
    *x = u / length;
    *y = v / length;
    *z = w / length;
}
__device__ double GPU_interp_scalar(double *var, int mmenemonics, int i, int j, int k, double coeff[8]){
	double interp;

	interp = coeff[0] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i, j, k)] + coeff[5] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i+1, j, k)] +
	coeff[4] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i, j + 1, k)] + coeff[7]  * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i+1, j+1, k)] +
	coeff[1] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i, j, k+1)] + coeff[6] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i+1, j, k+1)] +
	coeff[2] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i, j+1, k+1)] + coeff[3] * var[DEVICE_NPRIM_INDEX3D(mmenemonics, i+1, j+1, k+1)];

	return interp;
}
__device__ void GPU_get_connection(double X[4], double lconn[4][4][4])
{
	double r1, r2, r3, r4, sx, cx;
	double th, dthdx2, dthdx22, d2thdx22, sth, cth, sth2, cth2, sth4,
	    cth4, s2th, c2th;
	double a2, a3, a4, rho2, irho2, rho22, irho22, rho23, irho23,
	    irho23_dthdx2;
	double fac1, fac1_rho23, fac2, fac3, a2cth2, a2sth2, r1sth2,
	    a4cth4;
	/* required by broken math.h */
	//void sincos(double th, double *sth, double *cth);

	r1 = exp(X[1]);
	r2 = r1 * r1;
	r3 = r2 * r1;
	r4 = r3 * r1;

	//sincos(2. * M_PI * X[2], &sx, &cx);
	sx = sinf(2 * M_PI * X[2]);
	cx = cosf(2 * M_PI * X[2]);
	/* HARM-2D MKS */
	#if(HAMR)
	double x2_mod;
	x2_mod = (X[2] + 1.)/2.;
	th = M_PI * x2_mod;
    dthdx2 = M_PI * (1./2.);
    d2thdx22 = 0;
	#else
	th = M_PI * X[2] + 0.5 * (1 - hslope) * sx;
	dthdx2 = M_PI * (1. + (1 - hslope) * cx);
	d2thdx22 = -2. * M_PI * M_PI * (1 - hslope) * sx;
	#endif
	dthdx22 = dthdx2 * dthdx2;

	//sincos(th, &sth, &cth);
	sth = sinf(th);
	cth = cosf(th);
	sth2 = sth * sth;
	r1sth2 = r1 * sth2;
	sth4 = sth2 * sth2;
	cth2 = cth * cth;
	cth4 = cth2 * cth2;
	s2th = 2. * sth * cth;
	c2th = 2 * cth2 - 1.;

	a2 = d_a * d_a;
	a2sth2 = a2 * sth2;
	a2cth2 = a2 * cth2;
	a3 = a2 * d_a;
	a4 = a3 * d_a;
	a4cth4 = a4 * cth4;

	rho2 = r2 + a2cth2;                
	rho22 = rho2 * rho2;
	rho23 = rho22 * rho2;
	irho2 = 1. / rho2;
	irho22 = irho2 * irho2;
	irho23 = irho22 * irho2;
	irho23_dthdx2 = irho23 / dthdx2;

	fac1 = r2 - a2cth2;
	fac1_rho23 = fac1 * irho23;
	fac2 = a2 + 2 * r2 + a2 * c2th;
	fac3 = a2 + r1 * (-2. + r1);

	lconn[0][0][0] = 2. * r1 * fac1_rho23;
	lconn[0][0][1] = r1 * (2. * r1 + rho2) * fac1_rho23;
	lconn[0][0][2] = -a2 * r1 * s2th * dthdx2 * irho22;
	// printf("a2 = %le\n", a2);
	// printf("r1 = %le\n", r1);
	// printf("irho22 = %le\n", irho22);
	// printf("s2th = %le\n", s2th);
	// printf("dthdx2 = %le\n", dthdx2);

	lconn[0][0][3] = -2. * d_a * r1sth2 * fac1_rho23;

	//lconn[0][1][0] = lconn[0][0][1];
	lconn[0][1][1] = 2. * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
	lconn[0][1][2] = -a2 * r2 * s2th * dthdx2 * irho22;
	lconn[0][1][3] =
	    d_a * r1 * (-r1 * (r3 + 2 * fac1) + a4cth4) * sth2 * irho23;

	//lconn[0][2][0] = lconn[0][0][2];
	//lconn[0][2][1] = lconn[0][1][2];
	lconn[0][2][2] = -2. * r2 * dthdx22 * irho2;
	lconn[0][2][3] = a3 * r1sth2 * s2th * dthdx2 * irho22;
	//lconn[0][3][0] = lconn[0][0][3];
	//lconn[0][3][1] = lconn[0][1][3];
	//lconn[0][3][2] = lconn[0][2][3];
	lconn[0][3][3] =
	    2. * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;

	lconn[1][0][0] = fac3 * fac1 / (r1 * rho23);
	lconn[1][0][1] = fac1 * (-2. * r1 + a2sth2) * irho23;
	lconn[1][0][2] = 0.;
	lconn[1][0][3] = -d_a * sth2 * fac3 * fac1 / (r1 * rho23);

	//lconn[1][1][0] = lconn[1][0][1];
	lconn[1][1][1] =
	    (r4 * (-2. + r1) * (1. + r1) +
	     a2 * (a2 * r1 * (1. + 3. * r1) * cth4 + a4cth4 * cth2 +
		   r3 * sth2 + r1 * cth2 * (2. * r1 + 3. * r3 -
					    a2sth2))) * irho23;
	lconn[1][1][2] = -a2 * dthdx2 * s2th / fac2;
	lconn[1][1][3] =
	    d_a * sth2 * (a4 * r1 * cth4 + r2 * (2 * r1 + r3 - a2sth2) +
			a2cth2 * (2. * r1 * (-1. + r2) + a2sth2)) * irho23;

	//lconn[1][2][0] = lconn[1][0][2];
	//lconn[1][2][1] = lconn[1][1][2];
	lconn[1][2][2] = -fac3 * dthdx22 * irho2;
	lconn[1][2][3] = 0.;

	//lconn[1][3][0] = lconn[1][0][3];
	//lconn[1][3][1] = lconn[1][1][3];
	//lconn[1][3][2] = lconn[1][2][3];
	lconn[1][3][3] =
	    -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);

	lconn[2][0][0] = -a2 * r1 * s2th * irho23_dthdx2;
	lconn[2][0][1] = r1 * lconn[2][0][0];
	lconn[2][0][2] = 0.;
	lconn[2][0][3] = d_a * r1 * (a2 + r2) * s2th * irho23_dthdx2;

	//lconn[2][1][0] = lconn[2][0][1];
	lconn[2][1][1] = r2 * lconn[2][0][0];
	lconn[2][1][2] = r2 * irho2;
	lconn[2][1][3] =
	    (d_a * r1 * cth * sth *
	     (r3 * (2. + r1) +
	      a2 * (2. * r1 * (1. + r1) * cth2 + a2 * cth4 +
		    2 * r1sth2))) * irho23_dthdx2;

	//lconn[2][2][0] = lconn[2][0][2];
	//lconn[2][2][1] = lconn[2][1][2];
	lconn[2][2][2] =
	    -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;

	lconn[2][2][3] = 0.;

	//lconn[2][3][0] = lconn[2][0][3];
	//lconn[2][3][1] = lconn[2][1][3];
	//lconn[2][3][2] = lconn[2][2][3];
	lconn[2][3][3] =
	    -cth * sth * (rho23 +
			  a2sth2 * rho2 * (r1 * (4. + r1) + a2cth2) +
			  2. * r1 * a4 * sth4) * irho23_dthdx2;

	lconn[3][0][0] = d_a * fac1_rho23;
	lconn[3][0][1] = r1 * lconn[3][0][0];
	lconn[3][0][2] = -2. * d_a * r1 * cth * dthdx2 / (sth * rho22);
	lconn[3][0][3] = -a2sth2 * fac1_rho23;

	//lconn[3][1][0] = lconn[3][0][1];
	lconn[3][1][1] = d_a * r2 * fac1_rho23;
	lconn[3][1][2] =
	    -2 * d_a * r1 * (a2 + 2 * r1 * (2. + r1) +
			   a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
	lconn[3][1][3] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

	//lconn[3][2][0] = lconn[3][0][2];
	//lconn[3][2][1] = lconn[3][1][2];
	lconn[3][2][2] = -d_a * r1 * dthdx22 * irho2;
	lconn[3][2][3] =
	    dthdx2 * (0.25 * fac2 * fac2 * cth / sth +
		      a2 * r1 * s2th) * irho22;

	//lconn[3][3][0] = lconn[3][0][3];
	//lconn[3][3][1] = lconn[3][1][3];
	//lconn[3][3][2] = lconn[3][2][3];
	lconn[3][3][3] = (-d_a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;

}
__device__ int GPU_record_criterion(struct of_photon *ph)

{
	const double X1max = log(RMAX);
	/* this is coordinate and simulation
	   specific: stop at large distance */
	//printf("X[1] coord = %le, X1max = %le\n", ph->X[1], X1max);
	if (ph->X[1] > X1max)
		return (1);

	else
		return (0);

}

// void record_super_photon(struct of_photon *ph)
// {
// 	double lE, dx2;
// 	int iE, ix2;

// 	if (isnan(ph->w) || isnan(ph->E)) {
// 		fprintf(stderr, "record isnan: %g %g\n", ph->w, ph->E);
// 		return;
// 	}
// #pragma omp critical (MAXTAU)
// 	{
// 		if (ph->tau_scatt > max_tau_scatt)
// 			max_tau_scatt = ph->tau_scatt;
// 	}
// 	/* currently, bin in x2 coordinate */

// 	/* get theta bin, while folding around equator */
// 	dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);
// 	if (ph->X[2] < 0.5 * (startx[2] + stopx[2]))
// 		ix2 = (int) (ph->X[2] / dx2);
// 	else
// 		ix2 = (int) ((stopx[2] - ph->X[2]) / dx2);

// 	/* check limits */
// 	if (ix2 < 0 || ix2 >= N_THBINS)
// 		return;

// 	/* get energy bin */
// 	lE = log(ph->E);
// 	iE = (int) ((lE - lE0) / dlE + 2.5) - 2;	/* bin is centered on iE*dlE + lE0 */

// 	/* check limits */
// 	if (iE < 0 || iE >= N_EBINS)
// 		return;

// #pragma omp atomic
// 	N_superph_recorded++;
// #pragma omp atomic
// 	N_scatt += ph->nscatt;

// 	/* sum in photon */
// 	spect[ix2][iE].dNdlE += ph->w;
// 	spect[ix2][iE].dEdlE += ph->w * ph->E;
// 	spect[ix2][iE].tau_abs += ph->w * ph->tau_abs;
// 	spect[ix2][iE].tau_scatt += ph->w * ph->tau_scatt;
// 	spect[ix2][iE].X1iav += ph->w * ph->X1i;
// 	spect[ix2][iE].X2isq += ph->w * (ph->X2i * ph->X2i);
// 	spect[ix2][iE].X3fsq += ph->w * (ph->X[3] * ph->X[3]);
// 	spect[ix2][iE].ne0 += ph->w * (ph->ne0);
// 	spect[ix2][iE].b0 += ph->w * (ph->b0);
// 	spect[ix2][iE].thetae0 += ph->w * (ph->thetae0);
// 	spect[ix2][iE].nscatt += ph->nscatt;
// 	spect[ix2][iE].nph += 1.;

// }
__device__ void GPU_record_super_photon(struct of_photon *ph , struct of_spectrum* d_spect) {
    double lE, dx2;
    int iE, ix2;

    if (isnan(ph->w) || isnan(ph->E)) {
        printf("record isnan: %g %g\n", ph->w, ph->E);
        return;
    }

/*TODO: FIX RACE CONDITION BY USING https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf slide 7
taken from https://stackoverflow.com/questions/16785263/cuda-multiple-threads-writing-to-a-shared-variable*/
    if (ph->tau_scatt > d_max_tau_scatt) {
       d_max_tau_scatt = ph->tau_scatt;
    }

    // Bin in x2 coordinate
    dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
    ix2 = (ph->X[2] < 0.5 * (d_startx[2] + d_stopx[2])) ? (int)(ph->X[2] / dx2) : (int)((d_stopx[2] - ph->X[2]) / dx2);

    if (ix2 < 0 || ix2 >= N_THBINS)
        return;

    // Get energy bin
    lE = log(ph->E);
    iE = (int)((lE - d_lE0) / d_dlE + 2.5) - 2;

    if (iE < 0 || iE >= N_EBINS)
        return;

    atomicAdd(&d_N_superph_recorded, 1);
    atomicAdd(&d_N_scatt, ph->nscatt);
    // Sum in photon
	double inbetween = d_spect[(ix2 * N_EBINS) + iE].dNdlE;
    d_spect[(ix2 * N_EBINS) + iE].dNdlE += ph->w;
	if (isnan(d_spect[(ix2 * N_EBINS) + iE].dNdlE) || isinf(d_spect[(ix2 * N_EBINS) + iE].dNdlE))
	printf("d_spect is nan: (%le, %le), %le\n", inbetween, d_spect[(ix2 * N_EBINS) + iE].dNdlE, ph->w);
    d_spect[(ix2 * N_EBINS) + iE].dEdlE += ph->w * ph->E;
    d_spect[(ix2 * N_EBINS) + iE].tau_abs += ph->w * ph->tau_abs;
    d_spect[(ix2 * N_EBINS) + iE].tau_scatt += ph->w * ph->tau_scatt;
    d_spect[(ix2 * N_EBINS) + iE].X1iav += ph->w * ph->X1i;
    d_spect[(ix2 * N_EBINS) + iE].X2isq += ph->w * (ph->X2i * ph->X2i);
    d_spect[(ix2 * N_EBINS) + iE].X3fsq += ph->w * (ph->X[3] * ph->X[3]);
    d_spect[(ix2 * N_EBINS) + iE].ne0 += ph->w * (ph->ne0);
    d_spect[(ix2 * N_EBINS) + iE].b0 += ph->w * (ph->b0);
    d_spect[(ix2 * N_EBINS) + iE].thetae0 += ph->w * (ph->thetae0);
    d_spect[(ix2 * N_EBINS) + iE].nscatt += ph->nscatt;
    d_spect[(ix2 * N_EBINS) + iE].nph += 1.;
}

__device__ double atomicMax_double(double* address, double val) {

    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}
/* return electron scattering opacity, in cgs */
__device__ double GPU_kappa_es(double nu, double Thetae,  double * d_table_ptr)
{
	double Eg;

	/* assume pure hydrogen gas to 
	   convert cross section to opacity */
	Eg = HPL * nu / (ME * CL * CL);
	return (GPU_total_compton_cross_lkup(Eg, Thetae, d_table_ptr) / MP);
}
__device__ double GPU_total_compton_cross_num(double w, double thetae)
{
	double dmue, dgammae, mue, gammae, f, cross;
	__device__ double GPU_dNdgammae(double thetae, double gammae);
	__device__ double GPU_boostcross(double w, double mue, double gammae);
	__device__ double GPU_hc_klein_nishina(double we);

	if (isnan(w)) {
		printf("compton cross isnan: %g %g\n", w, thetae);
		return (0.);
	}

	/* check for easy-to-do limits */
	if (thetae < MINT && w < MINW)
		return (SIGMA_THOMSON);
	if (thetae < MINT)
		return (GPU_hc_klein_nishina(w) * SIGMA_THOMSON);

	dmue = DMUE;
	dgammae = thetae * DGAMMAE;

	/* integrate over mu_e, gamma_e, where mu_e is the cosine of the
	   angle between k and u_e, and the angle k is assumed to lie,
	   wlog, along the z axis */
	cross = 0.;
	for (mue = -1. + 0.5 * dmue; mue < 1.; mue += dmue)
		for (gammae = 1. + 0.5 * dgammae;
		     gammae < 1. + MAXGAMMA * thetae; gammae += dgammae) {

			f = 0.5 * GPU_dNdgammae(thetae, gammae);

			cross +=
			    dmue * dgammae * GPU_boostcross(w, mue,
							gammae) * f;

			if (isnan(cross)) {
				printf("Problem in GPU_hc_klein_nishina, cross is nan\n");
				printf("%g %g %g %g %g %g\n", w,
					thetae, mue, gammae,
					GPU_dNdgammae(thetae, gammae),
					GPU_boostcross(w, mue, gammae));
			}
		}


	return (cross * SIGMA_THOMSON);
}
__device__ double GPU_dNdgammae(double thetae, double gammae)
{
	double K2f;

	if (thetae > 1.e-2) {
		//K2f = gsl_sf_bessel_Kn(2, 1. / thetae) * exp(1. / thetae);
		K2f = bessk2(1. / thetae) * exp(1. / thetae); /*TODO: Check if this function is working correctly*/
	} else {
		K2f = sqrt(M_PI * thetae / 2.);
	}

	return ((gammae * sqrt(gammae * gammae - 1.) / (thetae * K2f)) *
		exp(-(gammae - 1.) / thetae));
}
__device__ double GPU_boostcross(double w, double mue, double gammae)
{
	double we, boostcross, v;
	__device__ double GPU_hc_klein_nishina(double we);

	/* energy in electron rest frame */
	v = sqrt(gammae * gammae - 1.) / gammae;
	we = w * gammae * (1. - mue * v);

	boostcross = GPU_hc_klein_nishina(we) * (1. - mue * v);

	if (boostcross > 2) {
		printf("w,mue,gammae: %g %g %g\n", w, mue,
			gammae);
		printf("v,we, boostcross: %g %g %g\n", v, we,
			boostcross);
		printf("kn: %g %g %g\n", v, we, boostcross);
	}

	if (isnan(boostcross)) {
		printf("isnan: %g %g %g\n", w, mue, gammae);
		printf("The code should exit, problem in function GPU_boostcross");
		//exit(0);
	}

	return (boostcross);
}
__device__ double GPU_hc_klein_nishina(double we)
{
	double sigma;

	if (we < 1.e-3)
		return (1. - 2. * we);

	sigma = (3. / 4.) * (2. / (we * we) +
			     (1. / (2. * we) -
			      (1. + we) / (we * we * we)) * log(1. +
								2. * we) +
			     (1. + we) / ((1. + 2. * we) * (1. + 2. * we))
	    );

	return (sigma);

}
/*Bessel0 function defined as Numerical Recipes book*/
__device__ double bessi0(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    }
    else
    {
        y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y *
                                                                                                                                     (-0.2057706e-1 +
                                                                                                                                      y *
                                                                                                                                          (0.2635537e-1 +
                                                                                                                                           y *
                                                                                                                                               (-0.1647633e-1 + y *
                                                                                                                                                                    0.392377e-2))))))));
    }
    return ans;
}
/*Bessel1 function defined as Numerical Recipes book*/
__device__ double bessi1(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 +
                                                                                     y * (0.301532e-2 + y * 0.32411e-3))))));
    }
    else
    {
        y = 3.75 / ax;
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
        ans *= (exp(ax) / sqrt(ax));
    }
    return xbess < 0.0 ? -ans : ans;
}
/*Modified bessel0 function defined as Numerical Recipes book*/
__device__ double bessk0(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (-log(xbess / 2.0) * bessi0(xbess)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 +
                                                                                          y * (0.3488590e-1 + y * (0.262698e-2 + y *
                                                                                                                                     (0.10750e-3 +
                                                                                                                                      y *
                                                                                                                                          0.74e-5))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (-0.7832358e-1 +
                                                               y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y *
                                                                                                                               (-0.251540e-2 +
                                                                                                                                y *
                                                                                                                                    0.53208e-3))))));
    }
    return ans;
}
/*Modified bessel1 function defined as Numerical Recipes book*/
__device__ double bessk1(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (log(xbess / 2.0) * bessi1(xbess)) + (1.0 / xbess) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 +
                                                                                                                    y *
                                                                                                                        (-0.1919402e-1 + y *
                                                                                                                                             (-0.110404e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.4686e-4)))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (0.23498619 + y *
                                                                                (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2 + y *
                                                                                                                                             (0.325614e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.68245e-3)))))));
    }
    return ans;
}
/*Modified bessel2 function defined as Numerical Recipes book*/
__device__ double bessk2(double xbess)
{
    int n, j;
    double bk, bkm, bkp, tox;
    n = 2;
    tox = 2.0 / xbess;
    bkm = bessk0(xbess);
    bk = bessk1(xbess);
    for (j = 1; j < n; j++)
    {
        bkp = bkm + j * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}
__device__ double GPU_jnu_inv(double nu, double Thetae, double Ne, double B, double theta)
{
	double j;

	j = GPU_jnu_synch(nu, Ne, Thetae, B, theta);

	return (j / (nu * nu));
}
__device__ double GPU_Bnu_inv(double nu, double Thetae)
{

	double x;

	x = HPL * nu / (ME * CL * CL * Thetae);

	if (x < 1.e-3)		/* Taylor expand */
		return ((2. * HPL / (CL * CL)) /
			(x / 24. * (24. + x * (12. + x * (4. + x)))));
	else
		return ((2. * HPL / (CL * CL)) / (exp(x) - 1.));
}
__device__ double GPU_total_compton_cross_lkup(double w, double thetae, double * d_table_ptr)
{
	int i, j;
	double lw, lT, di, dj, lcross;
	__device__ double GPU_total_compton_cross_num(double w, double thetae);
	__device__ double GPU_hc_klein_nishina(double we);

	/* cold/low-energy: just use thomson cross section */
	if (w * thetae < 1.e-6){
		return (SIGMA_THOMSON);
	}

	/* cold, but possible high energy photon: use klein-nishina */
	if (thetae < MINT){
		return (GPU_hc_klein_nishina(w) * SIGMA_THOMSON);
	}

	/* in-bounds for table */
	if ((w > MINW && w < MAXW) && (thetae > MINT && thetae < MAXT)) {

		lw = log10(w);
		lT = log10(thetae);
		i = (int) ((lw - d_lminw) / d_dlw);
		j = (int) ((lT - d_lmint) / d_dlT);
		di = (lw - d_lminw) / d_dlw - i;
		dj = (lT - d_lmint) / d_dlT - j;

		lcross =
		    (1. - di) * (1. - dj) * d_table_ptr[j + (NT+1) * i] + di * (1. -
								dj) *
		    d_table_ptr[j + (NT+1) * (i+1)] + (1. - di) * dj * d_table_ptr[(j+1) + (NT+1) * i] +
		    di * dj * d_table_ptr[(j+1) + (NT+1) * (i+1)];

		if (isnan(lcross)) {
			printf("Problem in GPU_total_compton_cross_lkup, lcross is nan!\n");	
			printf("lw = %g. lT =  %g, i =  %d, j =  %d, di =  %g, dj =  %g\n", lw, lT, i,
				j, di, dj);
			printf("table[i][j] = %le, table[i][j + 1] = %le, table[i +1][j] = %le, table[i+1][j+1] = %le\n", d_table_ptr[j + (NT+1) * i], d_table_ptr[(j+1) + (NT+1) * i], d_table_ptr[j + (NT+1) * (i+1)], d_table_ptr[(j+1) + (NT+1) * (i+1)]);
		}
		// printf("lcross = %le\n", lcross);
		// printf("table[i][j] = %le, table[i][j + 1] = %le, table[i +1][j] = %le, table[i+1][j+1] = %le\n", d_table_ptr[j + (NT+1) * i], d_table_ptr[(j+1) + (NT+1) * i], d_table_ptr[j + (NT+1) * (i+1)], d_table_ptr[(j+1) + (NT+1) * (i+1)]);
		// printf("lw = %g. lT =  %g, i =  %d, j =  %d, di =  %g, dj =  %g\n", lw, lT, i, j, di, dj);
		return (pow(10., lcross));
	}
	printf("out of bounds: %g %g\n", w, thetae);
	
	return (GPU_total_compton_cross_num(w, thetae));

}








































// void allocMemory(){
//     // Allocate device memory for each global variable
//     /*Model Global Parameters*/
//     printf( "Allocating memory on device!\n");
// 	// gpuErrchk(cudaMalloc((void**)&d_N1, sizeof(int)));
// 	// gpuErrchk(cudaMalloc((void**)&d_N2, sizeof(int)));
// 	// gpuErrchk(cudaMalloc((void**)&d_N3, sizeof(int)));
//     // gpuErrchk(cudaMalloc((void**)&d_startx, NDIM* sizeof(double)));
//     // gpuErrchk(cudaMalloc((void**)&d_dx, NDIM * sizeof(double)));
// 	// gpuErrchk(cudaMalloc((void**)&d_a, sizeof(double)));
// 	// gpuErrchk(cudaMalloc((void**)&d_gam, sizeof(double)));
// 	// gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
// 	// gpuErrchk(cudaMalloc((void**)&d_bias_norm, sizeof(double))); 
// }

// void freeMemory() {
//     // Free device memory for each model variable
//     printf( "Freeing memory on device!\n");
//     // cudaFree(d_N1);
//     // cudaFree(d_N2);
//     // cudaFree(d_N3);
//     // cudaFree(d_startx);
//     // cudaFree(d_dx);
//     // cudaFree(d_a);
//     // cudaFree(d_gam);
//     // cudaFree(d_p);
//     // cudaFree(d_bias_norm);
// }

// void transfer_data_to_GPU(){
//     // Perform cudaMemcpy for each variable
//     // printf( "Copying global model parameters from host to device!\n");
// 	// cudaMemcpyErrorCheck(d_a, &a, sizeof(double), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_N1, &N1, sizeof(int), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_N2, &N2, sizeof(int), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_N3, &N3, sizeof(int), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_gam, &gam, sizeof(double), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_startx, startx, NDIM * sizeof(double), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_dx, dx, NDIM * sizeof(double), cudaMemcpyHostToDevice);
//     // cudaMemcpyErrorCheck(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice);
// }


// __global__ void reading_data (double * p, int n1, int n2, int n3){
//     int i,j,l;
//     for(int k = 0; k < n1 * n2 * n3; k++){
//         l = 0;
// 		j = k % n2;
// 		i = (k - j) / n2;
//         //printf("Device k = %d\n", k);
//         printf("Device p[KRHO][%d][%d][%d] = %le\n", i, j, l, p[(KRHO * (n1 * n2 * n3) + j)]);
//     }
//     //printf("dp[KRHO][%d][%d][%d] is equal to = %le")
// }


/* this has precious information about copying structs*/
// void GPU_Memory(struct of_geom *geom)
// {
//     /*First we will declare the variables in device*/

//     int *d_N2, *d_N1;
//     struct of_geom *d_geom;
//     cudaMalloc(&d_N2, sizeof(int));
//     cudaMalloc(&d_N1, sizeof(int));
//     cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom));

//     printf( "N1 = %d, N2 = %d\n", N1, N2);
//     printf( "gcov[10, 0] = %lf\n", geom[DEVICE_SPATIAL_INDEX2D(10, 0)].gcov[0][0]);
//     printf( "gcon[10, 0] = %lf\n", geom[DEVICE_SPATIAL_INDEX2D(10, 0)].gcon[0][0]);

//     cudaMemcpy(d_N1, &N1, sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_N2, &N2, sizeof(int), cudaMemcpyHostToDevice);
//     // Copying data from host struct to device
//     cudaMemcpy(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice);
//     test_struct_data2<<<1, 30>>>(d_N1, d_N2, d_geom);

//     // Synchronize to ensure the kernel has completed
//     cudaDeviceSynchronize();
// }

// #define SPECTRUM_FILE_NAME	"./output/grmonty_hamr.spec"

// void report_spectrum(int N_superph_made)
// {
// 	int i, j;
// 	double dx2, dOmega, nuLnu, tau_scatt, L;
// 	FILE *fp;

// 	fp = fopen(SPECTRUM_FILE_NAME, "w");
// 	if (fp == NULL) {
// 		printf("trouble opening spectrum file\n");
// 		exit(0);
// 	}

// 	/* output */
// 	max_tau_scatt = 0.;
// 	L = 0.;
// 	for (i = 0; i < N_EBINS; i++) {

// 		/* output log_10(photon energy/(me c^2)) */
// 		fprintf(fp, "%10.5g ", (i * dlE + lE0) / M_LN10);

// 		for (j = 0; j < N_THBINS; j++) {

// 			/* convert accumulated photon number in each bin 
// 			   to \nu L_\nu, in units of Lsun */
// 			dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);

// 			/* factor of 2 accounts for folding around equator */
// 			dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);

// 			nuLnu =
// 			    (ME * CL * CL) * (4. * M_PI / dOmega) * (1. /
// 								     dlE);

// 			nuLnu *= spect[j][i].dEdlE;
// 			nuLnu /= LSUN;

// 			tau_scatt =
// 			    spect[j][i].tau_scatt / (spect[j][i].dNdlE +
// 						     SMALL);
// 			fprintf(fp,
// 				"%10.5g %10.5g %10.5g %10.5g %10.5g %10.5g ",
// 				nuLnu,
// 				spect[j][i].tau_abs / (spect[j][i].dNdlE +
// 						       SMALL), tau_scatt,
// 				spect[j][i].X1iav / (spect[j][i].dNdlE +
// 						     SMALL),
// 				sqrt(fabs
// 				     (spect[j][i].X2isq /
// 				      (spect[j][i].dNdlE + SMALL))),
// 				sqrt(fabs
// 				     (spect[j][i].X3fsq /
// 				      (spect[j][i].dNdlE + SMALL)))
// 			    );

// 			if (tau_scatt > max_tau_scatt)
// 				max_tau_scatt = tau_scatt;

// 			L += nuLnu * dOmega * dlE;
// 		}
// 		fprintf(fp, "\n");
// 	}
// 	printf(
// 		"luminosity %g, dMact %g, efficiency %g, L/Ladv %g, max_tau_scatt %g\n",
// 		L, dMact * M_UNIT / T_UNIT / (MSUN / YEAR),
// 		L * LSUN / (dMact * M_UNIT * CL * CL / T_UNIT),
// 		L * LSUN / (Ladv * M_UNIT * CL * CL / T_UNIT),
// 		max_tau_scatt);
// 	printf("\n");
// 	printf("N_superph_made: %d\n", N_superph_made);
// 	printf("N_superph_recorded: %d\n", N_superph_recorded);

// 	fclose(fp);

// }