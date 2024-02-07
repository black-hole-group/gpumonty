#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <curand_kernel.h>

extern "C"
{
#include "decs.h"
}

#include "gpu_header.h"

#include "defs_CUDA.h"



/*TODO: PASSAR AS VARIAVEIS STRUCT OF_PHOTON para o device*/

// Define the device random number generator state
__device__ curandState my_curand_state;

__global__ void test_struct_data2(){
	//printf("startx = (%lf, %lf, %lf, %lf), dx = (%lf, %lf, %lf, %lf)\n", d_startx[0], d_startx[1], d_startx[2], d_startx[3], d_dx[0], d_dx[1], d_dx[2], d_dx[3]);
	/*Done transfering geom*/
	// for (int i = 0; i < (N_ESAMP +1); i++){
	// 	printf("d_wgt[%d] = %le\n", i, d_wgt[i]);
	// }
}
void launch_loop(struct of_photon ph, int quit_flag, time_t time, double * p){
	/*Copying global variables*/
    cudaMemcpyToSymbol(d_N1, &N1, sizeof(int));
	cudaMemcpyToSymbol(d_Ns, &Ns, sizeof(int));
    cudaMemcpyToSymbol(d_N2, &N2, sizeof(int));
    cudaMemcpyToSymbol(d_N3, &N3, sizeof(int));
    cudaMemcpyToSymbol(d_dx, &dx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_startx, &startx, NDIM * sizeof(double));
	cudaMemcpyToSymbol(d_a, &a, sizeof(double));
	cudaMemcpyToSymbol(d_thetae_unit, &Thetae_unit, sizeof(double));
    gpuErrchk(cudaMalloc((void**)&d_p, NPRIM * N1 * N2 * N3*sizeof(double)));
    cudaMemcpyErrorCheck(d_p, p, NPRIM * N1 * N2 * N3* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_wgt, &wgt, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_F, &F, (N_ESAMP + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_nint, &nint, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_dndlnu_max, &dndlnu_max, (NINT + 1) * sizeof(double));
	cudaMemcpyToSymbol(d_K2, &K2, (N_ESAMP + 1) * sizeof(double));


	/*Transfering geom array*/
	struct of_geom *d_geom;
	gpuErrchk(cudaMalloc(&d_geom, N1 * N2 * sizeof(struct of_geom)));
    cudaMemcpyErrorCheck(d_geom, geom, N1 * N2 * sizeof(struct of_geom), cudaMemcpyHostToDevice);
	/*Done transfering geom*/

	// for (int i = 0; i < (N_ESAMP +1); i++){
	// 	printf( "Wgt[%d] = %le\n", i, wgt[i]);
	// }

	//test_struct_data2<<<1, 1>>>();
    GPU_mainloop<<<1,1>>>(ph, time, d_geom, d_p);
    cudaDeviceSynchronize();
}


__global__ void GPU_mainloop(struct of_photon ph, time_t time, struct of_geom * d_geom, double * d_p)
{
	struct of_photon d_ph = ph;
	int quit_flag = 0;
    int seed = 139 * (blockIdx.x * blockDim.x + threadIdx.x) + time;  // Use a different seed for each thread
    GPU_init_monty_rand(seed); /*Maybe there is a better way to do this?*/
    while(1){
        /*First thing we should do is make super photon*/
        if (!quit_flag){
            GPU_make_super_photon(&d_ph, &quit_flag, d_geom, d_p);
        }
			//printf("quit_flag after= %d", quit_flag);
			if (quit_flag){
				break;
            }
		
    }

}

__device__ void GPU_make_super_photon(struct of_photon *ph, int *quit_flag, struct of_geom * d_geom, double * d_p)
{
    int n2gen = -1;
    double dnmax;
    int zone_i, zone_j, zone_k;

	while (n2gen <= 0) {
		n2gen = GPU_get_zone(&zone_i, &zone_j, &zone_k, &dnmax, d_geom, d_p);
	}

	n2gen--;
	if (zone_i == d_N1)
		*quit_flag = 1;
	else
		*quit_flag = 0;

	if (*quit_flag != 1) {
		/* Initialize the superphoton energy, direction, weight, etc. */
		GPU_sample_zone_photon(zone_i, zone_j, zone_k, dnmax, ph, d_geom, d_p);
	}

	return;
}

__device__ int GPU_get_zone(int *i, int *j, int *k, double *dnmax, struct of_geom * d_geom, double * d_p)
{
/* Return the next zone and the number of superphotons that need to be		*
 * generated in it.								*/
	int in2gen;
	double n2gen;
	static int zi = 0;
	static int zj = 0;
	static int zk = -1;
	zone_flag = 1;
	zk++;
	if(zk >= d_N3){
		zk = 0;
		zj++;
		if (zj >= d_N2) {
			zj = 0;
			printf( "zi = %d\n", zi);
			zi++;
			if (zi >= d_N1) {
				in2gen = 1;
				*i = d_N1;
				return 1;
			}
		}
	}
	GPU_init_zone(zi, zj, zk, &n2gen, dnmax, d_geom, d_p);

	if (fmod(n2gen, 1.) > GPU_monty_rand()) {
		in2gen = (int) n2gen + 1;
	} else {
		in2gen = (int) n2gen;
	}

	*i = zi;
	*j = zj;
	*k = zk;

	return in2gen;
}

__device__ void GPU_sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph, struct of_geom * d_geom, double * d_p)
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
		nu = exp(GPU_monty_rand() * Nln + lnu_min);
		weight = GPU_linear_interp_weight(nu);
	} while (GPU_monty_rand() >
		 (GPU_F_eval(Thetae, Bmag, nu) / (weight + 1.e-100)) / dnmax);

	ph->w = weight;
	jmax = GPU_jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);
	do {
		cth = 2. * GPU_monty_rand() - 1.;
		th = acos(cth);

	} while (GPU_monty_rand() >
		 GPU_jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax);

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

__device__ void GPU_init_monty_rand(int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &my_curand_state);
}

__device__ double GPU_monty_rand() {
    return curand_uniform_double(&my_curand_state) - 1e-10;
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
		//printf( "K > Kmax\n");
		return 0.;
	} else if (K < KMIN) {
		/* use a good approximation */
		x = pow(K, 0.333333333333333333);
		//printf( "K < Kmin// x= %le\n", x);

		return (x * (37.67503800178 + 2.240274341836 * x));
	} else {
		//printf( "normal print K = %le, nu = %le, Bmag = %le, Thetae = %le\n", K, nu, Bmag, Thetae);
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
	int i;
	double di, lK;
	lK = log(K);
	di = (lK - lK_min) * dlK;
	i = (int) di;
	di = di - i;

	return exp((1. - di) * d_F[i] + di * d_F[i + 1]);
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
	for (k = 0; k < 4; k++)
		Econ[0][k] = Ucon[k];
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

