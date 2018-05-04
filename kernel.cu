#include <stdio.h>
#include "constants.h"
#include "kernel.h"  

#define TPB 32 // number of threads per block 
#define MAXNSTEP	1280000 // for geodesic integration



/*
  device global variables
  ========================
*/
// harm dimensions
 __constant__ int N1, N2, N3; 
 __constant__ int n_within_horizon;

// some coordinate parameters 
 __constant__ double a;
 __constant__ double R0, Rin, Rh, Rout, Rms;
 __constant__ double hslope;
 __constant__ double startx[NDIM], stopx[NDIM], dx[NDIM];
 __constant__ double dlE, lE0;
 __constant__ double gam;
 __constant__ double dMsim;

// units
 __constant__ double M_unit;
 __constant__ double L_unit;
 __constant__ double T_unit;
 __constant__ double RHO_unit;
 __constant__ double U_unit;
 __constant__ double B_unit;
 __constant__ double Ne_unit;
 __constant__ double Thetae_unit;


/* 
  Device functions
  =================
*/
#include "harm_utils.cuh"
#include "harm_model.cuh" // device functions previously in harm_model.c







/* grabs a photon from the device input array and puts it
   in a ph struct,
   such that minimal modification is required with respect to
   previously written host-code.
*/
__device__
struct d_photon arr2struct(int i, double *pharr) 
{
	struct d_photon ph;

	ph.X[0]=pharr[i*NPHVARS+X0];
	ph.X[1]=pharr[i*NPHVARS+X1]; 
	ph.X[2]=pharr[i*NPHVARS+X2]; 
	ph.X[3]=pharr[i*NPHVARS+X3];
	ph.K[0]=pharr[i*NPHVARS+K0_];
	ph.K[1]=pharr[i*NPHVARS+K1_];
	ph.K[2]=pharr[i*NPHVARS+K2_];
	ph.K[3]=pharr[i*NPHVARS+K3_];
	ph.dKdlam[0]=pharr[i*NPHVARS+D0];
	ph.dKdlam[1]=pharr[i*NPHVARS+D1];
	ph.dKdlam[2]=pharr[i*NPHVARS+D2];
	ph.dKdlam[3]=pharr[i*NPHVARS+D3];
	ph.w=pharr[i*NPHVARS+W];
	ph.E=pharr[i*NPHVARS+E_];
	ph.L=pharr[i*NPHVARS+L_];
	ph.X1i=pharr[i*NPHVARS+X1I];
	ph.X2i=pharr[i*NPHVARS+X2I];
	ph.tau_abs=pharr[i*NPHVARS+TAUA];
	ph.tau_scatt=pharr[i*NPHVARS+TAUS];
	ph.ne0=pharr[i*NPHVARS+NE0];
	ph.thetae0=pharr[i*NPHVARS+TH0];
	ph.b0=pharr[i*NPHVARS+B0];
	ph.E0=pharr[i*NPHVARS+E0_];
	ph.E0s=pharr[i*NPHVARS+E0S];
	ph.nscatt=pharr[i*NPHVARS+NS];

	return ph;
}





/*
	main transport subroutine for tracking, absorbing,
	and scattering superphotons

	assumes superphotons do not step out of simulation then back in
*/
__global__
void track_super_photon(double *d_p, double *d_pharr, int nph)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nph) return;

	// how to grab each photon:
	// printf("photon[%d]=%lf\n", i,d_pharr[i*nphvars+var]);

	/* grab the photon i corresponding to the current thread from
	   the input array.
	   Notice that I might be using unnecessary device memory here.
	   Should investigate using a struct pointing to d_pharr instead.
	*/
	struct d_photon ph=arr2struct(i, d_pharr);

	int bound_flag;
	double dtau_scatt, dtau_abs, dtau;
	double bi, bf;
	double alpha_scatti, alpha_scattf;
	double alpha_absi, alpha_absf;
	double dl, x1;
	double nu, Thetae, Ne, B, theta;
	struct d_photon php;
	double dtauK, frac;
	double bias = 0.;
	double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
	double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM],
	    Bcov[NDIM];
	int nstep = 0;

	/* quality control 
	   here, previously we had statements ph->X[0] which were
	   replaced by ph.X[0].
	*/
	if (isnan(ph.X[0]) || 
	    isnan(ph.X[1]) ||
	    isnan(ph.X[2]) ||
	    isnan(ph.X[3]) ||
	    isnan(ph.K[0]) ||
	    isnan(ph.K[1]) ||
	    isnan(ph.K[2]) || isnan(ph.K[3]) || ph.w == 0.) {
		// fprintf(stderr, "track_super_photon: bad input photon.\n");
		// fprintf(stderr,
		// 	"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
		// 	ph.X[0], ph.X[1], ph.X[2], ph.X[3], ph.K[0],
		// 	ph.K[1], ph.K[2], ph.K[3], ph.w, ph.nscatt);
		printf("track_super_photon: bad input photon.\n");
		printf("X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			ph.X[0], ph.X[1], ph.X[2], ph.X[3], ph.K[0],
			ph.K[1], ph.K[2], ph.K[3], ph.w, ph.nscatt);
		return;
	}

	dtauK = 2. * M_PI * L_unit / (ME * CL * CL / HBAR);

	/* Initialize opacities */
	d_gcov_func(ph.X, Gcov);
	get_fluid_params(ph.X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
			 Bcov);

	// theta = get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
	// nu = get_fluid_nu(ph->X, ph->K, Ucov);
	// alpha_scatti = alpha_inv_scatt(nu, Thetae, Ne);
	// alpha_absi = alpha_inv_abs(nu, Thetae, Ne, B, theta);
	// bi = bias_func(Thetae, ph->w);

	// /* Initialize dK/dlam */
	// init_dKdlam(ph->X, ph->K, ph->dKdlam);

	// /* This loop solves radiative transfer equation along a geodesic */
	// while (!stop_criterion(ph)) {

	// 	/* Save initial position/wave vector */
	// 	Xi[0] = ph->X[0];
	// 	Xi[1] = ph->X[1];
	// 	Xi[2] = ph->X[2];
	// 	Xi[3] = ph->X[3];
	// 	Ki[0] = ph->K[0];
	// 	Ki[1] = ph->K[1];
	// 	Ki[2] = ph->K[2];
	// 	Ki[3] = ph->K[3];
	// 	dKi[0] = ph->dKdlam[0];
	// 	dKi[1] = ph->dKdlam[1];
	// 	dKi[2] = ph->dKdlam[2];
	// 	dKi[3] = ph->dKdlam[3];
	// 	E0 = ph->E0s;

	// 	/* evaluate stepsize */
	// 	dl = stepsize(ph->X, ph->K);

	// 	/* step the geodesic */
	// 	push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s), 0);
	// 	if (stop_criterion(ph))
	// 		break;

	// 	/* allow photon to interact with matter, */
	// 	gcov_func(ph->X, Gcov);
	// 	get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov,
	// 			 Bcon, Bcov);
	// 	if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {

	// 		bound_flag = 0;
	// 		if (Ne == 0.)
	// 			bound_flag = 1;
	// 		if (!bound_flag) {
	// 			theta =
	// 			    get_bk_angle(ph->X, ph->K, Ucov, Bcov,
	// 					 B);
	// 			nu = get_fluid_nu(ph->X, ph->K, Ucov);
	// 			if (isnan(nu)) {
	// 				fprintf(stderr,
	// 					"isnan nu: track_super_photon dl,E0 %g %g\n",
	// 					dl, E0);
	// 				fprintf(stderr,
	// 					"Xi, %g %g %g %g\n", Xi[0],
	// 					Xi[1], Xi[2], Xi[3]);
	// 				fprintf(stderr,
	// 					"Ki, %g %g %g %g\n", Ki[0],
	// 					Ki[1], Ki[2], Ki[3]);
	// 				fprintf(stderr,
	// 					"dKi, %g %g %g %g\n",
	// 					dKi[0], dKi[1], dKi[2],
	// 					dKi[3]);
	// 				exit(1);
	// 			}
	// 		}

	// 		/* scattering optical depth along step */
	// 		if (bound_flag || nu < 0.) {
	// 			dtau_scatt =
	// 			    0.5 * alpha_scatti * dtauK * dl;
	// 			dtau_abs = 0.5 * alpha_absi * dtauK * dl;
	// 			alpha_scatti = alpha_absi = 0.;
	// 			bias = 0.;
	// 			bi = 0.;
	// 		} else {
	// 			alpha_scattf =
	// 			    alpha_inv_scatt(nu, Thetae, Ne);
	// 			dtau_scatt =
	// 			    0.5 * (alpha_scatti +
	// 				   alpha_scattf) * dtauK * dl;
	// 			alpha_scatti = alpha_scattf;

	// 			/* absorption optical depth along step */
	// 			alpha_absf =
	// 			    alpha_inv_abs(nu, Thetae, Ne, B,
	// 					  theta);
	// 			dtau_abs =
	// 			    0.5 * (alpha_absi +
	// 				   alpha_absf) * dtauK * dl;
	// 			alpha_absi = alpha_absf;

	// 			bf = bias_func(Thetae, ph->w);
	// 			bias = 0.5 * (bi + bf);
	// 			bi = bf;
	// 		}

	// 		x1 = -log(monty_rand());
	// 		php.w = ph->w / bias;
	// 		if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
	// 			if (isnan(php.w) || isinf(php.w)) {
	// 				fprintf(stderr,
	// 					"w isnan in track_super_photon: Ne, bias, ph->w, php.w  %g, %g, %g, %g\n",
	// 					Ne, bias, ph->w, php.w);
	// 			}

	// 			frac = x1 / (bias * dtau_scatt);

	// 			/* Apply absorption until scattering event */
	// 			dtau_abs *= frac;
	// 			if (dtau_abs > 100)
	// 				return;	/* This photon has been absorbed before scattering */

	// 			dtau_scatt *= frac;
	// 			dtau = dtau_abs + dtau_scatt;
	// 			if (dtau_abs < 1.e-3)
	// 				ph->w *=
	// 				    (1. -
	// 				     dtau / 24. * (24. -
	// 						   dtau * (12. -
	// 							   dtau *
	// 							   (4. -
	// 							    dtau))));
	// 			else
	// 				ph->w *= exp(-dtau);

	// 			/* Interpolate position and wave vector to scattering event */
	// 			push_photon(Xi, Ki, dKi, dl * frac, &E0,
	// 				    0);
	// 			ph->X[0] = Xi[0];
	// 			ph->X[1] = Xi[1];
	// 			ph->X[2] = Xi[2];
	// 			ph->X[3] = Xi[3];
	// 			ph->K[0] = Ki[0];
	// 			ph->K[1] = Ki[1];
	// 			ph->K[2] = Ki[2];
	// 			ph->K[3] = Ki[3];
	// 			ph->dKdlam[0] = dKi[0];
	// 			ph->dKdlam[1] = dKi[1];
	// 			ph->dKdlam[2] = dKi[2];
	// 			ph->dKdlam[3] = dKi[3];
	// 			ph->E0s = E0;

	// 			/* Get plasma parameters at new position */
	// 			gcov_func(ph->X, Gcov);
	// 			get_fluid_params(ph->X, Gcov, &Ne, &Thetae,
	// 					 &B, Ucon, Ucov, Bcon,
	// 					 Bcov);

	// 			if (Ne > 0.) {
	// 				scatter_super_photon(ph, &php, Ne,
	// 						     Thetae, B,
	// 						     Ucon, Bcon,
	// 						     Gcov);
	// 				if (ph->w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
	// 					return;
	// 				}
	// 				track_super_photon(&php);
	// 			}

	// 			theta =
	// 			    get_bk_angle(ph->X, ph->K, Ucov, Bcov,
	// 					 B);
	// 			nu = get_fluid_nu(ph->X, ph->K, Ucov);
	// 			if (nu < 0.) {
	// 				alpha_scatti = alpha_absi = 0.;
	// 			} else {
	// 				alpha_scatti =
	// 				    alpha_inv_scatt(nu, Thetae,
	// 						    Ne);
	// 				alpha_absi =
	// 				    alpha_inv_abs(nu, Thetae, Ne,
	// 						  B, theta);
	// 			}
	// 			bi = bias_func(Thetae, ph->w);

	// 			ph->tau_abs += dtau_abs;
	// 			ph->tau_scatt += dtau_scatt;

	// 		} else {
	// 			if (dtau_abs > 100)
	// 				return;	/* This photon has been absorbed */
	// 			ph->tau_abs += dtau_abs;
	// 			ph->tau_scatt += dtau_scatt;
	// 			dtau = dtau_abs + dtau_scatt;
	// 			if (dtau < 1.e-3)
	// 				ph->w *=
	// 				    (1. -
	// 				     dtau / 24. * (24. -
	// 						   dtau * (12. -
	// 							   dtau *
	// 							   (4. -
	// 							    dtau))));
	// 			else
	// 				ph->w *= exp(-dtau);
	// 		}
	// 	}

	// 	nstep++;

	// 	/* signs that something's wrong w/ the integration */
	// 	if (nstep > MAXNSTEP) {
	// 		fprintf(stderr,
	// 			"X1,X2,K1,K2,bias: %g %g %g %g %g\n",
	// 			ph->X[1], ph->X[2], ph->K[1], ph->K[2],
	// 			bias);
	// 		break;
	// 	}

	// }

	// /* accumulate result in spectrum on escape */
	// if (record_criterion(ph) && nstep < MAXNSTEP)
	// 	record_super_photon(ph);

	/* done! */
}



// __global__
// void test()
// {
// 	//const int i = blockIdx.x*blockDim.x + threadIdx.x;

// 	//if (i >= nph) return;
// 	printf("%d\n", N1);
// }





void launchKernel(double *p, simvars sim, allunits units, double *pharr, int nph) 
{
	// device variables
	double *d_p=0; // HARM arrays
	double *d_pharr=0; // superphoton array

	// define global device variables in constant memory
	cudaMemcpyToSymbol(N1, &sim.N1, sizeof(int));
	cudaMemcpyToSymbol(N2, &sim.N2, sizeof(int));
	cudaMemcpyToSymbol(N3, &sim.N3, sizeof(int));
	cudaMemcpyToSymbol(n_within_horizon, &sim.n_within_horizon, sizeof(int));
	cudaMemcpyToSymbol(a, &sim.a, sizeof(double));
	cudaMemcpyToSymbol(R0, &sim.R0, sizeof(double));
	cudaMemcpyToSymbol(Rin, &sim.Rin, sizeof(double));
	cudaMemcpyToSymbol(Rh, &sim.Rh, sizeof(double));
	cudaMemcpyToSymbol(Rout, &sim.Rout, sizeof(double));
	cudaMemcpyToSymbol(Rms, &sim.Rms, sizeof(double));
	cudaMemcpyToSymbol(hslope, &sim.hslope, sizeof(double));
	cudaMemcpyToSymbol(startx, &sim.startx, sizeof(double));
	cudaMemcpyToSymbol(stopx, &sim.stopx, sizeof(double));
	cudaMemcpyToSymbol(dx, &sim.dx, sizeof(double));
	cudaMemcpyToSymbol(dlE, &sim.dlE, sizeof(double));
	cudaMemcpyToSymbol(lE0, &sim.lE0, sizeof(double));
	cudaMemcpyToSymbol(gam, &sim.gam, sizeof(double));
	cudaMemcpyToSymbol(dMsim, &sim.dMsim, sizeof(double));
	cudaMemcpyToSymbol(M_unit, &units.M_unit, sizeof(double));
	cudaMemcpyToSymbol(L_unit, &units.L_unit, sizeof(double));
	cudaMemcpyToSymbol(T_unit, &units.T_unit, sizeof(double));
	cudaMemcpyToSymbol(RHO_unit, &units.RHO_unit, sizeof(double));
	cudaMemcpyToSymbol(U_unit, &units.U_unit, sizeof(double));
	cudaMemcpyToSymbol(B_unit, &units.B_unit, sizeof(double));
	cudaMemcpyToSymbol(Ne_unit, &units.Ne_unit, sizeof(double));
	cudaMemcpyToSymbol(Thetae_unit, &units.Thetae_unit, sizeof(double));

	// send HARM arrays to device
    cudaMalloc(&d_p, NPRIM*sim.N1*sim.N2*sizeof(double));
    cudaMemcpy(d_p, p, NPRIM*sim.N1*sim.N2*sizeof(double), cudaMemcpyHostToDevice);

    // send photon initial conditions to device
    cudaMalloc(&d_pharr, NPHVARS*nph*sizeof(double));
    cudaMemcpy(d_pharr, pharr, NPHVARS*nph*sizeof(double), cudaMemcpyHostToDevice);

	track_super_photon<<<(nph+TPB-1)/TPB, TPB>>>(d_p, d_pharr, nph);
	//test<<<1, 1>>>();

	// frees device memory
	cudaFree(d_p);
	cudaFree(d_pharr);
}