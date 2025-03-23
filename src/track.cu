
#include "decs.h"
#include "track.h"
#include "radiation.h"
#include "compton.h"
#include "metrics.h"

#define dtauK (L_UNIT / (ME * CL * CL / HPL))
__device__ void GPU_track_super_photon(struct of_photonSOA ph , cudaTextureObject_t d_p, const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const int round_scat, const unsigned long long photon_index, curandState localState)
{
	double dtau_scatt, dtau_abs, dtau;
	double bi, bf;
	double alpha_scatti, alpha_scattf;
	double alpha_absi, alpha_absf;
	double dl, x1;
	double nu, Thetae, Ne, B, theta;
	double weight_scat;
	double frac;
	double bias = 0.;
	double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
	double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM],
	    Bcov[NDIM];
	int nstep = 0;
	double XArray[NDIM] = {ph.X0[photon_index], ph.X1[photon_index], ph.X2[photon_index], ph.X3[photon_index]};
	double KArray[NDIM] = {ph.K0[photon_index], ph.K1[photon_index], ph.K2[photon_index], ph.K3[photon_index]};
	double dKdlamArray[NDIM] = {ph.dKdlam0[photon_index], ph.dKdlam1[photon_index], ph.dKdlam2[photon_index], ph.dKdlam3[photon_index]};
	double w = ph.w[photon_index];
	double E0s = ph.E0s[photon_index];
	double tau_scatt = 0;
	double tau_abs = 0;

	/* quality control */
	if (isnan(XArray[0]) ||
		isnan(XArray[1]) ||
		isnan(XArray[2]) ||
		isnan(XArray[3]) ||
		isnan(KArray[0]) ||
		isnan(KArray[1]) ||
		isnan(KArray[2]) || isnan(KArray[3]) || w == 0.) {
		printf("track_super_photon: bad input photon.\n");
		printf(
			"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			XArray[0], XArray[1], XArray[2], XArray[3], KArray[0],
			KArray[1], KArray[2], KArray[3], w, ph.nscatt[photon_index]);
		return;
	}
	

	/* Initialize opacities */
	gcov_func(XArray, Gcov);
	GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
			 Bcov, d_p);
	theta = GPU_get_bk_angle(XArray, KArray, Ucov, Bcov, B);
	nu = GPU_get_fluid_nu(XArray, KArray, Ucov);
	alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
	alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
	bi = GPU_bias_func(Thetae, w, round_scat);
	/* Initialize dK/dlam */
	GPU_init_dKdlam(XArray, KArray, dKdlamArray);
	
	while (!GPU_stop_criterion(XArray[1], &(w), localState)) {
		/* Save initial position/wave vector */
		Xi[0] = XArray[0];
		Xi[1] = XArray[1];
		Xi[2] = XArray[2];
		Xi[3] = XArray[3];
		Ki[0] = KArray[0];
		Ki[1] = KArray[1];
		Ki[2] = KArray[2];
		Ki[3] = KArray[3];
		dKi[0] = dKdlamArray[0];
		dKi[1] = dKdlamArray[1];
		dKi[2] = dKdlamArray[2];
		dKi[3] = dKdlamArray[3];
		E0 = E0s;

		/* evaluate stepsize */
		dl = GPU_stepsize(XArray, KArray);

		/* step the geodesic */
		GPU_push_photon(XArray, KArray, dKdlamArray, dl, &(E0s));
		

		if (GPU_stop_criterion(XArray[1], &(w), localState)){
			break;
		}

		/* allow photon to interact with matter, */
		gcov_func(XArray, Gcov);
		GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov,
				 Bcon, Bcov, d_p);

		
		
		if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {

			if (Ne != 0) {
				theta =
				    GPU_get_bk_angle(XArray, KArray, Ucov, Bcov,
						 B);
				nu = GPU_get_fluid_nu(XArray, KArray, Ucov);
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
			if (Ne == 0. || nu < 0.) {
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

				bf = GPU_bias_func(Thetae,w, round_scat);
				bias = 0.5 * (bi + bf);
				bi = bf;

			}

			x1 = -log(curand_uniform_double(&localState));
			weight_scat = w / bias;
			//if(0){
			if (bias * dtau_scatt > x1 && weight_scat > WEIGHT_MIN) {
				if (isnan(weight_scat) || isinf(weight_scat)) {
					printf(
						"w isnan in track_super_photon: Ne, bias, ph->w, weight_scat  %g, %g, %g, %g\n",
						Ne, bias, w, weight_scat);
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
						w *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
					}
					else{
						w *= exp(-dtau); 
					}
				#endif
				/* Interpolate position and wave vector to scattering event */

				GPU_push_photon(Xi, Ki, dKi, dl * frac, &E0);
				XArray[0] = Xi[0];
				XArray[1] = Xi[1];
				XArray[2] = Xi[2];
				XArray[3] = Xi[3];
				KArray[0] = Ki[0];
				KArray[1] = Ki[1];
				KArray[2] = Ki[2];
				KArray[3] = Ki[3];
				dKdlamArray[0] = dKi[0];
				dKdlamArray[1] = dKi[1];
				dKdlamArray[2] = dKi[2];
				dKdlamArray[3] = dKi[3];
				E0s = E0;
				

				/* Get plasma parameters at new position */
				gcov_func(XArray, Gcov);
				GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae,
						 &B, Ucon, Ucov, Bcon,
						 Bcov, d_p);
				if (Ne > 0.) {
					// GPU_scatter_super_photon(ph, &php, Ne,
					// 		     Thetae, B,
					// 		     Ucon, Bcon,
					// 		     Gcov, localState);
					if (w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
						return;
					}
					if(weight_scat > 0){
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

						scat_ofphoton.X0[my_local_index] = XArray[0];
						scat_ofphoton.X1[my_local_index] = XArray[1];
						scat_ofphoton.X2[my_local_index] = XArray[2];
						scat_ofphoton.X3[my_local_index] = XArray[3];
						scat_ofphoton.K0[my_local_index] = KArray[0];
						scat_ofphoton.K1[my_local_index] = KArray[1];
						scat_ofphoton.K2[my_local_index] = KArray[2];
						scat_ofphoton.K3[my_local_index] = KArray[3];
						scat_ofphoton.dKdlam0[my_local_index] = dKdlamArray[0];
						scat_ofphoton.dKdlam1[my_local_index] = dKdlamArray[1];
						scat_ofphoton.dKdlam2[my_local_index] = dKdlamArray[2];
						scat_ofphoton.dKdlam3[my_local_index] = dKdlamArray[3];
						scat_ofphoton.w[my_local_index] = weight_scat;
						scat_ofphoton.E0[my_local_index] = E0;
						scat_ofphoton.X1i[my_local_index] = ph.X1i[photon_index];
						scat_ofphoton.X2i[my_local_index] = ph.X2i[photon_index];
						scat_ofphoton.tau_abs[my_local_index] = tau_abs;
						scat_ofphoton.tau_scatt[my_local_index] = tau_scatt;
						scat_ofphoton.E[my_local_index] = ph.E[photon_index];
						scat_ofphoton.nscatt[my_local_index] = ph.nscatt[photon_index];

						if(scat_ofphoton.w[my_local_index] != weight_scat){
							printf("In GPU_track_super_photon, both weights should be the same! (%le, %le), %d\n", scat_ofphoton.w[my_local_index], weight_scat, my_local_index);
						}
					}
				}
				theta =
				    GPU_get_bk_angle(XArray, KArray, Ucov, Bcov,
						 B);
				nu = GPU_get_fluid_nu(XArray, KArray, Ucov);
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
				bi = GPU_bias_func(Thetae, ph.w[photon_index], round_scat);

				tau_abs += dtau_abs;
				tau_scatt += dtau_scatt;

			} else {
				if (dtau_abs > 100){
					return;	/* This photon has been absorbed */
				}
				tau_abs += dtau_abs;
				tau_scatt += dtau_scatt;
				dtau = dtau_abs + dtau_scatt;

				//Do not include absorption in the scattering test
				#ifndef SCATTERING_TEST
					if (dtau < 1.e-3){
							w *= (1. -dtau / 24. * (24. -dtau * (12. - dtau *(4. -dtau)))); //taylor expansion
					}else{
							w *= exp(-dtau); 
					}
				#endif
			}
		}

		nstep++;

		/* signs that something's wrong w/ the integration */
		if (nstep > MAXNSTEP) {
			printf(
				"X1,X2,K1,K2, nu, bias,: %g, %g, %g, %g, %g, %g\n",
				XArray[1], XArray[2], KArray[1], KArray[2], nu, bias);
			break;
		}
	}
	ph.X0[photon_index] = XArray[0];
	ph.X1[photon_index] = XArray[1];
	ph.X2[photon_index] = XArray[2];
	ph.X3[photon_index] = XArray[3];
	ph.K0[photon_index] = KArray[0];
	ph.K1[photon_index] = KArray[1];
	ph.K2[photon_index] = KArray[2];
	ph.K3[photon_index] = KArray[3];
	ph.dKdlam0[photon_index] = dKdlamArray[0];
	ph.dKdlam1[photon_index] = dKdlamArray[1];
	ph.dKdlam2[photon_index] = dKdlamArray[2];
	ph.dKdlam3[photon_index] = dKdlamArray[3];
	ph.w[photon_index] = w;
	ph.E0s[photon_index] = E0s;
	ph.tau_abs[photon_index] = tau_abs;
	ph.tau_scatt[photon_index] = tau_scatt;

	/* done! */
	return;
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

__device__ double GPU_stepsize(const double X[NDIM], const double K[NDIM])
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



// // //This one below is from gpu_monty
__device__ void GPU_push_photon(double X[NDIM], double Kcon[NDIM], double dKcon[NDIM], const double dl,
	double *E0)
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
					dKcon[k] = fma(-2.0, 
						(fma(Kcont[0], 
							fma(lconn[k][0][1], Kcont[1], 
								fma(lconn[k][0][2], Kcont[2], 
									lconn[k][0][3] * Kcont[3])),
							fma(Kcont[1], 
								fma(lconn[k][1][2], Kcont[2], 
									lconn[k][1][3] * Kcont[3]),
								lconn[k][2][3] * Kcont[2] * Kcont[3]))),
						-1.0 * (fma(lconn[k][0][0], Kcont[0] * Kcont[0],
								fma(lconn[k][1][1], Kcont[1] * Kcont[1],
									fma(lconn[k][2][2], Kcont[2] * Kcont[2],
										lconn[k][3][3] * Kcont[3] * Kcont[3])))));
                        K[k] = fma(dl_2, dKcon[k], Kcon[k]);
                        err += fabs((Kcont[k] - K[k]) / (K[k] + SMALL));
                }
        } while ((err > ETOL || isinf(err) || isnan(err)) && iter < MAX_ITER);
        FAST_CPY(K, Kcon);

		gcov_func(X, Gcov);
        *E0 = -(Kcon[0] * Gcov[0][0] + Kcon[1] * Gcov[0][1] +
               Kcon[2] * Gcov[0][2] + Kcon[3] * Gcov[0][3]);

		/* done! */
}
