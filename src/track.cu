
#include "decs.h"
#include "track.h"
#include "radiation.h"
#include "compton.h"
#include "metrics.h"

__device__ void GPU_track_super_photon(struct of_photon *ph , double * d_p, double * d_table_ptr, struct of_photon * scat_ofphoton, int round_scat, int photon_index, int instant_partition, curandState localState)
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
	
	while (!GPU_stop_criterion(ph, localState)) {
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
		

		if (GPU_stop_criterion(ph, localState)){
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

			x1 = -log(curand_uniform_double(&localState));
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
					// GPU_scatter_super_photon(ph, &php, Ne,
					// 		     Thetae, B,
					// 		     Ucon, Bcon,
					// 		     Gcov, localState);
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
						memcpy(&scat_ofphoton[my_local_index], ph, sizeof(struct of_photon));
						scat_ofphoton[my_local_index].w = php.w;
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
