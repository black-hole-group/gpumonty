
#include "decs.h"
#include "track.h"
#include "radiation.h"
#include "compton.h"
#include "metrics.h"

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

#define dtauK (L_UNIT / (ME * CL * CL / HPL))
__noinline__ __device__ void GPU_track_super_photon(struct of_photonSOA ph, 
	#ifdef DO_NOT_USE_TEXTURE_MEMORY
	 	double * __restrict__ d_p,
	#else
		cudaTextureObject_t d_p,
	#endif
	const double * __restrict__ d_table_ptr, struct of_photonSOA scat_ofphoton, const unsigned long long starting_scattering_index, const int round_scat, const unsigned long long photon_index, curandState *  localState, cudaTextureObject_t besselTexObj)
{
	// Keeping only essential variables at function scope
	double XArray[NDIM] = {ph.X0[photon_index], ph.X1[photon_index], ph.X2[photon_index], ph.X3[photon_index]};
	double KArray[NDIM] = {ph.K0[photon_index], ph.K1[photon_index], ph.K2[photon_index], ph.K3[photon_index]};
	double dKdlamArray[NDIM] = {ph.dKdlam0[photon_index], ph.dKdlam1[photon_index], ph.dKdlam2[photon_index], ph.dKdlam3[photon_index]};
	
	double E0s = ph.E0s[photon_index];
	double tau_abs = 0;
	double tau_scatt = 0;
	int nstep = 0;

	if (ph.w[photon_index] < 1e-100) {
		return;
	}
	
	/* quality control */
	if (isnan(XArray[0]) || isnan(XArray[1]) || isnan(XArray[2]) || isnan(XArray[3]) ||
		isnan(KArray[0]) || isnan(KArray[1]) || isnan(KArray[2]) || isnan(KArray[3]) || 
		ph.w[photon_index] == 0.) {
		printf("track_super_photon: bad input photon.\n");
		printf("X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			XArray[0], XArray[1], XArray[2], XArray[3], KArray[0],
			KArray[1], KArray[2], KArray[3], ph.w[photon_index], ph.nscatt[photon_index]);
		return;
	}
	if (photon_index == 0){
		XArray[0] = 0.;
		XArray[1] = 3.860329427347717e-01;
		XArray[2] = 4.570312500000000e-01;
		XArray[3] = 0.0;
		KArray[0] = 1.649037731505288e-11;
		KArray[1] = -4.752716515150061e-13;
		KArray[2] = 5.029172547659908e-13;
		KArray[3] = 6.840589551248962e-12;
		ph.w[photon_index] = 8.226374526377148e+40;

	}

	/* Initialize opacities and fluid parameters */
	double alpha_scatti, alpha_absi, bi;
	{
		double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
		double Ne, Thetae, B, theta, nu;
		
		gcov_func(XArray, Gcov);
		#ifndef SPHERE_TEST
			GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, d_p);
		#else
			GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);
		#endif
		
		theta = GPU_get_bk_angle(XArray, KArray, Ucov, Bcov, B);
		nu = GPU_get_fluid_nu(XArray, KArray, Ucov);
		alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
		alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta, besselTexObj);
		bi = GPU_bias_func(Thetae, ph.w[photon_index], round_scat);
		// if(photon_index == 0){
		// 	printf("Initial opacities: nu, alpha_scatti, alpha_absi, bi: %.15e %.15e %.15e %.15e\n", nu, alpha_scatti, alpha_absi, bi);
		// 	printf("Ne = %.15e, Thetae = %.15e, B = %.15e, theta = %.15e\n", Ne, Thetae, B, theta);
		// }
	}
	/* Initialize dK/dlam */
	GPU_init_dKdlam(XArray, KArray, dKdlamArray);
	//if(photon_index == 0)
	// printf("dKdlamArray = %.15e, %.15e, %.15e, %15e\n", dKdlamArray[0], dKdlamArray[1], dKdlamArray[2], dKdlamArray[3]);
	while (!GPU_stop_criterion(XArray[1], &(ph.w[photon_index]), localState)) {
		/* Save initial state for this step */
		double Xi[NDIM] = {XArray[0], XArray[1], XArray[2], XArray[3]};
		double Ki[NDIM] = {KArray[0], KArray[1], KArray[2], KArray[3]};
		double dKi[NDIM] = {dKdlamArray[0], dKdlamArray[1], dKdlamArray[2], dKdlamArray[3]};
		double E0 = E0s;

		/* evaluate stepsize and step the geodesic */
		double dl = GPU_stepsize(XArray, KArray);
		GPU_push_photon(XArray, KArray, dKdlamArray, dl, &E0s);
		// if(photon_index == 0){
		// 	printf("dl = %.15e\n", dl);
		// 	if(nstep > 10.)
		// 	return;
		// }
		if (GPU_stop_criterion(XArray[1], &(ph.w[photon_index]), localState)){
			break;
		}

		/* Get fluid parameters at new position */
		double Ne, Thetae, B;
		{
			double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
			gcov_func(XArray, Gcov);
			#ifndef SPHERE_TEST
				GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, d_p);
			#else
				GPU_get_fluid_params(XArray, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);
			#endif

			// if(photon_index == 0){
			// 	printf("Printing Gcov in a matrix format:\n");
			// 	for(int row = 0; row < NDIM; row++){
			// 		for(int col = 0; col < NDIM; col++){
			// 			printf("%.15e ", Gcov[row][col]);
			// 		}
			// 		printf("\n");
			// 	}
			// 	printf("Ucon: %.15e %.15e %.15e %.15e\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3]);
			// 	printf("Ucov: %.15e %.15e %.15e %.15e\n", Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
			// 	printf("Bcon: %.15e %.15e %.15e %.15e\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3]);
			// 	printf("Bcov: %.15e %.15e %.15e %.15e\n", Bcov[0], Bcov[1], Bcov[2], Bcov[3]);
			// 	printf("Ne = %.15e, Thetae = %.15e, B = %.15e\n", Ne, Thetae, B);
			// }

			/* Process interactions with matter */
			if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {
				double theta, nu;
				if (Ne != 0) {
					theta = GPU_get_bk_angle(XArray, KArray, Ucov, Bcov, B);
					nu = GPU_get_fluid_nu(XArray, KArray, Ucov);
					if (isnan(nu)) {
						printf("isnan nu: track_super_photon dl,E0 %g %g\n", dl, E0);
						printf("Xi, %g %g %g %g\n", Xi[0], Xi[1], Xi[2], Xi[3]);
						printf("Ki, %g %g %g %g\n", Ki[0], Ki[1], Ki[2], Ki[3]);
						printf("dKi, %g %g %g %g\n", dKi[0], dKi[1], dKi[2], dKi[3]);
					}
				}

				/* Calculate optical depths */
				double dtau_scatt, dtau_abs, bias;
				if (Ne == 0. || nu < 0.) {
					dtau_scatt = 0.5 * alpha_scatti * dtauK * dl;
					dtau_abs = 0.5 * alpha_absi * dtauK * dl;
					alpha_scatti = alpha_absi = 0.;
					bias = 0.;
					bi = 0.;
					
				} else {
					double alpha_scattf = GPU_alpha_inv_scatt(nu, Thetae, Ne, d_table_ptr);
					dtau_scatt = 0.5 * (alpha_scatti + alpha_scattf) * dtauK * dl;
					alpha_scatti = alpha_scattf;

					double alpha_absf = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta, besselTexObj);
					dtau_abs = 0.5 * (alpha_absi + alpha_absf) * dtauK * dl;
					alpha_absi = alpha_absf;

					double bf = GPU_bias_func(Thetae, ph.w[photon_index], round_scat);
					bias = 0.5 * (bi + bf);
					bi = bf;
				}

				

				/* Test for scattering event */
				double x1 = -log(curand_uniform_double(localState));
				double boost = pow((round_scat + 1), 2);
				boost = 1.;
				double weight_scat = ph.w[photon_index] / bias;

				if (bias * dtau_scatt * boost > x1 && weight_scat > WEIGHT_MIN) {
					/* Scattering event occurs */
					if (isnan(weight_scat) || isinf(weight_scat)) {
						printf("w isnan in track_super_photon: Ne, bias, ph->w, weight_scat  %g, %g, %g, %g\n",
							Ne, bias, ph.w[photon_index], weight_scat);
					}

					double frac = x1 / (bias * dtau_scatt *boost);
					
					/* Apply absorption until scattering event */
					dtau_abs *= frac;	


					if (dtau_abs > 100) {
						return;	/* This photon has been absorbed before scattering */
					}
					dtau_scatt *= frac;
					double dtau = dtau_abs + dtau_scatt;

					/* Update photon weight */
					if (dtau_abs < 1.e-3) {
						ph.w[photon_index] *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
					} else {
						ph.w[photon_index] *= exp(-dtau); 
					}

					/* Interpolate position and wave vector to scattering event */
					GPU_push_photon(Xi, Ki, dKi, dl * frac, &E0);
					XArray[0] = Xi[0]; XArray[1] = Xi[1]; XArray[2] = Xi[2]; XArray[3] = Xi[3];
					KArray[0] = Ki[0]; KArray[1] = Ki[1]; KArray[2] = Ki[2]; KArray[3] = Ki[3];
					dKdlamArray[0] = dKi[0]; dKdlamArray[1] = dKi[1]; dKdlamArray[2] = dKi[2]; dKdlamArray[3] = dKi[3];
					E0s = E0;

					/* Get plasma parameters at scattering location and store scattered photon */
					{
						double Gcov_scat[NDIM][NDIM], Ucon_scat[NDIM], Ucov_scat[NDIM], Bcon_scat[NDIM], Bcov_scat[NDIM];
						double Ne_scat, Thetae_scat, B_scat;
						
						gcov_func(XArray, Gcov_scat);
						#ifndef SPHERE_TEST
							GPU_get_fluid_params(XArray, Gcov_scat, &Ne_scat, &Thetae_scat, &B_scat, Ucon_scat, Ucov_scat, Bcon_scat, Bcov_scat, d_p);
						#else
							GPU_get_fluid_params(XArray, Gcov_scat, &Ne_scat, &Thetae_scat, &B_scat, Ucon_scat, Ucov_scat, Bcon_scat, Bcov_scat);
						#endif
						
						if (Ne_scat > 0.) {
							if (ph.w[photon_index] < 1.e-100) {
								return;
							}
							if (weight_scat > 0) {
								atomicAdd(&d_N_scatt, 1);
								unsigned long long my_local_index = starting_scattering_index + atomicAdd(&d_num_scat_phs[round_scat], 1);

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
							}
						}
						
						/* Update opacities for next step */
						double theta_scat = GPU_get_bk_angle(XArray, KArray, Ucov_scat, Bcov_scat, B_scat);
						double nu_scat = GPU_get_fluid_nu(XArray, KArray, Ucov_scat);
						if (nu_scat < 0.) {
							alpha_scatti = alpha_absi = 0.;
						} else {
							alpha_scatti = GPU_alpha_inv_scatt(nu_scat, Thetae_scat, Ne_scat, d_table_ptr);
							alpha_absi = GPU_alpha_inv_abs(nu_scat, Thetae_scat, Ne_scat, B_scat, theta_scat, besselTexObj);
						}
						bi = GPU_bias_func(Thetae_scat, ph.w[photon_index], round_scat);
					}

					tau_abs += dtau_abs;
					tau_scatt += dtau_scatt;

				} else {
					/* No scattering - just apply absorption */
					if (dtau_abs > 100) {
						return;	/* This photon has been absorbed */
					}
					tau_abs += dtau_abs;
					tau_scatt += dtau_scatt;
					double dtau = dtau_abs + dtau_scatt;
					
					if (dtau < 1.e-3) {
						ph.w[photon_index] *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
					} else {
						ph.w[photon_index] *= exp(-dtau); 
					}
					// if(photon_index == 0)
					// 	printf("tau_abs = %.15e, tau_scatt = %.15e, dtau = %.15e, ph.w = %.15e\n", tau_abs, tau_scatt, dtau, ph.w[photon_index]);

				}
			}
		}

		nstep++;

		/* Check for integration problems */
		if (nstep > MAXNSTEP) {
			printf("Too many steps in track_super_photon, likely stuck. Cancelling this particular photon!\n");
			// printf("X1,X2,K1,K2, nu, bias,: %g, %g, %g, %g, %g, %g\n",
			// 	XArray[1], XArray[2], KArray[1], KArray[2], nu, bias);
			break;
		}
	}
	
	/* Update photon state */
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
	ph.E0s[photon_index] = E0s;
	ph.tau_abs[photon_index] = tau_abs;
	ph.tau_scatt[photon_index] = tau_scatt;
	// if(photon_index == 0){
	// 	printf("It got here!! nstep = %llu\n", nstep);
	// }
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
