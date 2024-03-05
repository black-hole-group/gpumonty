#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
bool condition = true;
void function(double * param, double param_2[2], double param_3){
    *param *= 10;
    int n = 0;
    param_2[0] *= 10;
    param_3 *= 11;
    param_2[1] *= 10;
    if(condition){
        n++;
        condition = false;
        function(param, param_2, param_3);        
    }
    for(int i = 0; i < 2; i++){
        fprintf(stderr,"(%d) X[%d] = %lf\n", n, i, param_2[i]);
    } 
    fprintf(stderr, "(%d) param = %le\n", n, *param);
    fprintf(stderr, "(%d) param3 = %le\n", n, param_3);

}

void main(){
    double * E0;
    double X[2];
    double random = 11;
    X[0] = 1;
    X[1] = 2;
    E0 = (double *) malloc(sizeof(double));
    *E0 = 5;
    function(E0, X, random);
    free(E0);
}




__device__ void GPU_track_super_photon(curandStateMtgp32 *state, struct of_photon *ph, double * d_p, struct local_track_var * local_track_vars)
{
	int recursive_index = 0;
	int max_recursions = MAXNSTEP;
	bool starting_recursion = false;
	bool jump_to_ending = false;
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
		printf( "track_super_photon: bad input photon.\n");
		printf(
			"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
			ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
		return;
	}

	dtauK = 2. * M_PI * L_UNIT / (ME * CL * CL / HBAR);

	/* Initialize opacities */
	GPU_gcov_func_hamr(ph->X, Gcov);
	GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
			Bcov, d_p);

	theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
	nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
	alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne);
	alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
	bi = GPU_bias_func(Thetae, ph->w);

	/* Initialize dK/dlam */
	GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
	recursive_round:
		while (!GPU_stop_criterion(state, ph) || jump_to_ending) {
			if(!jump_to_ending){
				/*****************************condition******************************/
				if(starting_recursion){
				starting_recursion = false;
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
					printf( "track_super_photon: bad input photon.\n");
					printf(
						"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
						ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
						ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
					return;
				}

				dtauK = 2. * M_PI * L_UNIT / (ME * CL * CL / HBAR);

				/* Initialize opacities */
				GPU_gcov_func_hamr(ph->X, Gcov);
				GPU_get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
						Bcov, d_p);

				theta = GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
				alpha_scatti = GPU_alpha_inv_scatt(nu, Thetae, Ne);
				alpha_absi = GPU_alpha_inv_abs(nu, Thetae, Ne, B, theta);
				bi = GPU_bias_func(Thetae, ph->w);

				/* Initialize dK/dlam */
				GPU_init_dKdlam(ph->X, ph->K, ph->dKdlam);
				}

				/*****************************************************end of condition******************************************/

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
				//printf("it just evaluated stepsize\n");
				/* step the geodesic */
				GPU_push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s),0);
				//printf("First push photon\n");
				if (GPU_stop_criterion(state,ph))
					break;

				/* allow photon to interact with matter, */
				GPU_gcov_func_hamr(ph->X, Gcov);
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
							//exit(1);
							printf("The function should exit the code!");
							return;
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
							GPU_alpha_inv_scatt(nu, Thetae, Ne);
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
					if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
						if (isnan(php.w) || isinf(php.w)) {
							printf(
								"w isnan in track_super_photon: Ne, bias, ph->w, php.w  %g, %g, %g, %g\n",
								Ne, bias, ph->w, php.w);
						}

						frac = x1 / (bias * dtau_scatt);

						/* Apply absorption until scattering event */
						dtau_abs *= frac;
						if (dtau_abs > 100)
							return;	/* This photon has been absorbed before scattering */

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
						GPU_push_photon(Xi, Ki, dKi, dl * frac, &E0,0);
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
						GPU_gcov_func_hamr(ph->X, Gcov);
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
							/*Modifying condition to true*/
							printf("It got recursive! (%d)\n", recursive_index);
							//printf("Nstep = %d\n", nstep);
							starting_recursion = true;
							/*Saving local variables to a certain index*/
							local_track_vars[recursive_index].bound_flag = bound_flag;
							local_track_vars[recursive_index].dtau_scatt = dtau_scatt;
							local_track_vars[recursive_index].dtau_abs = dtau_abs;
							local_track_vars[recursive_index].dtau = dtau;
							local_track_vars[recursive_index].bi = bi;
							local_track_vars[recursive_index].bf = bf;
							local_track_vars[recursive_index].alpha_scatti = alpha_scatti;
							local_track_vars[recursive_index].alpha_scattf = alpha_scattf;
							local_track_vars[recursive_index].alpha_absi = alpha_absi;
							local_track_vars[recursive_index].alpha_absf = alpha_absf;
							local_track_vars[recursive_index].dl = dl;
							local_track_vars[recursive_index].x1 = x1;
							local_track_vars[recursive_index].nu = nu;
							local_track_vars[recursive_index].Thetae = Thetae;
							local_track_vars[recursive_index].Ne = Ne;
							local_track_vars[recursive_index].B = B;
							local_track_vars[recursive_index].theta = theta;
							local_track_vars[recursive_index].dtauK = dtauK;
							local_track_vars[recursive_index].frac = frac;
							local_track_vars[recursive_index].bias = bias;
							local_track_vars[recursive_index].E0 = E0;
							for (int i = 0; i < NDIM; i++) {
								local_track_vars[recursive_index].Xi[i] = Xi[i];
								local_track_vars[recursive_index].Ki[i] = Ki[i];
								local_track_vars[recursive_index].dKi[i] = dKi[i];
								local_track_vars[recursive_index].Ucon[i] = Ucon[i];
								local_track_vars[recursive_index].Ucov[i] = Ucov[i];
								local_track_vars[recursive_index].Bcon[i] = Bcon[i];
								local_track_vars[recursive_index].Bcov[i] = Bcov[i];
							}
							for (int i = 0; i < NDIM; i++) for (int j= 0; j<NDIM; j++){
								local_track_vars[recursive_index].Gcov[i][j] = Gcov[i][j];
							}

							local_track_vars[recursive_index].nstep = nstep;
							local_track_vars[recursive_index].php = php;
							local_track_vars[recursive_index].ph = ph;
							/*transfer content from php to ph*/
							ph = &php;
							/*Increasing the place in the list*/
							//printf("Recursive_index = %d\n", recursive_index);
							recursive_index++;
							//printf("all the value for ph and php are updated\n");
							//printf("recursive_index = %d\n", recursive_index);

							continue;
							//track_super_photon(&php);
						}

					}
				}
			}else{
				jump_to_ending = false;
				recursive_index--;
				//printf("It's leaving recursion!(%d)\n", recursive_index);
				//printf("Recursive index = %d\n", recursive_index);
				bound_flag = local_track_vars[recursive_index].bound_flag;
				dtau_scatt = local_track_vars[recursive_index].dtau_scatt;
				dtau_abs = local_track_vars[recursive_index].dtau_abs;
				dtau = local_track_vars[recursive_index].dtau;
				bi = local_track_vars[recursive_index].bi;
				bf = local_track_vars[recursive_index].bf;
				alpha_scatti = local_track_vars[recursive_index].alpha_scatti;
				alpha_scattf = local_track_vars[recursive_index].alpha_scattf;
				alpha_absi = local_track_vars[recursive_index].alpha_absi;
				alpha_absf = local_track_vars[recursive_index].alpha_absf;
				dl = local_track_vars[recursive_index].dl;
				x1 = local_track_vars[recursive_index].x1;
				nu = local_track_vars[recursive_index].nu;
				Thetae = local_track_vars[recursive_index].Thetae;
				Ne = local_track_vars[recursive_index].Ne;
				B = local_track_vars[recursive_index].B;
				theta = local_track_vars[recursive_index].theta;
				dtauK = local_track_vars[recursive_index].dtauK;
				frac = local_track_vars[recursive_index].frac;
				bias = local_track_vars[recursive_index].bias;
				E0 = local_track_vars[recursive_index].E0;

				for (int i = 0; i < NDIM; i++) {
					Xi[i] = local_track_vars[recursive_index].Xi[i];
					Ki[i] = local_track_vars[recursive_index].Ki[i];
					dKi[i] = local_track_vars[recursive_index].dKi[i];
					Ucon[i] = local_track_vars[recursive_index].Ucon[i];
					Ucov[i] = local_track_vars[recursive_index].Ucov[i];
					Bcon[i] = local_track_vars[recursive_index].Bcon[i];
					Bcov[i] = local_track_vars[recursive_index].Bcov[i];
				}
				for (int i = 0; i < NDIM; i++) for (int j= 0; j<NDIM; j++){
						Gcov[i][j]= local_track_vars[recursive_index].Gcov[i][j];
				}
				nstep = local_track_vars[recursive_index].nstep;
				ph = local_track_vars[recursive_index].ph;
				php = local_track_vars[recursive_index].php;

				//printf("Nstep = %d\n", nstep);
			}
			if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {
				if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
				theta =
					GPU_get_bk_angle(ph->X, ph->K, Ucov, Bcov,
						B);
				nu = GPU_get_fluid_nu(ph->X, ph->K, Ucov);
				if (nu < 0.) {
					alpha_scatti = alpha_absi = 0.;
				} else {
					alpha_scatti =
						GPU_alpha_inv_scatt(nu, Thetae,
								Ne);
					alpha_absi =
						GPU_alpha_inv_abs(nu, Thetae, Ne,
							B, theta);
				}
				bi = GPU_bias_func(Thetae, ph->w);

				ph->tau_abs += dtau_abs;
				ph->tau_scatt += dtau_scatt;

			} else {
				if (dtau_abs > 100)
					return;	/* This photon has been absorbed */
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
		//printf("Nstep = %d\n", nstep);
		/* signs that something's wrong w/ the integration */
		if (nstep > MAXNSTEP) {
			printf(
				"X1,X2,K1,K2,bias: %g %g %g %g %g\n",
				ph->X[1], ph->X[2], ph->K[1], ph->K[2],
				bias);
			break;
		}
	}
	//printf("It left the while!\n");

	/* accumulate result in spectrum on escape */
	if (GPU_record_criterion(ph) && nstep < MAXNSTEP)
		GPU_record_super_photon(ph);

	if(recursive_index > 0){
		jump_to_ending = true;
		goto recursive_round;
	}
	/* done! */
	return;
}