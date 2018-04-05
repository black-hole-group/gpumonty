
/*

	main transport subroutine for tracking, absorbing,
	and scattering superphotons

	assumes superphotons do not step out of simulation then back in

*/

#include "decs.h"
#include <string.h>

#define MAXNSTEP	1280000

void track_super_photon(struct of_photon *ph)
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

	// output filename
	char *file_out, *ran_string;

	/* quality control */
	if (isnan(ph->X[0]) ||
	    isnan(ph->X[1]) ||
	    isnan(ph->X[2]) ||
	    isnan(ph->X[3]) ||
	    isnan(ph->K[0]) ||
	    isnan(ph->K[1]) ||
	    isnan(ph->K[2]) || isnan(ph->K[3]) ) {
		fprintf(stderr, "track_super_photon: bad input photon.\n");
		fprintf(stderr,
			"X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
			ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
			ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
		return;
	}


	dtauK = 2. * M_PI * L_unit / (ME * CL * CL / HBAR); // constant C in eq. 16 of Dolence et al. 2009

	// Initialize filename that will hold photon's trajectory
    ran_string=malloc(SIZE_STR*sizeof(char)); 
    file_out=malloc((8+SIZE_STR)*sizeof(char)); // +1 for zero terminator
    rand_string(ran_string,10); // random string
    // Concatenates strings to get full filename
    strcpy(file_out, "photon_");
    strcat(file_out, ran_string);

	dtauK = 2. * M_PI * L_unit / (ME * CL * CL / HBAR);


	/* Initialize opacities */
	gcov_func(ph->X, Gcov);
	get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon,
				 Bcov);
	theta = get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
	nu = get_fluid_nu(ph->X, ph->K, Ucov);
	alpha_scatti = alpha_inv_scatt(nu, Thetae, Ne);
	alpha_absi = alpha_inv_abs(nu, Thetae, Ne, B, theta);
	bi = bias_func(Thetae, ph->w);

	/* Initialize dK/dlam */
	init_dKdlam(ph->X, ph->K, ph->dKdlam);

	// Initialize file that will hold photon trajectory
	FILE *f = fopen(file_out, "w");

	// Geodesic integration happens inside this loop
	while (!stop_criterion(ph)) {

		/* Save initial position/wave vector. */
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
//        fprintf(stderr, "X[3] = %g\n", Xi[3]);

		/* evaluate stepsize */
		dl = stepsize(ph->X, ph->K);

		/* step the geodesic */
		push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s), 0);
		if (stop_criterion(ph))
			break;

		/* allow photon to interact with matter, */
		gcov_func(ph->X, Gcov);
		get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov,
				 Bcon, Bcov);
		if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.) {

			bound_flag = 0;
			if (Ne == 0.)
				bound_flag = 1;
			if (!bound_flag) {
				theta =
				    get_bk_angle(ph->X, ph->K, Ucov, Bcov,
						 B);
				nu = get_fluid_nu(ph->X, ph->K, Ucov);
				if (isnan(nu)) {
					fprintf(stderr,
						"isnan nu: track_super_photon dl,E0 %g %g\n",
						dl, E0);
					fprintf(stderr,
						"Xi, %g %g %g %g\n", Xi[0],
						Xi[1], Xi[2], Xi[3]);
					fprintf(stderr,
						"Ki, %g %g %g %g\n", Ki[0],
						Ki[1], Ki[2], Ki[3]);
					fprintf(stderr,
						"dKi, %g %g %g %g\n",
						dKi[0], dKi[1], dKi[2],
						dKi[3]);
					exit(1);
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
				    alpha_inv_scatt(nu, Thetae, Ne);
				dtau_scatt =
				    0.5 * (alpha_scatti +
					   alpha_scattf) * dtauK * dl; // eq. 23 of Dolence et al. 2009 - GS
				alpha_scatti = alpha_scattf;

				/* absorption optical depth along step */
				alpha_absf =
				    alpha_inv_abs(nu, Thetae, Ne, B,
						  theta);
				dtau_abs =
				    0.5 * (alpha_absi +
					   alpha_absf) * dtauK * dl; // eq. 19 of Dolence et al. 2009 - GS
				alpha_absi = alpha_absf;

				bf = bias_func(Thetae, ph->w);
				bias = 0.5 * (bi + bf);
				bi = bf;
			}

			x1 = -log(monty_rand());
			php.w = ph->w / bias;
			if (bias * dtau_scatt > x1 && php.w > WEIGHT_MIN) {
				if (isnan(php.w) || isinf(php.w)) {
					fprintf(stderr,
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
					ph->w *= exp(-dtau); // eq. 20 of Dolence et al. 2009 - GS

				/* Interpolate position and wave vector to scattering event */
				push_photon(Xi, Ki, dKi, dl * frac, &E0,
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
				get_fluid_params(ph->X, Gcov, &Ne, &Thetae,
						 &B, Ucon, Ucov, Bcon,
						 Bcov);

				if (Ne > 0.) {
					scatter_super_photon(ph, &php, Ne,
							     Thetae, B,
							     Ucon, Bcon,
							     Gcov);
					if (ph->w < 1.e-100) {	/* must have been a problem popping k back onto light cone */
						return;
					}
					track_super_photon(&php);
				}

				theta =
				    get_bk_angle(ph->X, ph->K, Ucov, Bcov,
						 B);
				nu = get_fluid_nu(ph->X, ph->K, Ucov);
				if (nu < 0.) {
					alpha_scatti = alpha_absi = 0.;
				} else {
					alpha_scatti =
					    alpha_inv_scatt(nu, Thetae,
							    Ne);
					alpha_absi =
					    alpha_inv_abs(nu, Thetae, Ne,
							  B, theta);
				}
				bi = bias_func(Thetae, ph->w);

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

		// saves photon worldline here
		fprintf(f, "%E %E %E %E %E %E %E %E\n", ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0], ph->K[1], ph->K[2], ph->K[3]);

		/* signs that something's wrong w/ the integration */
		if (nstep > MAXNSTEP) {
//			fprintf(stderr,
//				"X1,X2,K1,K2,bias: %g %g %g %g %g\n",
//				ph->X[1], ph->X[2], ph->K[1], ph->K[2],
//				bias);
			fprintf(stderr,
				"X1,X2,X3,K1,K2,K3,bias: %g %g %g %g %g %g %g\n",
				ph->X[1], ph->X[2], ph->X[3], ph->K[1], ph->K[2], ph->K[3],
				bias);
			break;
		}

	}

	fclose(f); // closes file with geodesic worldline

	/* accumulate result in spectrum on escape */
	if (record_criterion(ph) && nstep < MAXNSTEP)
		record_super_photon(ph);

	/* done! */
	return;
}
