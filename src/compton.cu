#include "decs.h"
#include"compton.h"
#include "tetrads.h"
#include "curand.h"


__device__ void GPU_scatter_super_photon(struct of_photon *ph, struct of_photon *php,double Ne, double Thetae, double B, double Ucon[NDIM], double Bcon[NDIM], double Gcov[NDIM][NDIM], curandState localState)
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
	GPU_sample_electron_distr_p( K_tetrad, P, Thetae, localState);

	/* given electron momentum P, find the new
	   photon momentum Kp */
	GPU_sample_scattered_photon( K_tetrad, P, K_tetrad_p, localState);


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

__device__ void GPU_sample_scattered_photon(double k[4], double p[4], double kp[4], curandState localState)
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
		k0p = GPU_sample_klein_nishina( ke[0], localState);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = GPU_sample_thomson(localState);
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
	generate_random_direction(&n0x, &n0y, &n0z, localState); /*This currently matches gsl function used*/
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
	phi = 2. * M_PI * curand_uniform_double(&localState);	
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

__device__ double GPU_sample_thomson(curandState localState)
{
	double x1, x2;

	do {

		x1 = 2. * curand_uniform_double(&localState) - 1.;
		x2 = (3. / 4.) * curand_uniform_double(&localState);

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}

__device__ double GPU_sample_klein_nishina(double k0, curandState localState)
{
	double k0pmin, k0pmax, k0p_tent, x1;
	int n = 0;

	/* a low efficiency sampling algorithm, particularly for large k0;
	   limiting efficiency is log(2 k0)/(2 k0) */
	k0pmin = k0 / (1. + 2. * k0);	/* at theta = Pi */
	k0pmax = k0;		/* at theta = 0 */
	do {

		/* tentative value */
		k0p_tent = k0pmin + (k0pmax - k0pmin) * curand_uniform_double(&localState);

		/* rejection sample in box of height = kn(kmin) */
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= curand_uniform_double(&localState);

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


__device__ void GPU_sample_electron_distr_p(double k[4], double p[4], double Thetae, curandState localState)
{
	double beta_e, mu, phi, cphi, sphi, gamma_e, sigma_KN;
	double K, sth, cth, x1, n0dotv0, v0, v1;
	double n0x, n0y, n0z;
	double v0x, v0y, v0z;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	int sample_cnt = 0;
	do {
		GPU_sample_beta_distr( Thetae, &gamma_e, &beta_e, localState);
		mu = GPU_sample_mu_distr( beta_e, localState);
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

		x1 = curand_uniform_double(&localState);

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
	generate_random_direction( &n0x, &n0y, &n0z, localState);
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
	phi = curand_uniform_double(&localState) * 2. * M_PI;	/* orient uniformly */  
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
__device__ void GPU_sample_beta_distr(double Thetae, double *gamma_e, double *beta_e, curandState localState)
{
	double y;

	/* checked */
	y = GPU_sample_y_distr( Thetae, localState);

	/* checked */
	*gamma_e = y * y * Thetae + 1.;
	*beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));

	return;

}
__device__ double GPU_sample_y_distr(double Thetae, curandState localState)
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
		x1 = curand_uniform_double(&localState);
		if (x1 < pi_3) {
			x = chi_square(3, localState);
		} else if (x1 < pi_3 + pi_4) {
			x = chi_square(4, localState);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			x = chi_square(5, localState);
		} else {
			x = chi_square(6, localState);
		}

		/* this translates between defn of distr in
		   Canfield et al. and standard chisq distr */
		y = sqrt(x / 2);

		x2 = curand_uniform_double(&localState);
		num = sqrt(1. + 0.5 * Thetae * y * y);
		den = (1. + y * sqrt(0.5 * Thetae));

		prob = num / den;

	} while (x2 >= prob);
	return (y);
}

__device__ double GPU_sample_mu_distr(double beta_e, curandState localState)
{
	double mu, x1, det;

	x1 = curand_uniform_double(&localState);
	det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * x1;
	if (det < 0.)
		printf("det < 0  %g %g\n\n", beta_e, x1);
	mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}