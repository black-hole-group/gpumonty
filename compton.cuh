/*

Routines for treating Compton scattering via Monte Carlo.

Uses the cuRAND random number generator.
The choice of generator can be changed in init_monty_rand;
now set to Mersenne twister.

Sampling procedures for electron distribution is based on
Canfield, Howard, and Liang, 1987, ApJ 323, 565.

*/

//#include "decs.h"
//#pragma omp threadprivate(r)

/* return pseudo-random value between 0 and 1 */
__device__
double d_monty_rand(curandState *state)
{
	return (curand_uniform_double(state));	
	//return (gsl_rng_uniform(r));
}




/*

Lorentz boost vector v into frame given by four-velocity u.
Result goes out in vp.
Assumes all four-velocities are given in orthonormal coordinates.

*/
__device__
void boost(double v[4], double u[4], double vp[4])
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



/* return a cos(theta) consistent w/ Thomson
   differential cross section */

/* uses simple rejection scheme */
__device__
double sample_thomson(curandState *state)
{
	double x1, x2;

	do {

		x1 = 2. * d_monty_rand(state) - 1.;
		x2 = (3. / 4.) * d_monty_rand(state);

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}


/*  

   differential cross section for scattering from 
   frequency a -> frequency ap.  Frequencies are
   in units of m_e.  Unnormalized!
   
*/
__device__
double klein_nishina(double a, double ap)
{
	double ch, kn;

	ch = 1. + 1. / a - 1. / ap;
	kn = (a / ap + ap / a - 1. + ch * ch) / (a * a);

	return (kn);
}

/*

sample Klein-Nishina differential cross section.

This routine is inefficient; it needs improvement.

*/
__device__
double sample_klein_nishina(double k0, curandState *state)
{
	double k0pmin, k0pmax, k0p_tent, x1;
	int n = 0;

	/* a low efficiency sampling algorithm, particularly for large k0;
	   limiting efficiency is log(2 k0)/(2 k0) */
	k0pmin = k0 / (1. + 2. * k0);	/* at theta = Pi */
	k0pmax = k0;		/* at theta = 0 */
	do {

		/* tentative value */
		k0p_tent = k0pmin + (k0pmax - k0pmin) * d_monty_rand(state);

		/* rejection sample in box of height = kn(kmin) */
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= d_monty_rand(state);

		n++;

	} while (x1 >= klein_nishina(k0, k0p_tent));

	return (k0p_tent);
}





/*
   given photon w/ wavevector $k$ colliding w/ electron with
   momentum $p$, ($p$ is actually the four-velocity) 
   find new wavevector $kp$ 
*/
__device__
void sample_scattered_photon(double k[4], double p[4], double kp[4], curandState *state)
{
	double ke[4], kpe[4];
	double k0p;
	double n0x, n0y, n0z, n0dotv0, v0x, v0y, v0z, v1x, v1y, v1z, v2x,
	    v2y, v2z, v1, dir1, dir2, dir3;
	double cth, sth, phi, cphi, sphi;
	void sincos(double x, double *sin, double *cos);

	/* boost into the electron frame
	   ke == photon momentum in elecron frame */

	boost(k, p, ke);
	if (ke[0] > 1.e-4) {
		k0p = sample_klein_nishina(ke[0], state);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = sample_thomson(state);
	}
	sth = sqrt(fabs(1. - cth * cth));

	/* unit vector 1 for scattering coordinate system is
	   oriented along initial photon wavevector */
	v0x = ke[1] / ke[0];
	v0y = ke[2] / ke[0];
	v0z = ke[3] / ke[0];

	/* randomly pick zero-angle for scattering coordinate system.
	   There's undoubtedly a better way to do this. */
	gsl_ran_dir_3d(r, &n0x, &n0y, &n0z);
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
	phi = 2. * M_PI * d_monty_rand(state);
	sincos(phi, &sphi, &cphi);

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
	boost(kpe, p, kp);

	/* quality control */
	if (kp[0] < 0 || isnan(kp[0])) {
		fprintf(stderr, "in sample_scattered_photon:\n");
		fprintf(stderr, "kp[0], kpe[0]: %g %g\n", kp[0], kpe[0]);
		fprintf(stderr, "kpe: %g %g %g %g\n", kpe[0], kpe[1],
			kpe[2], kpe[3]);
		fprintf(stderr, "k:  %g %g %g %g\n", k[0], k[1], k[2],
			k[3]);
		fprintf(stderr, "ke: %g %g %g %g\n", ke[0], ke[1], ke[2],
			ke[3]);
		fprintf(stderr, "p:   %g %g %g %g\n", p[0], p[1], p[2],
			p[3]);
		fprintf(stderr, "kp:  %g %g %g %g\n", kp[0], kp[1], kp[2],
			kp[3]);
	}

	/* done! */
}


/* 

   sample y, which is the temperature-normalized
   kinetic energy.
   Uses procedure outlined in Canfield et al. 1987,
   p. 572 et seq. 
   
*/
__device__
double sample_y_distr(double Thetae, curandState *state)
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
		x1 = d_monty_rand(state);

		if (x1 < pi_3) {
			x = gsl_ran_chisq(r, 3);
		} else if (x1 < pi_3 + pi_4) {
			x = gsl_ran_chisq(r, 4);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			x = gsl_ran_chisq(r, 5);
		} else {
			x = gsl_ran_chisq(r, 6);
		}

		/* this translates between defn of distr in
		   Canfield et al. and standard chisq distr */
		y = sqrt(x / 2);

		x2 = d_monty_rand(state);
		num = sqrt(1. + 0.5 * Thetae * y * y);
		den = (1. + y * sqrt(0.5 * Thetae));

		prob = num / den;

	} while (x2 >= prob);

	return (y);
}


/* 
   sample dimensionless speed of electron
   from relativistic maxwellian 

   checked. 
   
*/
__device__
void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e, curandState *state)
{
	double y;

	/* checked */
	y = sample_y_distr(Thetae, state);

	/* checked */
	*gamma_e = y * y * Thetae + 1.;
	*beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));

	return;

}



__device__
double sample_mu_distr(double beta_e, curandState state)
{
	double mu, x1, det;

	x1 = d_monty_rand(state);
	det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * x1;
	if (det < 0.)
		fprintf(stderr, "det < 0  %g %g\n\n", beta_e, x1);
	mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}



/* 

	sample electron distribution to find which electron was
	scattered.

*/
__device__
void sample_electron_distr_p(double k[4], double p[4], double Thetae, curandState *state)
{
	double beta_e, mu, phi, cphi, sphi, gamma_e, sigma_KN;
	double K, sth, cth, x1, n0dotv0, v0, v1;
	double n0x, n0y, n0z;
	double v0x, v0y, v0z;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	int sample_cnt = 0;
	void sincos(double x, double *sin, double *cos);

	do {
		sample_beta_distr(Thetae, &gamma_e, &beta_e, state);
		mu = sample_mu_distr(beta_e, state);
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

		x1 = d_monty_rand(state);

		sample_cnt++;

		if (sample_cnt > 10000000) {
			fprintf(stderr,
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
	gsl_ran_dir_3d(r, &n0x, &n0y, &n0z);
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
	phi = d_monty_rand(state) * 2. * M_PI;	/* orient uniformly */
	sincos(phi, &sphi, &cphi);
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
		fprintf(stderr, "betae error: %g %g %g %g\n",
			p[0], p[1], p[2], p[3]);
	}

	return;
}
