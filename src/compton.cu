/*
 * GPUmonty - /compton.cu
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */
#include "decs.h"
#include"compton.h"
#include "tetrads.h"
#include "curand.h"
#include "kappa_sampler.h"
#include "jnu_mixed.h"

__device__ void scatter_super_photon(
    struct of_photonSOA ph,
    struct of_photonSOA php,
    double Ne,
    double Thetae,
    double B,
    double Ucon[NDIM],
    double Bcon[NDIM],
    double Gcov[NDIM][NDIM],
    curandState *s,
    unsigned long long photon_index)
{
    // Load photon momentum into local array for convenience

    double Kcoord[NDIM];
    Kcoord[0] = ph.K0[photon_index];
    Kcoord[1] = ph.K1[photon_index];
    Kcoord[2] = ph.K2[photon_index];
    Kcoord[3] = ph.K3[photon_index];

#ifndef NDEBUG
    if (isnan(Kcoord[1])) {
        printf("scatter: bad input photon\n");
    }
#endif

    if (Kcoord[0] > 1.e5 || Kcoord[0] < 0. ||
        isnan(Kcoord[0]) || isnan(Kcoord[1]) || isnan(Kcoord[3])) {
        ph.w[photon_index] = 0.0;
        return;
    }

    // Set up tetrad for scattering event

    double Econ[NDIM][NDIM];
    double Ecov[NDIM][NDIM];

    {
        double Bhat[NDIM];

        if (B > 0.0) {
#pragma unroll
            for (int k = 0; k < NDIM; ++k)
                Bhat[k] = Bcon[k] / (B / d_B_unit);
        } else {
            Bhat[0] = 0.0;
            Bhat[1] = 1.0;
            Bhat[2] = 0.0;
            Bhat[3] = 0.0;
        }

        make_tetrad(Ucon, Bhat, Gcov, Econ, Ecov);
    }

    // Transform photon momentum to tetrad frame and check for light cone violations

    double Ktet[NDIM];
    coordinate_to_tetrad(Ecov, Kcoord, Ktet);

    if (Ktet[0] > 1.e5 || Ktet[0] < 0.0 || isnan(Ktet[1])) {
        ph.w[photon_index] = 0.0;
        return;
    }

    // Sample electron momentum and scattered photon in tetrad frame

    double Ktet_p[NDIM];
    {
        double P[NDIM];

        sample_electron_distr_p(Ktet, P, Thetae, s);

        if (isnan(P[1]) || isnan(P[2]) || isnan(P[3])) {
            ph.w[photon_index] = 0.0;
            return;
        }

        sample_scattered_photon(Ktet, P, Ktet_p, s);
    }

    //Back to coordinate frame

    double Kout[NDIM];
    tetrad_to_coordinate(Econ, Ktet_p, Kout);

    if (isnan(Kout[1]) || Kout[0] < 0.0) {
        php.w[photon_index] = 0.0;
        return;
    }

    php.K0[photon_index] = Kout[0];
    php.K1[photon_index] = Kout[1];
    php.K2[photon_index] = Kout[2];
    php.K3[photon_index] = Kout[3];

    // Absorption bookkeeping

    {
        /* flip sign for absorption bookkeeping */
        Ktet_p[0] = -Ktet_p[0];

        double tmpK[NDIM];
        tetrad_to_coordinate(Ecov, Ktet_p, tmpK);

        php.E0[photon_index] = ph.E[photon_index];
        php.E[photon_index] =
            php.E0s[photon_index] = -tmpK[0];

        php.tau_abs[photon_index]   = 0.0;
        php.tau_scatt[photon_index] = 0.0;
        php.nscatt[photon_index]    = ph.nscatt[photon_index] + 1;
    }
}


__device__ void sample_scattered_photon(double k[4], double p[4], double kp[4], curandState * localState)
{
	double ke[4], kpe[4];
	
	/* boost into the electron frame
	   ke == photon momentum in elecron frame */
	boost(k, p, ke);
	
	double k0p, cth;
	if (ke[0] > 1.e-4) {
		k0p = sample_klein_nishina(ke[0], localState);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = sample_thomson(localState);
	}
	
	double sth = sqrt(fabs(1. - cth * cth));

	/* unit vector 1 for scattering coordinate system is
	   oriented along initial photon wavevector */
	double v0x, v0y, v0z;
	{
		// Explicitly compute kemag instead of using ke[0] to ensure that photon
		// is created normalized and doesn't inherit light cone errors from the
		// original superphoton
		double kemag = sqrt(ke[1]*ke[1] + ke[2]*ke[2] + ke[3]*ke[3]);
		v0x = ke[1]/kemag;
		v0y = ke[2]/kemag;
		v0z = ke[3]/kemag;
	}
	
	/* unit vector 2 */
	double v1x, v1y, v1z;
	{
		/* randomly pick zero-angle for scattering coordinate system.
		   There's undoubtedly a better way to do this. */
		double n0x, n0y, n0z;
		generate_random_direction(&n0x, &n0y, &n0z, localState); /*This currently matches gsl function used*/
		double n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

		v1x = n0x - (n0dotv0) * v0x;
		v1y = n0y - (n0dotv0) * v0y;
		v1z = n0z - (n0dotv0) * v0z;
		double v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
		v1x /= v1;
		v1y /= v1;
		v1z /= v1;
	}

	/* now resolve new momentum vector along unit vectors */
	/* create a four-vector $p$ */
	/* solve for orientation of scattered photon */


	p[1] *= -1.;
	p[2] *= -1.;
	p[3] *= -1.;

	{
		
		/* find one more unit vector using cross product;
		this guy is automatically normalized */
		double v2x = v0y * v1z - v0z * v1y;
		double v2y = v0z * v1x - v0x * v1z;
		double v2z = v0x * v1y - v0y * v1x;

		/* find phi for new photon */
		double phi = 2. * M_PI * curand_uniform_double(localState);
		double sphi, cphi;
		sincos(phi, &sphi, &cphi);
		
		double dir1 = cth * v0x + sth * (cphi * v1x + sphi * v2x);
		double dir2 = cth * v0y + sth * (cphi * v1y + sphi * v2y);
		double dir3 = cth * v0z + sth * (cphi * v1z + sphi * v2z);

		kpe[0] = k0p;
		kpe[1] = k0p * dir1;
		kpe[2] = k0p * dir2;
		kpe[3] = k0p * dir3;
	}
	
	/* transform k back to lab frame */
	boost(kpe, p, kp);

	/* quality control */
	if (kp[0] < 0 || isnan(kp[0])) {
		printf("in sample_scattered_photon: %le, %le, p = (%le, %le, %le, %le)\n", kp[0], kpe[0], p[0], p[1], p[2], p[3]);
		// printf("k0p[0] = %g\n", k0p);
		// printf("kp[0], kpe[0]: %g %g\n", kp[0], kpe[0]);
		// printf("kpe: %g %g %g %g\n", kpe[0], kpe[1],
		// 	kpe[2], kpe[3]);
		// printf("k:  %g %g %g %g\n", k[0], k[1], k[2],
		// 	k[3]);
		// printf("ke: %g %g %g %g\n", ke[0], ke[1], ke[2],
		// 	ke[3]);
		// printf("p:   %g %g %g %g\n", p[0], p[1], p[2],
		// 	p[3]);
		// printf("kp:  %g %g %g %g\n", kp[0], kp[1], kp[2],
		// 	kp[3]);
		// printf("phi = %g, cphi = %g, sphi = %g\n", phi, cphi, sphi);
		// printf("cth = %g, sth = %g\n", cth, sth);
	}
	/* done! */
}

__device__ void boost(double v[4], double u[4], double vp[4])
{
	
	double g = u[0];
	double gm1 = g - 1.;
	
	// Compute V and handle small values efficiently
	double g_inv_sq = 1. / (g * g);
	double V = sqrt(fabs(1. - g_inv_sq));
	double gV_inv = 1. / (g * V + SMALL);
	
	// Compute normalized direction components directly in expressions
	double n1 = u[1] * gV_inv;
	double n2 = u[2] * gV_inv;
	double n3 = u[3] * gV_inv;
	
	/* general Lorentz boost into frame u from lab frame */
	vp[0] = u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3];
	
	// Compute cross terms once and reuse
	double n1_gm1 = n1 * gm1;
	double n2_gm1 = n2 * gm1;
	double n3_gm1 = n3 * gm1;
	
	vp[1] = -u[1] * v[0] + (1. + n1 * n1_gm1) * v[1] + 
	        n1_gm1 * n2 * v[2] + n1_gm1 * n3 * v[3];
	        
	vp[2] = -u[2] * v[0] + n2_gm1 * n1 * v[1] + (1. + n2 * n2_gm1) * v[2] +
	        n2_gm1 * n3 * v[3];
	        
	vp[3] = -u[3] * v[0] + n3_gm1 * n1 * v[1] + n3_gm1 * n2 * v[2] +
	        (1. + n3 * n3_gm1) * v[3];
}

__device__ double sample_thomson(curandState * localState)
{
	double x1, x2;

	do {

		x1 = 2. * curand_uniform_double(localState ) - 1.;
		x2 = (3. / 4.) * curand_uniform_double(localState );

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}

__device__ double sample_klein_nishina(double k0, curandState * localState)
{
	double k0pmin, k0p_tent, x1;

	/* a low efficiency sampling algorithm, particularly for large k0;
	   limiting efficiency is log(2 k0)/(2 k0) */
	k0pmin = k0 / (1. + 2. * k0);	/* at theta = Pi */
	do {
		/* tentative value */
		/*Here we use k0pmax as k0 at theta = 0*/
		k0p_tent = k0pmin + (k0 - k0pmin) * curand_uniform_double(localState );

		/* rejection sample in box of height = kn(kmin) */
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= curand_uniform_double(localState );

	} while (x1 >= klein_nishina(k0, k0p_tent));

	return (k0p_tent);
}



__device__  double klein_nishina(const double a, const double ap)
{
    const double inv_a = 1.0 / a;
    const double inv_ap = 1.0 / ap;
    const double ch = 1.0 + inv_a - inv_ap;
    return (a * inv_ap + ap * inv_a - 1.0 + ch * ch) / (a * a);
}


__device__ void sample_electron_distr_p(
    double k[4],
    double p[4],
    double Thetae,
    curandState *s)
{
    // Sampling Electron Velocity

    double beta_e, gamma_e, mu;

    {
        int sample_cnt = 0;

        while (true) {

            sample_beta_distr(Thetae, &gamma_e, &beta_e, s);

            /* sample mu safely */
            {
                double mu_loc =
                    sample_mu_distr(beta_e, curand_uniform_double(s));
                mu = fmin(1.0, fmax(-1.0, mu_loc));
            }

            /* acceptance probability */
            double accept;
            {
                const double K =
                    gamma_e * (1.0 - beta_e * mu) * k[0];

                if (K < 1.e-3) {
                    accept = 1.0 - 2.0 * K;
                } else {
                    const double invK = 1.0 / K;
                    const double t = 1.0 + 2.0 * K;

                    accept =
                        (3.0 * invK * invK * 0.25) *
                        (2.0 +
                         K * K * (1.0 + K) / (t * t) +
                         (K * K - 2.0 * K - 2.0) *
                         0.5 * invK * log(t));
                }
            }

            if (curand_uniform_double(s) < accept)
                break;

            if (++sample_cnt > 10000000) {
                Thetae *= 0.5;
                sample_cnt = 0;
            }
        }
    }

    //Constructing the orthonormal basis

    // v0: direction of photon
    double v0x = k[1], v0y = k[2], v0z = k[3];
    {
        const double invv0 =
            rsqrt(v0x * v0x + v0y * v0y + v0z * v0z);
        v0x *= invv0;
        v0y *= invv0;
        v0z *= invv0;
    }

    // v1: perpendicular direction
    double v1x, v1y, v1z;
    {
        double n0x, n0y, n0z;
        generate_random_direction(&n0x, &n0y, &n0z, s);

        const double n0dotv0 =
            v0x * n0x + v0y * n0y + v0z * n0z;

        v1x = n0x - n0dotv0 * v0x;
        v1y = n0y - n0dotv0 * v0y;
        v1z = n0z - n0dotv0 * v0z;

        const double invv1 =
            rsqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        v1x *= invv1;
        v1y *= invv1;
        v1z *= invv1;
    }

    // Momentum Direction

    double sphi, cphi;
    {
        const double phi =
            curand_uniform_double(s) * 2.0 * M_PI;
        sincos(phi, &sphi, &cphi);
    }

    const double sth = sqrt(1.0 - mu * mu);
    const double gb  = gamma_e * beta_e;

    /* v2 = v0 x v1 (computed inline) */
    const double v2x = v0y * v1z - v0z * v1y;
    const double v2y = v0z * v1x - v0x * v1z;
    const double v2z = v0x * v1y - v0y * v1x;

    /* ===============================
       4. Output four-momentum
       =============================== */

    p[0] = gamma_e;
    p[1] = gb * (mu * v0x + sth * (cphi * v1x + sphi * v2x));
    p[2] = gb * (mu * v0y + sth * (cphi * v1y + sphi * v2y));
    p[3] = gb * (mu * v0z + sth * (cphi * v1z + sphi * v2z));
}



__device__ void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e, curandState * localState)
{
    if (d_thermal_synch) {
        double y = sample_y_distr(Thetae, localState);
        double w = Thetae;
        *gamma_e = y * y * w + 1.;
        *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
        
    } else if (d_kappa_synch) {
        double y = sample_y_distr_nth(Thetae, localState);
        double w = (KAPPA_SYNCH - 3.) / KAPPA_SYNCH * Thetae;
        *gamma_e = y * y * w + 1.;
        *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
        
    } else if (d_powerlaw_synch) {
        double x = curand_uniform_double(localState);
        // Inverse transform sampling for bounded power-law
        *gamma_e = pow( (1. - x) * pow(POWERLAW_GAMMA_MIN, 1. - POWERLAW_SLOPE) + x * pow(POWERLAW_GAMMA_MAX, 1. - POWERLAW_SLOPE), 1. / (1. - POWERLAW_SLOPE) );
        *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
    }
}



#define SQRT_MPI_OVER4 (0.443113462726379) // sqrt(M_PI) / 4
#define INV_SQRT_2 (0.7071067811865475) // 1 / sqrt(2) or sqrt(0.5)
__device__ double sample_y_distr(const double Thetae, curandState * localState)
{
	double pi_3, pi_4, pi_5;
	const double sqrt_thetae = sqrt(Thetae);

	pi_3 =  SQRT_MPI_OVER4;
	pi_4 = INV_SQRT_2 * sqrt_thetae / 2.;
	pi_5 = 3. * SQRT_MPI_OVER4 * Thetae / 2.;

	{
		const double pi_6 = Thetae * INV_SQRT_2 * sqrt_thetae;
		const double S_3 = pi_3 + pi_4 + pi_5 + pi_6;

		pi_3 /= S_3;
		pi_4 /= S_3;
		pi_5 /= S_3;
	}

	const double inv_sqrt2_sqrt_thetae = INV_SQRT_2 * sqrt(Thetae);

	double y, prob;
	do {
		const double x1 = curand_uniform_double(localState );
		
		if (x1 < pi_3) {
			y = sqrt(chi_square(3, localState)/2.);
		} else if (x1 < pi_3 + pi_4) {
			y = sqrt(chi_square(4, localState)/2.);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			y = sqrt(chi_square(5, localState)/2.);
		} else {
			y = sqrt(chi_square(6, localState)/2.);
		}

		/* this translates between defn of distr in
		   Canfield et al. and standard chisq distr */
		const double num = sqrt(1. + 0.5 * Thetae * y * y);
		const double den = (1. + y * inv_sqrt2_sqrt_thetae);

		prob = num / den;

	} while (curand_uniform_double(localState ) >= prob);
	return (y);
}
#undef SQRT_MPI
#undef INV_SQRT_2


__device__ double sample_mu_distr(const double beta_e, double random)
{
	double det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * random;
	if (det < 0.)
		printf("det < 0  %g\n\n", beta_e);
	double mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}