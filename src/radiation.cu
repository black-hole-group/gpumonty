/*
 * GPUmonty - radiation.cu
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
#include "radiation.h"
#include "jnu_mixed.h"
#include "hotcross.h"
#include "utils.h"
#include "model.h"


__device__ double Bnu_inv(const double nu, const double Thetae)
{

	double x;

	x = HPL * nu / (ME * CL * CL * Thetae);

	if (x < 1.e-3){	/* Taylor expand */
		return ((2. * HPL / (CL * CL)) /
			(x / 24. * (24. + x * (12. + x * (4. + x)))));
	}
	else{
		return ((2. * HPL / (CL * CL)) / (exp(x) - 1.));
	}
}

__device__ double get_model_kappa(double X[NDIM]
	#ifndef SPHERE_TEST
	, double * d_p
	#endif
)
{
	#if VARIABLE_KAPPA
		#ifdef SPHERE_TEST
		double sigma, beta;
		get_model_sigma_beta(X, &beta, &sigma);
		#else
		double sigma, beta;
		get_model_sigma_beta(X, d_p, &sigma, &beta);
		#endif
	double kappa = 2.8 + 0.7*pow(sigma,-0.5) + 3.7*pow(sigma,-0.19)*tanh(23.4*pow(sigma,0.26)*beta);
	return fmax(KAPPA_MIN, kappa);  // Beware this clips kappa of NaN -> kappa_min as well
	#else
	return KAPPA_SYNCH;
	#endif
}
__host__ __device__ double get_model_kappa_ijk(const int i, const int j, const int k, const double * d_p)
{
	#if VARIABLE_KAPPA
		#ifdef SPHERE_TEST
			#ifdef __CUDA_ARCH__
				double local_L_unit = d_L_unit;
				double local_tp_over_te = d_tp_over_te;
			#else
				double local_L_unit = L_unit;
				double local_tp_over_te = params.tp_over_te;
			#endif
			double gam = 13./9.;
			double model_ne0 = MODEL_TAU0/SIGMA_THOMSON/SPHERE_RADIUS/local_L_unit;
			double custom_thetae_unit = (MP/ME * (gam - 1.)/(1. + local_tp_over_te));
			double model_B = CL * sqrt(8 * M_PI * (gam - 1.) * (MP + ME)/ BETA0) * sqrt(model_ne0 * THETAE_VALUE)/sqrt(custom_thetae_unit);
			double sigma = (model_B * model_B)/(model_ne0)/((4. * M_PI * CL * (MP + ME)));
			double beta = BETA0;
		#else
			double sigma = d_p[NPRIM_INDEX3D(SIGMA, i,j,k)];
			double beta = d_p[NPRIM_INDEX3D(BETA, i,j,k)];
		#endif
		double kappa = 2.8 + 0.7*pow(sigma,-0.5) + 3.7*pow(sigma,-0.19)*tanh(23.4*pow(sigma,0.26)*beta);
		return fmax(KAPPA_MIN, kappa);  // Beware this clips kappa of NaN -> kappa_min as well
	#else
	return KAPPA_SYNCH;
	#endif
}

__device__ double jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa)
{
	double j;
	double K2 = K2_eval(Thetae);

	j = jnu_total(nu, Ne, Thetae, B, theta, K2, kappa);

	return (j / (nu * nu));
}

/* return Lorentz invariant scattering opacity */
__device__ double alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double kappa, const double * __restrict__ d_table_ptr)
{
	return (nu * kappa_es(nu, Thetae, d_table_ptr, kappa) * Ne * MP);
}

/* return Lorentz invariant absorption opacity */
__device__ double alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa)
{
	if (d_kappa_synch){
		return (anu_synch_kappa(nu, Ne, Thetae, B, theta, kappa));
	}else if (d_powerlaw_synch){
		return (anu_synch_powerlaw(nu, Ne, B, theta));
	}else{
		//Fallback in case only bremsstrahlung, only thermal synchrotron or bremsstrahlung + thermal synchrotron active. 
		return alpha_inv_abs_thermal(nu, Thetae, Ne, B, theta, kappa);
	}
}


__device__ double anu_synch_powerlaw(double nu, double Ne, double B, double theta) {
    double sth = sin(theta);
    if (sth < 1e-150) {
        return 0.0;
    }

    double nuc = (EE * B) / (2.0 * M_PI * ME * CL);
    double X = nu / (nuc * sth);

    double norm_num = pow(3.0, (POWERLAW_SLOPE + 1.0) / 2.0) * (POWERLAW_SLOPE - 1.0);
    double norm_den = 4.0 * (pow(POWERLAW_GAMMA_MIN, 1.0 - POWERLAW_SLOPE) - pow(POWERLAW_GAMMA_MAX, 1.0 - POWERLAW_SLOPE));
    
    double gamma_term1 = tgamma((3.0 * POWERLAW_SLOPE + 2.0) / 12.0);
    double gamma_term2 = tgamma((3.0 * POWERLAW_SLOPE + 22.0) / 12.0);

    double As = (norm_num / norm_den) * gamma_term1 * gamma_term2 * pow(X, -(POWERLAW_SLOPE + 2.0) / 2.0);

    double factor = (Ne * (EE * EE)) / (nu * ME * CL);

    return nu * As * factor;
}

__device__ double anu_synch_kappa(double nu, double Ne, double Thetae, double B, double theta, const double kappa) {


	if (kappa > KAPPA_MAX){
		return alpha_inv_abs_thermal(nu, Thetae, Ne, B, theta, kappa);
	}
    if (Thetae < THETAE_MIN) {
        return 0.0;
    }

    double sth = sin(theta);
    if (sth < 1e-150) {
        return 0.0;
    }

    double w = (kappa - 3.0) / kappa * Thetae;
    double z = -kappa * w;

    if (fabs(z) == 1.0) {
        return 0.0;
    }
	

    double w_kappa = w * kappa; 
    
    double nuc = EE * B / (2.0 * M_PI * ME * CL);
    double nus = nuc * sth * (w_kappa * w_kappa); 
    double X_kappa = nu / nus;

    if (X_kappa > 1e10) {
        return 0.0;
    }

    double hyp2F1;
    double a = kappa - 1.0 / 3.0;
    double b = kappa + 1.0;
    double c = kappa + 2.0 / 3.0;

    if (fabs(z) < 1.0) {
        hyp2F1 = cuda_hyperg_2F1(a, b, c, z);
    } else {
        double inv_one_minus_z = 1.0 / (1.0 - z); 
        hyp2F1 = pow(1.0 - z, -a) * tgamma(c) * tgamma(b - a) /
                     (tgamma(b) * tgamma(c - a)) *
                     cuda_hyperg_2F1(a, c - b, a - b + 1.0, inv_one_minus_z) +
                 pow(1.0 - z, -b) * tgamma(c) * tgamma(a - b) /
                     (tgamma(a) * tgamma(c - b)) *
                     cuda_hyperg_2F1(b, c - a, b - a + 1.0, inv_one_minus_z);
    }

    double k_term = (kappa - 2.0) * (kappa - 1.0) * kappa;
    
    double A_low = pow(X_kappa, -5.0 / 3.0) * pow(3.0, 1.0 / 6.0) * (10.0 / 41.0) *
                   (4.0 * M_PI * M_PI) / pow(w_kappa, 16.0 / 3.0 - kappa) * k_term / (3.0 * kappa - 1.0) * tgamma(5.0 / 3.0) * hyp2F1;

    double A_high = pow(X_kappa, -(3.0 + kappa) / 2.0) * (2.0 * pow(M_PI, 5.0 / 2.0) / 3.0) *
                    (k_term / pow(w_kappa, 5.0)) *
                    (2.0 * tgamma(2.0 + kappa / 2.0) / (2.0 + kappa) - 1.0) *
                    (pow(3.0 / kappa, 19.0 / 4.0) + 0.6);

    double x = pow(-1.75 + 1.6 * kappa, -0.86);
    double A_s = pow((pow(A_low, -x) + pow(A_high, -x)), -1.0 / x);

    double factor = (Ne * EE) / (B * sth);

    return nu * factor * A_s * exp(-nu / NU_CUTOFF);
}

/* return Lorentz invariant absorption opacity for thermal synchrotron */
__device__ double alpha_inv_abs_thermal(const double nu, const double Thetae, const double Ne, const double B, const double theta, const double kappa)
{
	double j, bnu;
	j = jnu_inv(nu, Thetae, Ne, B, theta, kappa);
	bnu = Bnu_inv(nu, Thetae);
	if (j > 0){
		return (j / (bnu + 1.e-100));
	}
	return 0;
}


/* return electron scattering opacity, in cgs */
//#define SCATTERING_OPACITY_CONSTANT (HPL/(ME * CL * CL))
#define SCATTERING_OPACITY_CONSTANT (8.093299734781324e-21)/*h/(m_e c^2)*/
__device__ double kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr, const double kappa)
{
	double Eg;

	/* assume pure hydrogen gas to 
	   convert cross section to opacity */
	Eg = SCATTERING_OPACITY_CONSTANT * nu;
	double result = (total_compton_cross_lkup(Eg, Thetae, kappa, d_table_ptr) / MP);
	if (isnan(result)){
		printf("kappa_es is nan: %le, %le\n", nu, Thetae);
	}
	return result;
}

/* get frequency in fluid frame, in Hz */
__device__ double get_fluid_nu(const double X[NDIM]  , const double K[NDIM]  , const double Ucov[NDIM]  )
{
	double ener, nu;

	/* this is the energy in electron rest-mass units */
	ener = -(K[0] * Ucov[0] +
		 K[1] * Ucov[1] + K[2] * Ucov[2] + K[3] * Ucov[3]);

	nu = ener * ME * CL * CL / HPL;


	if (isnan(ener)) {
		printf("isnan get_fluid_nu, K: %g %g %g %g\n",
			K[0], K[1], K[2], K[3]);
		printf("isnan get_fluid_nu, X: %g %g %g %g\n",
			X[0], X[1], X[2], X[3]);
		printf("isnan get_fluid_nu, U: %g %g %g %g\n",
			Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
	}

	return nu;
}

__device__ double get_bk_angle(const double X[NDIM] , const double K[NDIM]  , const double Ucov[NDIM]  ,
		    const double Bcov[NDIM]  , const double B)
{

	double k, mu;


	if (B == 0.)
		return (M_PI / 2.);

	k = fabs(K[0] * Ucov[0] + K[1] * Ucov[1] + K[2] * Ucov[2] +
		 K[3] * Ucov[3]);

	/* B is in cgs but Bcov is in code units */
	mu = (K[0] * Bcov[0] + K[1] * Bcov[1] + K[2] * Bcov[2] +
	      K[3] * Bcov[3]) / (k * B / d_B_unit);

	if (fabs(mu) > 1.)
		mu /= fabs(mu);

	return (acos(mu));
}