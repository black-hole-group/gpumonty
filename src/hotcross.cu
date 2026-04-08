/*
 * GPUmonty - hotcross.cu
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
#include "hotcross.h"
#include "jnu_mixed.h"
/* 

   given energy of photon in fluid rest frame w, in units of electron rest mass
   energy, and temperature of plasma, again in electron rest-mass units, return hot
   cross section in cgs.
   
   This has been checked against Wienke's Table 1, with some disagreement at
   the one part in 10^{-3} level, see wienke_table_1 in the subdirectory hotcross.
   It is not clear what this is due to, but Table 1 does appear to have been evaluated
   using Monte Carlo integration (!).

   A better way to do this would be to make a table in w*thetae and w/thetae; most
   	of the variation is accounted for by w*thetae.
   
*/



#include <gsl/gsl_sf_bessel.h>

__host__ void init_hotcross(void)
{
    int i, j;
    double lw, lT;

    double dlw = log10(MAXW / MINW) / NW;
    double dlT = log10(MAXT / MINT) / NT;
    double lminw = log10(MINW);
    double lmint = log10(MINT);

    fprintf(stderr, "making lookup table for compton cross section...\n");
    #pragma omp parallel for private(i,j,lw,lT)
    for (i = 0; i <= NW; i++) {
        for (j = 0; j <= NT; j++) {
            lw = lminw + i * dlw;
            lT = lmint + j * dlT;
            
            double w_val = pow(10., lw);
            double thetae = pow(10., lT);
            double norm = getnorm_dNdgammae(thetae); // Apply the Kappa distribution norm

            table[i][j] = log10(total_compton_cross_num(w_val, thetae, norm));
            
            if (isnan(table[i][j])) {
                fprintf(stderr, "NaN generated at %d %d %g %g\n", i, j, lw, lT);
                exit(0);
            }
        }
    }
    fprintf(stderr, "done.\n\n");

    return;
}


__device__ double total_compton_cross_lkup(double w, double thetae, const double * __restrict__ d_table_ptr)
{
	/* cold/low-energy: just use thomson cross section */
	if (w * thetae < 1.e-6){
		return (SIGMA_THOMSON);
	}

	/* cold, but possible high energy photo n: use klein-nishina */
	if (thetae < MINT){
		return (hc_klein_nishina(w) * SIGMA_THOMSON);
	}


	/* in-bounds for table */
	if ((w > MINW && w < MAXW) && (thetae > MINT && thetae < MAXT)) {
		
		double lw = log10(w);
		double lT = log10(thetae);
		int i = (int) ((lw - d_lminw) / d_dlw);
		int j = (int) ((lT - d_lmint) / d_dlT2);
		double di = (lw - d_lminw) / d_dlw - i;
		double dj = (lT - d_lmint) / d_dlT2 - j;
		double lcross =
		    (1. - di) * (1. - dj) * d_table_ptr[j + (NT+1) * i] + di * (1. -
								dj) *
		    d_table_ptr[j + (NT+1) * (i+1)] + (1. - di) * dj * d_table_ptr[(j+1) + (NT+1) * i] +
		    di * dj * d_table_ptr[(j+1) + (NT+1) * (i+1)];

		//lcross = tex2D<float>(tableTexObj, i, j);
		if (isnan(lcross)) {
			printf("Problem in total_compton_cross_lkup, lcross is nan!\n");	
			// printf("lw = %g. lT =  %g, i =  %d, j =  %d, di =  %g, dj =  %g\n", lw, lT, i,
			// 	j, di, dj);
			// printf("table[i][j] = %le, table[i][j + 1] = %le, table[i +1][j] = %le, table[i+1][j+1] = %le\n", d_table_ptr[j + (NT+1) * i], d_table_ptr[(j+1) + (NT+1) * i], d_table_ptr[j + (NT+1) * (i+1)], d_table_ptr[(j+1) + (NT+1) * (i+1)]);
		}
		return (pow(10., lcross));
	}
	printf("out of bounds: %g %g\n", w, thetae);
	
	return (total_compton_cross_num(w, thetae, getnorm_dNdgammae(thetae)));

}


__host__ __device__ double total_compton_cross_num(double w, double thetae, double norm)
{

	if (isnan(w)) {
		printf("compton cross isnan: %g %g\n", w, thetae);
		return (0.);
	}

	/* check for easy-to-do limits */
	if (thetae < MINT && w < MINW)
		return (SIGMA_THOMSON);
	if (thetae < MINT)
		return (hc_klein_nishina(w) * SIGMA_THOMSON);

	double dgammae = thetae * DGAMMAE;

	/* integrate over mu_e, gamma_e, where mu_e is the cosine of the
	   angle between k and u_e, and the angle k is assumed to lie,
	   wlog, along the z axis */
	double cross = 0.;
	for (double mue = -1. + 0.5 * DMUE; mue < 1.; mue += DMUE)
		for (double gammae = 1. + 0.5 * dgammae;
		    gammae < 1. + MAXGAMMA * thetae; gammae += dgammae) {

			double f = 0.5 * norm * dNdgammae(thetae, gammae);

			cross +=
			    DMUE * dgammae * boostcross(w, mue,
							gammae) * f;
			if (isnan(cross)) {
				printf("Problem in hc_klein_nishina, cross is nan\n");
				// printf("%g %g %g %g %g %g\n", w,
				// 	thetae, mue, gammae,
				// 	dNdgammae(thetae, gammae),
				// 	boostcross(w, mue, gammae));
			}
		}


	return (cross * SIGMA_THOMSON);
}

__host__ __device__ double getnorm_dNdgammae(double thetae){
	#ifdef __CUDA_ARCH__
		const int is_kappa_synch = d_kappa_synch;
		const int is_thermal_synch = d_thermal_synch;
		const int is_powerlaw_synch = d_powerlaw_synch;
	#else
		const int is_kappa_synch = params.kappa_synch;
		const int is_thermal_synch = params.thermal_synch;
		const int is_powerlaw_synch = params.powerlaw_synch;
	#endif

	if(is_kappa_synch){
		return dNdgammae_kappa_norm(thetae);
	}
	if(is_powerlaw_synch){
		return 1.0; // powerlaw is already normalized by construction
	}
	if(is_thermal_synch){
		return 1.0; // thermal is already normalized by construction
	}
	return 0.0;
}

__host__ __device__ double dNdgammae(double thetae, double gammae) {
    #ifdef __CUDA_ARCH__
		const int is_kappa_synch = d_kappa_synch;
		const int is_thermal_synch = d_thermal_synch;
		const int is_powerlaw_synch = d_powerlaw_synch;
	#else
		const int is_kappa_synch = params.kappa_synch;
		const int is_thermal_synch = params.thermal_synch;
		const int is_powerlaw_synch = params.powerlaw_synch;
	#endif

	if(is_kappa_synch){
		return dNdgammae_kappa(thetae, gammae);
	}
	if(is_powerlaw_synch){
		return dNdgammae_powerlaw(thetae, gammae);
	}
	if(is_thermal_synch){
		return dNdgammae_th(thetae, gammae);
	}
	return 0.0;
}

__host__ __device__ double kappa_integrand(double beta, double thetae) {
    double gamma = exp(beta);
    double w = (KAPPA_SYNCH - 3.) / KAPPA_SYNCH * thetae;

    return gamma * gamma * sqrt(gamma * gamma - 1.) *
           pow((1. + (gamma - 1.) / (KAPPA_SYNCH * w)), -(KAPPA_SYNCH + 1.)) *
           exp(-gamma / GAMMA_MAX);
}

// Transformed Integrand over 'u' where u = sqrt(gamma - 1)
__host__ __device__ double kappa_integrand_u(double u, double thetae) {
    double u2 = u * u;
    double gamma = 1.0 + u2;
    double w = (KAPPA_SYNCH - 3.) / KAPPA_SYNCH * thetae;

    return 2.0 * u2 * gamma * sqrt(2.0 + u2) *
           pow(1.0 + u2 / (KAPPA_SYNCH * w), -(KAPPA_SYNCH + 1.0)) *
           exp(-gamma / GAMMA_MAX);
}


__device__ double simpsons_rule_u(double a, double b, int N, double thetae) {
    double h = (b - a) / N;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 1; i < N; i += 2) {
        sum1 += kappa_integrand_u(a + i * h, thetae);
    }
    for (int i = 2; i < N - 1; i += 2) {
        sum2 += kappa_integrand_u(a + i * h, thetae);
    }

    double result = kappa_integrand_u(a, thetae) + 
                    kappa_integrand_u(b, thetae) + 
                    4.0 * sum1 + 
                    2.0 * sum2;
    
    return result * h / 3.0;
}

__device__ double dNdgammae_kappa_norm(double thetae) {
    const int N = 500; 
    
    double a = 0.0;
    
    // Original upper bound for beta was log(1. + 1000.*thetae)
    // gamma max was roughly 1 + 1000 * thetae
    // So the new upper bound for u = sqrt(gamma - 1) is:
    double b = sqrt(1000.0 * thetae); 
    
    double integral = simpsons_rule_u(a, b, N, thetae);

    return 1.0 / integral;
}

__host__ __device__ double dNdgammae_th(double thetae, double gammae)
{
	double K2f;

	if (thetae > 1.e-2) {
        #ifdef __CUDA_ARCH__
            K2f = K2_eval(thetae)  * exp(1. / thetae);
        #else
            K2f = gsl_sf_bessel_Kn(2, 1. / thetae) * exp(1. / thetae);
        #endif
	} else {
		K2f = sqrt(M_PI * thetae / 2.);
	}

	return ((gammae * sqrt(gammae * gammae - 1.) / (thetae * K2f)) *
		exp(-(gammae - 1.) / thetae));
}

// __host__ __device__ double dNdgammae_th(double thetae, double gammae)
// {
// 	double K2f;

// 	if (thetae > 1.e-2) {
// 		K2f = bessk2(1. / thetae) * exp(1. / thetae);
// 	} else {
// 		K2f = sqrt(M_PI * thetae / 2.);
// 	}

// 	return ((gammae * sqrt(gammae * gammae - 1.) / (thetae * K2f)) *
// 		exp(-(gammae - 1.) / thetae));
// }

__host__ __device__ double dNdgammae_kappa(double thetae, double gammae) {
    double w = (KAPPA_SYNCH - 3.) / KAPPA_SYNCH * thetae;
    // This is not yet normalized!
    double dNdgam = gammae * sqrt(gammae * gammae - 1.) *
                    pow((1 + (gammae - 1) / (KAPPA_SYNCH * w)), -(KAPPA_SYNCH + 1)) *
                    exp(-gammae / GAMMA_MAX);
    return dNdgam;
}

__host__ __device__ double dNdgammae_powerlaw(double thetae, double gammae)
{
//    double exp_cutoff = exp(-gammae / GAMMA_MAX);
//    (void)exp_cutoff;

   if (gammae < POWERLAW_GAMMA_MIN || POWERLAW_GAMMA_MAX < gammae) return 0.;

  // note no exponential cutoff. this means we're not using powerlaw_gamma_cut
  // or gamma_max. this choice makes normalization easier and seems consistent
  // with the symphony emissivity formula
  return (POWERLAW_SLOPE -1.) * pow(gammae, -POWERLAW_SLOPE) / ( pow(POWERLAW_GAMMA_MIN, 1-POWERLAW_SLOPE) - pow(POWERLAW_GAMMA_MAX, 1-POWERLAW_SLOPE) );
}

__host__ __device__ double boostcross(double w, double mue, double gammae)
{
	double we, boostcross, v;

	/* energy in electron rest frame */
	v = sqrt(gammae * gammae - 1.) / gammae;
	we = w * gammae * (1. - mue * v);

	boostcross = hc_klein_nishina(we) * (1. - mue * v);

	if (boostcross > 2) {
		printf("w,mue,gammae: %g %g %g\n", w, mue,
			gammae);
		printf("v,we, boostcross: %g %g %g\n", v, we,
			boostcross);
		printf("kn: %g %g %g\n", v, we, boostcross);
	}

	if (isnan(boostcross)) {
		printf("isnan: %g %g %g\n", w, mue, gammae);
		printf("The code should exit, problem in function boostcross\n");
		//exit(0);
	}

	return (boostcross);
}

__host__ __device__ double hc_klein_nishina(double we)
{
	double sigma;

	if (we < 1.e-3)
		return (1. - 2. * we);

	sigma = (3. / 4.) * (2. / (we * we) +
			     (1. / (2. * we) -
			      (1. + we) / (we * we * we)) * log(1. +
								2. * we) +
			     (1. + we) / ((1. + 2. * we) * (1. + 2. * we))
	    );

	return (sigma);

}