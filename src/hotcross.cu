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
__global__ void generate_hotcross_kernel(double *d_table_flat, 
                                         int NW_max, int NT_max, int kappa_nsamp,
                                         double lminw, double dlw, double lmint, double dlT,
                                         double kappa_min, double dkappa, double kappa_synch) 
{
    // Map thread coordinates to your i, j, k loop indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to w (0 to NW)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to T (0 to NT)
    int k = blockIdx.z * blockDim.z + threadIdx.z; // Corresponds to kappa (0 to kappa_nsamp-1)

    // Ensure threads outside the exact bounds do nothing
    if (i <= NW_max && j <= NT_max && k < kappa_nsamp) {
        
        #if VARIABLE_KAPPA
        double kappa = kappa_min + k * dkappa;
        #else
        double kappa = kappa_synch;
        #endif

        // Temperature math
        double lT = lmint + j * dlT;
        double temp_val = pow(10.0, lT);
        double norm = getnorm_dNdgammae(temp_val, kappa);

        // Frequency/energy math
        double lw = lminw + i * dlw;
        double w_val = pow(10.0, lw);

        // Compute the cross section
        double value = total_compton_cross_num(w_val, temp_val, norm, kappa);

        // Flatten the [k][i][j] index into a 1D offset
        // Dimensions are: [kappa_nsamp] x [NW_max + 1] x [NT_max + 1]
        int flat_idx = k * ((NW_max + 1) * (NT_max + 1)) + i * (NT_max + 1) + j;
        
        d_table_flat[flat_idx] = log10(value);
    }
}


__host__ void init_hotcross(void)
{
    double dlw = log10(MAXW / MINW) / NW;
    double dlT = log10(MAXT / MINT) / NT;
    double lminw = log10(MINW);
    double lmint = log10(MINT);
    int kappa_nsamp = 1;

    #if VARIABLE_KAPPA
        if (params.kappa_synch) {
            kappa_nsamp = KAPPA_NSAMP;
        }
    #endif

    fprintf(stderr, "making lookup tables for compton cross section for variable kappa on GPU...\n");

    // Calculate total size and allocate 1D arrays on Host and Device
    size_t num_elements = kappa_nsamp * (NW + 1) * (NT + 1);
    size_t bytes = num_elements * sizeof(double);
    
    double *d_table_flat = NULL;
    double *h_table_flat = (double *)malloc(bytes);
    cudaMalloc((void**)&d_table_flat, bytes);

    // Configure the 3D Grid of GPU Threads
    // We use 16x16 blocks for the i and j dimensions, and map k to the z dimension
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((NW + 16) / 16, (NT + 16) / 16, kappa_nsamp);

    fprintf(stderr, "launching GPU kernel... ");
    //sent d_kappa_synch, d_thermal_synch, d_powerlaw_synch to device constant memory
    cudaMemcpyToSymbol(d_kappa_synch, &params.kappa_synch, sizeof(int));
    cudaMemcpyToSymbol(d_thermal_synch, &params.thermal_synch, sizeof(int));
    cudaMemcpyToSymbol(d_powerlaw_synch, &params.powerlaw_synch, sizeof(int));

    generate_hotcross_kernel<<<numBlocks, threadsPerBlock>>>(
        d_table_flat, NW, NT, kappa_nsamp,
        lminw, dlw, lmint, dlT,
        KAPPA_MIN, DKAPPA, KAPPA_SYNCH // Pass constants directly to avoid device symbol lookups
    );

    // Ensure kernel finishes and check for launch errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened array back to the CPU
    cudaMemcpy(h_table_flat, d_table_flat, bytes, cudaMemcpyDeviceToHost);

    // Unpack the 1D array into your standard 3D CPU array & check for NaNs
    for (int k = 0; k < kappa_nsamp; ++k) {
        for (int i = 0; i <= NW; i++) {
            for (int j = 0; j <= NT; j++) {
                int flat_idx = k * ((NW + 1) * (NT + 1)) + i * (NT + 1) + j;
                table[k][i][j] = h_table_flat[flat_idx];

                if (isnan(table[k][i][j])) {
                    double lw = lminw + i * dlw;
                    double lT = lmint + j * dlT;
                    fprintf(stderr, "NaN detected! i=%d j=%d lw=%g lT=%g\n", i, j, lw, lT);
                    exit(1);
                }
            }
        }
    }

    fprintf(stderr, "Done!\n");

    free(h_table_flat);
    cudaFree(d_table_flat);
}

__device__ double total_compton_cross_lkup(double w, double thetae, double kappa, const double * __restrict__ d_table_ptr)
{
    if (w * thetae < 1.e-6){
        return (SIGMA_THOMSON);
    }
    if (thetae < MINT){
        return (hc_klein_nishina(w) * SIGMA_THOMSON);
    }

    if ((w >= MINW && w <= MAXW) && (thetae >= MINT && thetae <= MAXT)) {
        
        double lw = log10(w);
        double lT = log10(thetae);

        double exact_i = (lw - d_lminw) / d_dlw;
        int i = (int) fmax(0.0, fmin(exact_i, NW - 1.0));
        double di = fmax(0.0, fmin(exact_i - i, 1.0));

        double exact_j = (lT - d_lmint) / d_dlT2;
        int j = (int) fmax(0.0, fmin(exact_j, NT - 1.0));
        double dj = fmax(0.0, fmin(exact_j - j, 1.0));

        const int stride_i = NT + 1;
        const int stride_k = (NW + 1) * stride_i;
        
        int idx = j + stride_i * i; 

        #if VARIABLE_KAPPA
            double exact_k = (kappa - KAPPA_MIN) / DKAPPA;
            int k = (int) fmax(0.0, fmin(exact_k, KAPPA_NSAMP - 2.)); 
            double dk = fmax(0.0, fmin(exact_k - k, 1.0));

            // Shift index to the correct kappa slice
            idx += stride_k * k; 

            // Interpolate along J (theta), then I (w) for slice K
            double c00 = d_table_ptr[idx] + dj * (d_table_ptr[idx + 1] - d_table_ptr[idx]);
            double c01 = d_table_ptr[idx + stride_i] + dj * (d_table_ptr[idx + stride_i + 1] - d_table_ptr[idx + stride_i]);
            double val_k0 = c00 + di * (c01 - c00);

            // Shift index to the next kappa slice (K+1)
            idx += stride_k;

            // Interpolate along J, then I for slice K+1
            double c10 = d_table_ptr[idx] + dj * (d_table_ptr[idx + 1] - d_table_ptr[idx]);
            double c11 = d_table_ptr[idx + stride_i] + dj * (d_table_ptr[idx + stride_i + 1] - d_table_ptr[idx + stride_i]);
            double val_k1 = c10 + di * (c11 - c10);

            // Final interpolation along K (kappa)
            double lcross = val_k0 + dk * (val_k1 - val_k0);

        #else
            // If kappa is static, we only do a 2D Bilinear interpolation.
            double c00 = d_table_ptr[idx] + dj * (d_table_ptr[idx + 1] - d_table_ptr[idx]);
            double c01 = d_table_ptr[idx + stride_i] + dj * (d_table_ptr[idx + stride_i + 1] - d_table_ptr[idx + stride_i]);
            double lcross = c00 + di * (c01 - c00);
        #endif

        if (isnan(lcross)) {
            printf("Problem in total_compton_cross_lkup, lcross is nan!\n");
            printf("w: %g, thetae: %g, kappa: %g\n", w, thetae, kappa);
            printf("c00: %g, c01: %g, c10: %g, c11: %g\n", 
                d_table_ptr[idx], d_table_ptr[idx + 1], 
                d_table_ptr[idx + stride_k], d_table_ptr[idx + stride_k + 1]);
        }
        return (pow(10., lcross));
    }
    
    printf("out of bounds: %g %g\n", w, thetae);
    return (total_compton_cross_num(w, thetae, getnorm_dNdgammae(thetae, kappa), kappa));
}



__host__ __device__ double total_compton_cross_num(double w, double thetae, double norm, double kappa)
{
    if (isnan(w)) {
        printf("compton cross isnan: %g %g\n", w, thetae);
        return 0.0;
    }

    if (thetae < MINT && w < MINW)
        return SIGMA_THOMSON;
    if (thetae < MINT)
        return hc_klein_nishina(w) * SIGMA_THOMSON;

    double dgammae = thetae * DGAMMAE;
    double max_gammae = 1. + MAXGAMMA * thetae;

    double cross = 0.0;

    for (double gammae = 1. + 0.5 * dgammae; gammae < max_gammae; gammae += dgammae) {
        
        double f = 0.5 * norm * dNdgammae(thetae, gammae, kappa);
        
        double mue_integral = 0.0;
        
        for (double mue = -1. + 0.5 * DMUE; mue < 1.0; mue += DMUE) {
            mue_integral += boostcross(w, mue, gammae);
        }
        
        cross += mue_integral * f * DMUE * dgammae;
    }

    if (isnan(cross)) {
        printf("Problem in hc_klein_nishina, cross is nan\n");
    }

    return cross * SIGMA_THOMSON;
}

__host__ __device__ double getnorm_dNdgammae(double thetae, double kappa){
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
		return dNdgammae_kappa_norm(thetae, kappa);
	}
	if(is_powerlaw_synch){
		return 1.0; // powerlaw is already normalized by construction
	}
	if(is_thermal_synch){
		return 1.0; // thermal is already normalized by construction
	}
	return 0.0;
}

__host__ __device__ double dNdgammae(double thetae, double gammae, double kappa) {
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
		return dNdgammae_kappa(thetae, gammae, kappa);
	}
	if(is_powerlaw_synch){
		return dNdgammae_powerlaw(thetae, gammae);
	}
	if(is_thermal_synch){
		return dNdgammae_th(thetae, gammae);
	}
	return 0.0;
}


// Transformed Integrand over 'u' where u = sqrt(gamma - 1)
__host__ __device__ double kappa_integrand_u(double u, double thetae, double kappa) {
    double u2 = u * u;
    double gamma = 1.0 + u2;
    double w = (kappa - 3.) / kappa * thetae;

    return 2.0 * u2 * gamma * sqrt(2.0 + u2) *
           pow(1.0 + u2 / (kappa * w), -(kappa + 1.0)) *
           exp(-gamma / GAMMA_MAX);
}


__host__ __device__ double simpsons_rule_u(double a, double b, int N, double thetae, double kappa) {
    double h = (b - a) / N;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 1; i < N; i += 2) {
        sum1 += kappa_integrand_u(a + i * h, thetae, kappa);
    }
    for (int i = 2; i < N - 1; i += 2) {
        sum2 += kappa_integrand_u(a + i * h, thetae, kappa);
    }

    double result = kappa_integrand_u(a, thetae, kappa) + 
                    kappa_integrand_u(b, thetae, kappa) + 
                    4.0 * sum1 + 
                    2.0 * sum2;
    
    return result * h / 3.0;
}

__host__ __device__ double dNdgammae_kappa_norm(double thetae, double kappa) {
    const int N = 500; 
    
    double a = 0.0;
    
    // Original upper bound for beta was log(1. + 1000.*thetae)
    // gamma max was roughly 1 + 1000 * thetae
    // So the new upper bound for u = sqrt(gamma - 1) is:
    double b = sqrt(1000.0 * thetae); 
    
    double integral = simpsons_rule_u(a, b, N, thetae, kappa);

    return 1.0 / integral;
}

__host__ __device__ double dNdgammae_th(double thetae, double gammae)
{
	double K2f;

	if (thetae > 1.e-2) {
        #ifdef __CUDA_ARCH__
            K2f = bessk2(1. / thetae) * exp(1. / thetae);
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

__host__ __device__ double dNdgammae_kappa(double thetae, double gammae, double kappa) {
    double w = (kappa - 3.) / kappa * thetae;
    // This is not yet normalized!
    double dNdgam = gammae * sqrt(gammae * gammae - 1.) *
                    pow((1 + (gammae - 1) / (kappa * w)), -(kappa + 1)) *
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

/*Bessel0 function defined as Numerical Recipes book*/
__host__ __device__ double bessi0(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    }
    else
    {
        y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y *
                                                                                                                                     (-0.2057706e-1 +
                                                                                                                                      y *
                                                                                                                                          (0.2635537e-1 +
                                                                                                                                           y *
                                                                                                                                               (-0.1647633e-1 + y *
                                                                                                                                                                    0.392377e-2))))))));
    }
    return ans;
}
/*Bessel1 function defined as Numerical Recipes book*/
__host__ __device__ double bessi1(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 +
                                                                                     y * (0.301532e-2 + y * 0.32411e-3))))));
    }
    else
    {
        y = 3.75 / ax;
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
        ans *= (exp(ax) / sqrt(ax));
    }
    return xbess < 0.0 ? -ans : ans;
}
/*Modified bessel0 function defined as Numerical Recipes book*/

__host__ __device__ double bessk0(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (-log(xbess / 2.0) * bessi0(xbess)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 +
                                                                                          y * (0.3488590e-1 + y * (0.262698e-2 + y *
                                                                                                                                     (0.10750e-3 +
                                                                                                                                      y *
                                                                                                                                          0.74e-5))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (-0.7832358e-1 +
                                                               y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y *
                                                                                                                               (-0.251540e-2 +
                                                                                                                                y *
                                                                                                                                    0.53208e-3))))));
    }
    return ans;
}
/*Modified bessel1 function defined as Numerical Recipes book*/
__host__ __device__ double bessk1(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (log(xbess / 2.0) * bessi1(xbess)) + (1.0 / xbess) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 +
                                                                                                                    y *
                                                                                                                        (-0.1919402e-1 + y *
                                                                                                                                             (-0.110404e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.4686e-4)))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (0.23498619 + y *
                                                                                (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2 + y *
                                                                                                                                             (0.325614e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.68245e-3)))))));
    }
    return ans;
}
/*Modified bessel2 function defined as Numerical Recipes book*/
__host__ __device__ double bessk2(double xbess)
{
    int n, j;
    double bk, bkm, bkp, tox;
    n = 2;
    tox = 2.0 / xbess;
    bkm = bessk0(xbess);
    bk = bessk1(xbess);
    for (j = 1; j < n; j++)
    {
        bkp = bkm + j * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}