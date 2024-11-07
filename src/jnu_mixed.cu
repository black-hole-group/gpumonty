
/***********************************************************************************
    Copyright 2013 Joshua C. Dolence, Charles F. Gammie, Monika Mo\'scibrodzka,
                   and Po Kin Leung

                        GRMONTY  version 1.0   (released February 1, 2013)

    This file is part of GRMONTY.  GRMONTY v1.0 is a program that calculates the
    emergent spectrum from a model using a Monte Carlo technique.

    This version of GRMONTY is configured to use input files from the HARM code
    available on the same site.   It assumes that the source is a plasma near a
    black hole described by Kerr-Schild coordinates that radiates via thermal 
    synchrotron and inverse compton scattering.
    
    You are morally obligated to cite the following paper in any
    scientific literature that results from use of any part of GRMONTY:

    Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009,
        Astrophysical Journal Supplement, 184, 387

    Further, we strongly encourage you to obtain the latest version of 
    GRMONTY directly from our distribution website:
    http://rainman.astro.illinois.edu/codelib/

    GRMONTY is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    GRMONTY is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GRMONTY; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/


#include "decs.h"

#pragma omp threadprivate(r)
/* 

"mixed" emissivity formula 

interpolates between Petrosian limit and
classical thermal synchrotron limit

good for Thetae > 1

*/

#define CST 1.88774862536	/* 2^{11/12} */
__host__ __device__ double jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta)
{
	double K2, nuc, nus, x, f, j, sth, xp1, xx;

	if (Thetae < THETAE_MIN)
		return 0.;

	K2 = K2_eval(Thetae);

	nuc = EE * B / (2. * M_PI * ME * CL);
	sth = sin(theta);
	// #if(SPHERE_TEST)
	// sth = 1;
	// #endif
	nus = (2. / 9.) * nuc * Thetae * Thetae * sth;
	if (nu > 1.e12 * nus)
		return (0.);
	x = nu / nus;
	xp1 = pow(x, 1. / 3.);
	xx = sqrt(x) + CST * sqrt(xp1);
	f = xx * xx;
	j = (M_SQRT2 * M_PI * EE * EE * Ne * nus / (3. * CL * K2)) * f *
	    exp(-xp1);

	return (j);
}
#undef CST

#define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
__host__ double int_jnu(double Ne, double Thetae, double Bmag, double nu)
{
/* Returns energy per unit time at							*
 * frequency nu in cgs										*/

	double j_fac, K2;
	double F_eval(double Thetae, double B, double nu);
	double K2_eval(double Thetae);


	if (Thetae < THETAE_MIN)
		return 0.;

	K2 = K2_eval(Thetae);
	if (K2 == 0.)
		return 0.;

	j_fac = Ne * Bmag * Thetae * Thetae / K2;
	printf("K2 = %.12e, \n", K2);
	printf("F_eval = %.12e, j_fac = %.12e, JCST = %.12e \n", F_eval(Thetae, Bmag, nu), j_fac, JCST);
	return JCST * j_fac * F_eval(Thetae, Bmag, nu);
}

#undef JCST

#define CST 1.88774862536	/* 2^{11/12} */
double jnu_integrand(double th, void *params)
{

	double K = *(double *) params;
	double sth = sin(th);
	// #if(SPHERE_TEST)
	// sth = 1;
	// #endif
	double x = K / sth;

	if (sth < 1.e-150 || x > 2.e8)
		return 0.;
	//fprintf(stderr, "sth = %le, x = %le, Inside jnu_integrand = %le\n",sth, x, sth *  sth * pow(sqrt(x) + CST * pow(x, 1. / 6.), 2.) * exp(-pow(x, 1. / 3.)) );
	return sth * sth * pow(sqrt(x) + CST * pow(x, 1. / 6.),
			       2.) * exp(-pow(x, 1. / 3.));
}

#undef CST

/* Tables */
//double F[N_ESAMP + 1], K2[N_ESAMP + 1]; //PEDRO EDIT -> F is being declared twice here and in grmonty.c, so I've just commented this and added line below \/
// extern double F[N_ESAMP + 1];
// double K2[N_ESAMP + 1];
// double lK_min, dlK;
// double lT_min;
// extern double dlT; //PEDRO EDIT -> F is being declared twice here and in hotcross.c, so I've just defined this as an extern


//F is the emissiviti's table
__host__ void init_emiss_tables(void)
{

	int k;
	double result, err, K, T;
	gsl_function func;
	gsl_integration_workspace *w;

	func.function = &jnu_integrand;
	func.params = &K;

	double lK_min = log(KMIN);
	double dlK = log(KMAX / KMIN) / (N_ESAMP);

	double lT_min = log(TMIN);
	double dlT = log(TMAX / TMIN) / (N_ESAMP);

	/*  build table for F(K) where F(K) is given by
	   \int_0^\pi ( (K/\sin\theta)^{1/2} + 2^{11/12}(K/\sin\theta)^{1/6})^2 \exp[-(K/\sin\theta)^{1/3}]
	   so that J_{\nu} = const.*F(K)
	 */
	w = gsl_integration_workspace_alloc(1000);
	for (k = 0; k <= N_ESAMP; k++) {
		K = exp(k * dlK + lK_min);
		gsl_integration_qag(&func, 0., M_PI / 2., EPSABS, EPSREL,
				    1000, GSL_INTEG_GAUSS61, w, &result,
				    &err);
		F[k] = log(4 * M_PI * result);
		if(k < 10)
		printf("K = %.12f, result = %.12f, table = %.12f\n", K, result, log(4 * M_PI * result));
	}
	gsl_integration_workspace_free(w);

	/*  build table for quick evaluation of the bessel function K2 for emissivity */
	for (k = 0; k <= N_ESAMP; k++) {
		T = exp(k * dlT + lT_min);
		K2[k] = log(gsl_sf_bessel_Kn(2, 1. / T));

	}

	/* Avoid doing divisions later */
	dlK = 1. / dlK;
	dlT = 1. / dlT;

	fprintf(stderr, "done.\n\n");

	return;
}

/* rapid evaluation of K_2(1/\Thetae) */

__host__ __device__ double K2_eval(double Thetae)
{

	if (Thetae < THETAE_MIN)
		return 0.;
	if (Thetae > TMAX)
		return 2. * Thetae * Thetae;

	return linear_interp_K2(Thetae);
}

#define KFAC	(9*M_PI*ME*CL/EE)
__host__ __device__ double F_eval(double Thetae, double Bmag, double nu)
{

	double K, x;

	K = KFAC * nu / (Bmag * Thetae * Thetae);
	if (K > KMAX) {
		return 0.;
	} else if (K < KMIN) {
		/* use a good approximation */
		x = pow(K, 0.333333333333333333);
		return (x * (37.67503800178 + 2.240274341836 * x));
	} else {
		return linear_interp_F(K);
	}
}




__host__ __device__ double linear_interp_F(double K)
{
	//K = 160.75741686406892;
	double lK_min = log(KMIN);
    double dlK = log(KMAX / KMIN) / (N_ESAMP);
	dlK = 1./dlK;
	double result;
	#ifdef __CUDA_ARCH__
	double * Ftable;
	Ftable = d_F;
	#else
	double * Ftable;
	Ftable = F;
	#endif
	int i;
	double di, lK;
	lK = log(K);
	di = (lK - lK_min) * dlK;
	i = (int) di;
	di = di - i;
	result = exp((1. - di) * Ftable[i] + di * Ftable[i + 1]);
	//result =  exp(tex1D<float>(FTexObj, di + 0.5f));
	//printf("Manual Linear Interp = %le, Tex Linear interp = %le, i = %d, di = %le\n", result,  exp(tex1D<float>(FTexObj, (lK - lK_min) * dlK + 0.5f)), i, (lK - lK_min) * dlK);
	//printf("linear_interp = %le, %le, %lf, %d\n", result, K, di, i);
	return result;
}
__host__ __device__ double linear_interp_K2(double Thetae)
{
	#ifdef __CUDA_ARCH__
	double * bessel_table;
	bessel_table = d_K2;
	#else
	double * bessel_table;
	bessel_table = K2;
	#endif

	int i;
	double di, lT;
	
	lT = log(Thetae);

	di = (lT - d_lT_min) * d_dlT;
	i = (int) di;
	di = di - i;
	//printf("di = %le, i = %d, result = %le\n", di, i, exp((1. - di) * bessel_table[i] + di * bessel_table[i + 1]));
	return exp((1. - di) * bessel_table[i] + di * bessel_table[i + 1]);
}