/*
 * GPUmonty - jnu_mixed.cu
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
#include "jnu_mixed.h"

/* 

"mixed" emissivity formula 

interpolates between Petrosian limit and
classical thermal synchrotron limit

good for Thetae > 1

*/

#define CST 1.88774862536	/* 2^{11/12} */

__host__ __device__ double jnu_synch(const double nu, const double Ne, const double Thetae, const double B,
		 const double theta
		#ifdef __CUDA_ARCH__
		, cudaTextureObject_t besselTexObj
		#endif
		)
{

	double K2, nuc, nus, x, f, j, sth, xp1, xx;

	if (Thetae < THETAE_MIN)
		return 0.;

	#ifdef __CUDA_ARCH__
	K2 = K2_eval(Thetae, besselTexObj);
	#else
	K2 = K2_eval(Thetae);
	#endif

	nuc = EE * B / (2. * M_PI * ME * CL);
	sth = sin(theta);
	
	

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
__host__ __device__ double int_jnu(double Ne, double Thetae, double Bmag, double nu)
{
/* Returns energy per unit time at							*
 * frequency nu in cgs										*/

	double j_fac, K2;
	double F_eval(const double Thetae, const double B, const double nu);


	if (Thetae < THETAE_MIN)
		return 0.;
	#ifdef __CUDA_ARCH__
	K2 = K2_eval(Thetae, NULL);
	#else
	K2 = K2_eval(Thetae);
	#endif

	if (K2 == 0.)
		return 0.;

	j_fac = Ne * Bmag * Thetae * Thetae / K2;
	return JCST * j_fac * F_eval(Thetae, Bmag, nu);
}

#undef JCST

#define CST 1.88774862536	/* 2^{11/12} */
double jnu_integrand(double th, void *params)
{

	double K = *(double *) params;
	double sth = sin(th);

	double x = K / sth;

	if (sth < 1.e-150 || x > 2.e8)
		return 0.;
	return sth * sth * pow(sqrt(x) + CST * pow(x, 1. / 6.),
			       2.) * exp(-pow(x, 1. / 3.));
}

#undef CST


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
		//if(k < 10)
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


__host__ __device__ double K2_eval(const double Thetae
#ifdef __CUDA_ARCH__
	,cudaTextureObject_t besselTexObj
#endif
	)
{

	if (Thetae < THETAE_MIN)
		return 0.;
	if (Thetae > TMAX)
		return 2. * Thetae * Thetae;
	#ifdef __CUDA_ARCH__
	return linear_interp_K2(Thetae, besselTexObj);
	#else
	return linear_interp_K2(Thetae);
	#endif
}

__host__ __device__ double F_eval(const double Thetae, const double Bmag, const double nu)
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




__host__ __device__ double linear_interp_F(const double K)
{
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
	return result;
}
__host__ __device__ double linear_interp_K2(const double Thetae
#ifdef __CUDA_ARCH__
	, cudaTextureObject_t besselTexObj
#endif
	)
{
	int i;
	double di, lT;
	double * bessel_table;
	bessel_table = &di;

	lT = log(Thetae);
	di = (lT - d_lT_min) * d_dlT1;

	#ifdef __CUDA_ARCH__
	//return __expf(tex1D<float>(besselTexObj, di + 0.5f));
	bessel_table = d_K2;
	#else
	bessel_table = K2;
	#endif

	i = (int) di;
	di = di - i;
	return exp((1. - di) * bessel_table[i] + di * bessel_table[i + 1]);
}