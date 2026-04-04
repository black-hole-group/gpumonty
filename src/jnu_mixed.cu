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
#include "utils.h"


__device__ double jnu_total(const double nu, const double Ne, const double Thetae, const double B, const double theta, const double K2){
	double j = 0;


	if(d_thermal_synch)
		j += jnu_synch(nu, Ne, Thetae, B, theta,K2 );
	if(d_kappa_synch)
		j += jnu_synch_nonthermal_kappa(nu, Ne, Thetae, B, theta);
	if(d_powerlaw_synch)
		j += jnu_synch_nonthermal_powerlaw(nu, Ne, Thetae, B, theta);
	if(d_bremsstrahlung)
		j += jnu_bremss(nu, Ne, Thetae);

	
	
	return j;
}

#define CST 1.88774862536	/* 2^{11/12} */
#define SYNCH_FAC (1.139685508680628e-29) /* √2·π·e²/(3·c) [CGS] */
#define NUC_FAC (2.799248729308765e+06) /* e/(2π·mₑ·c) [Hz/G] (cyclotron freq per unit B) */
__host__ __device__ double jnu_synch(const double nu, const double Ne, const double Thetae, const double B,
		 const double theta, const double K2)
{

	double nuc, nus, x, f, j, sth, xp1, xx;

	if (Thetae < THETAE_MIN)
		return 0.;

	

	nuc = NUC_FAC * B;
	sth = sin(theta);
	
	

	nus = (2. / 9.) * nuc * Thetae * Thetae * sth;
	if (nu > 1.e12 * nus)
		return (0.);
	x = nu / nus;
	xp1 = pow(x, 1. / 3.);
	xx = sqrt(x) + CST * sqrt(xp1);
	f = xx * xx;
	j = (SYNCH_FAC * Ne * nus / (K2)) * f *
	    exp(-xp1);

	return (j);
}
#undef CST

//
/* (8/3)·(2π/3)^½ · e⁶/(mₑc³) · (kʙmₑ)^(-½) · g_ff [CGS], g_ff assumed to be 1.2 */
#define BREMS_FAC (6.533236526124812e-39)
__host__ __device__  double jnu_bremss(const double nu, const double Ne, const double Thetae){
	if (Thetae < THETAE_MIN) 
		return 0.;

	double Te = Thetae * ME * CL * CL / KBOL;
	double x = HPL*nu/(KBOL*Te);
	double efac, jv;

	if (x < 1.e-3) {
		efac = (24. - 24.*x + 12.*x*x - 4.*x*x*x + x*x*x*x) / 24.;
	} else {
		efac = exp(-x);
	}

	//Method from Rybicki & Lightman, ultimately from Novikov & Thorne
	double rel = (1. + 4.4e-10*Te);

	
	//rsqrt(x) is 1/sqrt(x)
	jv = BREMS_FAC * rsqrt(Te) * Ne*Ne * efac*rel;
	return jv;
}
#undef BREMS_FAC

__host__ __device__ double int_jnu_total(const double Ne, const double Thetae, const double Bmag, const double nu, const double K2)
{
	#ifdef __CUDA_ARCH__
		int is_thermal_synch = d_thermal_synch;
		int is_bremsstrahlung = d_bremsstrahlung;
		int is_kappa_synch = d_kappa_synch;
		int is_powerlaw_synch = d_powerlaw_synch;
	#else
		int is_thermal_synch = params.thermal_synch;
		int is_bremsstrahlung = params.bremsstrahlung;
		int is_kappa_synch = params.kappa_synch;
		int is_powerlaw_synch = params.powerlaw_synch;
	#endif
	double intj = 0;

	if(is_thermal_synch)
		intj += int_jnu_thermal_synch(Ne, Thetae, Bmag, nu, K2);
	if(is_kappa_synch || is_powerlaw_synch)
		intj += int_jnu_nth(Ne, Thetae, Bmag, nu);
	if(is_bremsstrahlung)
		intj += int_jnu_bremss(Ne, Thetae, nu);
	
	return intj;
}

//#define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
#define JCST (7.089473804413026e-24) /* √2·e³/(27·mₑ·c²) [CGS] */
__host__ __device__ double int_jnu_thermal_synch(double Ne, double Thetae, double Bmag, double nu, double K2)
{
/* Returns energy per unit time at							*
 * frequency nu in cgs										*/

	double j_fac;
	double F_eval(const double Thetae, const double B, const double nu, int ACCZONE);


	if (Thetae < THETAE_MIN)
		return 0.;



	if (K2 == 0.)
		return 0.;

	j_fac = Ne * Bmag * Thetae * Thetae / K2;
	return JCST * j_fac * F_eval(Thetae, Bmag, nu, 0);
}

#undef JCST

__host__ __device__ double int_jnu_bremss(const double Ne, const double Thetae, const double nu)
{
	return 4 * M_PI * jnu_bremss(nu, Ne, Thetae);
}

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


__host__ __device__ double K2_eval(const double Thetae)
{

	if (Thetae < THETAE_MIN)
		return 0.;
	if (Thetae > TMAX)
		return 2. * Thetae * Thetae;

	return linear_interp_K2(Thetae);
}



__host__ __device__ double F_eval_th(const double Thetae, const double Bmag, const double nu)
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
		return linear_interp_F_th(K);
	}
}


__host__ __device__ double F_eval(double Thetae, double Bmag, double nu, int ACCZONE)
{
	#ifdef __CUDA_ARCH__
		int is_thermal_synch = d_thermal_synch;
		int is_kappa_synch = d_kappa_synch;
		int is_powerlaw_synch = d_powerlaw_synch;
	#else
		int is_thermal_synch = params.thermal_synch;
		int is_kappa_synch = params.kappa_synch;
		int is_powerlaw_synch = params.powerlaw_synch;
	#endif

	double F_eval;
	if (is_thermal_synch) {
		F_eval = F_eval_th(Thetae, Bmag, nu);
	} else if (is_kappa_synch) {
		F_eval = F_eval_kappa(Thetae, Bmag, nu);
	} else if (is_powerlaw_synch) {
		F_eval = F_eval_powerlaw(Thetae, Bmag, nu);
	}
	
	return F_eval;
}



__host__ __device__ double linear_interp_F_th(const double K)
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
__host__ __device__ double linear_interp_K2(const double Thetae)
{
	int i;
	double di, lT;
	double * bessel_table;
	bessel_table = &di;

	lT = log(Thetae);
	di = (lT - d_lT_min) * d_dlT1;

	#ifdef __CUDA_ARCH__
	bessel_table = d_K2;
	#else
	bessel_table = K2;
	#endif

	i = (int) di;
	di = di - i;
	return exp((1. - di) * bessel_table[i] + di * bessel_table[i + 1]);
}




//Nonthermal part

__device__ double hypergeom_eval(double X) {
    double a = 1.;
    double b = - KAPPA_SYNCH - 1. / 2.;
    double c = 1. / 2.;

    if (fabs(X) > 1) {
        double z = -X;
        double hyp2F1 = pow(1. - z, -a) * cuda_sf_gamma(c) * cuda_sf_gamma(b - a) /
                     (cuda_sf_gamma(b) * cuda_sf_gamma(c - a)) *
                     cuda_hyperg_2F1(a, c - b, a - b + 1., 1. / (1. - z)) +
                 pow(1. - z, -b) * cuda_sf_gamma(c) * cuda_sf_gamma(a - b) /
                     (cuda_sf_gamma(a) * cuda_sf_gamma(c - b)) *
                     cuda_hyperg_2F1(b, c - a, b - a + 1., 1. / (1. - z));
        return hyp2F1;
    } else {
        return cuda_hyperg_2F1(a, b, c, -X);
    }

    return 0;
}





#include <gsl/gsl_sf_gamma.h>


__host__ double jnu_integrand_powerlaw(double th, void *params) {
    double K = *(double *)params;
    double sth = sin(th);
    double x = K / sth;

    double factor;
    double Js;
    double p = 3.;
    double gmin = 25.;
    double gmax = 1.e7;

    // if (sth < 1.e-150 || x > 1.e8)
    //    return 0.;

    factor = sth;

    Js = pow(3., p / 2.) * (p - 1) * sth /
         (2 * (p + 1) * (pow(gmin, 1 - p) - pow(gmax, 1 - p)));
    Js *= gsl_sf_gamma((3 * p - 1) / 12.) * gsl_sf_gamma((3 * p + 19) / 12.) *
          pow(x, -(p - 1) / 2.);

    return Js * factor;
}



__host__  double jnu_integrand_kappa(double th, void *params) {

    double jnu;
    double K = *(double *)params;
    double sth = sin(th);
    double X_kappa = K / sth;
    double x, factor;
    double J_low, J_high, J_s;
    double kappa = KAPPA_SYNCH;

    sth = sin(th);
    if (X_kappa > 2.e8)
        return 0.;

    factor = (sth * sth);


    J_low = pow(X_kappa, 1. / 3.) * 4. * M_PI * tgamma(kappa - 4. / 3.) /
            (pow(3., 7. / 3.) * tgamma(kappa - 2.));

    J_high = pow(X_kappa, -(kappa - 2.) / 2.) * pow(3., (kappa - 1.) / 2.) *
             (kappa - 2.) * (kappa - 1.) / 4. * tgamma(kappa / 4. - 1. / 3.) *
             tgamma(kappa / 4. + 4. / 3.);
    x = 3. * pow(kappa, -3. / 2.);

    J_s = pow((pow(J_low, -x) + pow(J_high, -x)), -1. / x);

    jnu = J_s * factor; // *exp(-X_kappa/1.e7);
    return jnu;
}




#define HYPMIN (1e-5)
#define HYPMAX (10000)
#define N_HYP (9000)
#define kappa_min (3.)
#define kappa_max (10.)
#define N_k (70)
#define dkappa (0.1)

double hypergeom[N_k][N_HYP];
__host__ void init_emiss_tables_nth(void) {

    int k;
    double result, err, K;
    gsl_function func;
    gsl_integration_workspace *w;
	if(params.kappa_synch){
		func.function = &jnu_integrand_kappa;
		func.params = &K;
	} else if(params.powerlaw_synch){
		func.function = &jnu_integrand_powerlaw;
		func.params = &K;
	}

    double lK_min = log(KMIN);
    double dlK = log(KMAX / KMIN) / (N_ESAMP);

    double lT_min = log(TMIN);
    double dlT = log(TMAX / TMIN) / (N_ESAMP);

    /*  build table for F(K) where F(K) is given by
       \int_0^\pi ( (K/\sin\theta)^{1/2} + 2^{11/12}(K/\sin\theta)^{1/6})^2
       \exp[-(K/\sin\theta)^{1/3}]
       so that J_{\nu} = const.*F(K)
     */
    w = gsl_integration_workspace_alloc(5000);
    for (k = 0; k <= N_ESAMP; k++) {
        K = exp(k * dlK + lK_min);
        gsl_integration_qag(&func, 0., M_PI / 2., EPSABS, EPSREL, 5000,
                            GSL_INTEG_GAUSS61, w, &result, &err);
        //   gsl_integration_qags(&func, 0.01*M_PI/2., 0.99*M_PI/2. , EPSABS,
        //   EPSREL, 10000,
        //                       w, &result, &err);
        //	fprintf(stderr,"results %e err %e rel err
        //%e\n",result,err,err/result);
        F_nth[k] = log(4. * M_PI * result);
    }
    gsl_integration_workspace_free(w);

    FILE *input;
    input = fopen("hyper2f1.txt", "r");
    double dummy;
    for (int j = 0; j < N_HYP; j++) {
        for (int i = 0; i < N_k; i++) {
            // Check if fscanf successfully read exactly 1 item
            if (fscanf(input, "%lf", &dummy) != 1) {
                fprintf(stderr, "Error: Failed to read expected data from hyper2f1.txt at j=%d, i=%d\n", j, i);
                fclose(input);
                exit(1); 
            }
            hypergeom[i][j] = (dummy);
        }
    }
	fclose(input);
    /* Avoid doing divisions later */
    dlK = 1. / dlK;
    dlT = 1. / dlT;
    fprintf(stderr, "done reading hypergeom2F1.\n\n");

    return;
}

#undef HYPMIN
#undef HYPMAX
#undef N_HYP
#undef kappa_min
#undef kappa_max
#undef N_k
#undef dkappa



__host__ __device__ double F_eval_kappa(double Thetae, double Bmag, double nu) {

    double K;
    double linear_interp_F_nth(double);

    double nuc = EE * Bmag / (2. * M_PI * ME * CL);
    double kappa = KAPPA_SYNCH;
    double w = (kappa - 3.) / kappa * Thetae;
    double nus = nuc * pow(w * kappa, 2.);

    K = nu / nus;
    if (K > KMAX)
        return 0.;
    if (K < KMIN) {
        return (0);
    }
    double F_value = linear_interp_F_nth(K) * exp(-nu / NU_CUTOFF);
    return F_value;
}




__device__ double jnu_synch_nonthermal_powerlaw(double nu, double Ne, double Thetae, double B,double theta) {
    double nuc, sth, Xs, factor;
    double Js;
    double p = 3.;
    double gmin = 25.;
    double gmax = 1.e7;

    sth = sin(theta);
    nuc = EE * B / (2. * M_PI * ME * CL);
    factor = (Ne * pow(EE, 2.) * nuc) / CL;

    if (Thetae < THETAE_MIN || sth < 1e-150)
        return 0.;
    if (nu > 1.e8 * nuc)
        return (0.);

    Xs = nu / (nuc * sth);

    Js = pow(3., p / 2.) * (p - 1) * sth /
         (2 * (p + 1) * (pow(gmin, 1 - p) - pow(gmax, 1 - p)));
    Js *= cuda_sf_gamma((3 * p - 1) / 12.) * cuda_sf_gamma((3 * p + 19) / 12.) *
          pow(Xs, -(p - 1) / 2.);

    return Js * factor;
}

__device__ double jnu_synch_nonthermal_kappa(double nu, double Ne, double Thetae, double B, double theta) 
{
    // emissivity for the kappa distribution function, see Pandya et al. 2016
    double nuc, sth, nus, x, w, X_kappa, factor;
    double J_low, J_high, J_s;
    double kappa = KAPPA_SYNCH;
    w = (kappa - 3.) / kappa * Thetae;
    nuc = EE * B / (2. * M_PI * ME * CL);
    sth = sin(theta);

    factor = (Ne * pow(EE, 2.) * nuc * sth) / CL;

    nus = nuc * sth * (w * kappa) * (w * kappa);


    X_kappa = nu / nus;


    J_low = pow(X_kappa, 1. / 3.) * 4. * M_PI * tgamma(kappa - 4. / 3.) /
            (pow(3., 7. / 3.) * tgamma(kappa - 2.));

    J_high = pow(X_kappa, -(kappa - 2.) / 2.) * pow(3., (kappa - 1.) / 2.) *
             (kappa - 2.) * (kappa - 1.) / 4. * tgamma(kappa / 4. - 1. / 3.) *
             tgamma(kappa / 4. + 4. / 3.);

    x = 3. * pow(kappa, -3. / 2.);

    J_s = pow((pow(J_low, -x) + pow(J_high, -x)), -1. / x);

    return (J_s * factor) * exp(-nu / NU_CUTOFF);
}

#define JCST (EE * EE * EE /(2 * M_PI * ME * CL * CL))
__host__ __device__ double int_jnu_nth(double Ne, double Thetae, double Bmag, double nu) {
    /* Returns energy per unit time at *
     * frequency nu in cgs
     */
    int ACCZONE = 0;
    double F_eval(double Thetae, double B, double nu, int ACCZONE);

    if (Thetae < THETAE_MIN) {
        return 0.;
    }

    return JCST * Ne * Bmag * F_eval(Thetae, Bmag, nu, ACCZONE);
}
#undef JCST
__host__ __device__ double linear_interp_F_nth(double K) {

    int i;
    double di, lK;
	double lK_min = log(KMIN);
	double dlK = log(KMAX / KMIN) / (N_ESAMP);
	#ifdef __CUDA_ARCH__
	double * local_F_nth;
	local_F_nth = d_F_nth;
	#else
	double * local_F_nth;
	local_F_nth = F_nth;
	#endif


    lK = log(K);

    di = (lK - lK_min) * dlK;
    i = (int)di;
    di = di - i;

    return exp((1. - di) * local_F_nth[i] + di * local_F_nth[i + 1]);
}


__host__ __device__ double F_eval_powerlaw(double Thetae, double Bmag, double nu) {

    double K;
    double linear_interp_F_nth(double);
    double nuc = EE * Bmag / (2. * M_PI * ME * CL);

    K = nu / nuc;
    if (K > KMAX)
        return 0.;
    if (K < KMIN)
        return 0.;
    return linear_interp_F_nth(K) * exp(-nu / NU_CUTOFF);
}
