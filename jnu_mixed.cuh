
//#include "decs.h"
//#pragma omp threadprivate(r)
/* 

"mixed" emissivity formula 

interpolates between Petrosian limit and
classical thermal synchrotron limit

good for Thetae > 1

*/

#define CST 1.88774862536	/* 2^{11/12} */




__device__
double d_linear_interp_K2(double Thetae)
{

	int i;
	double di, lT;

	lT = log(Thetae);

	di = (lT - lT_min) * dlT;
	i = (int) di;
	di = di - i;

	return exp((1. - di) * K2[i] + di * K2[i + 1]);
}




/* rapid evaluation of K_2(1/\Thetae) */
__device__
double d_K2_eval(double Thetae)
{

	//__device__ double d_linear_interp_K2(double);

	if (Thetae < THETAE_MIN)
		return 0.;
	if (Thetae > TMAX)
		return 2. * Thetae * Thetae;

	return d_linear_interp_K2(Thetae);
}




__device__
double d_jnu_synch(double nu, double Ne, double Thetae, double B,
		 double theta)
{
	double K2, nuc, nus, x, f, j, sth, xp1, xx;
	//__device__ double d_K2_eval(double Thetae);

	if (Thetae < THETAE_MIN)
		return 0.;

	K2 = d_K2_eval(Thetae);

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

// #define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
// double int_jnu(double Ne, double Thetae, double Bmag, double nu)
// {
// /* Returns energy per unit time at							*
//  * frequency nu in cgs										*/

// 	double j_fac, K2;
// 	double F_eval(double Thetae, double B, double nu);
// 	double K2_eval(double Thetae);


// 	if (Thetae < THETAE_MIN)
// 		return 0.;

// 	K2 = K2_eval(Thetae);
// 	if (K2 == 0.)
// 		return 0.;

// 	j_fac = Ne * Bmag * Thetae * Thetae / K2;

// 	return JCST * j_fac * F_eval(Thetae, Bmag, nu);
// }

// #undef JCST

// #define CST 1.88774862536	/* 2^{11/12} */
// double jnu_integrand(double th, void *params)
// {

// 	double K = *(double *) params;
// 	double sth = sin(th);
// 	double x = K / sth;

// 	if (sth < 1.e-150 || x > 2.e8)
// 		return 0.;

// 	return sth * sth * pow(sqrt(x) + CST * pow(x, 1. / 6.),
// 			       2.) * exp(-pow(x, 1. / 3.));
// }

// #undef CST





// void init_emiss_tables(void)
// {

// 	int k;
// 	double result, err, K, T;
// 	gsl_function func;
// 	gsl_integration_workspace *w;

// 	func.function = &jnu_integrand;
// 	func.params = &K;

// 	lK_min = log(KMIN);
// 	dlK = log(KMAX / KMIN) / (N_ESAMP);

// 	lT_min = log(TMIN);
// 	dlT = log(TMAX / TMIN) / (N_ESAMP);

// 	/*  build table for F(K) where F(K) is given by
// 	   \int_0^\pi ( (K/\sin\theta)^{1/2} + 2^{11/12}(K/\sin\theta)^{1/6})^2 \exp[-(K/\sin\theta)^{1/3}]
// 	   so that J_{\nu} = const.*F(K)
// 	 */
// 	w = gsl_integration_workspace_alloc(1000);
// 	for (k = 0; k <= N_ESAMP; k++) {
// 		K = exp(k * dlK + lK_min);
// 		gsl_integration_qag(&func, 0., M_PI / 2., EPSABS, EPSREL,
// 				    1000, GSL_INTEG_GAUSS61, w, &result,
// 				    &err);
// 		FF[k] = log(4 * M_PI * result);
// 	}
// 	gsl_integration_workspace_free(w);

// 	/*  build table for quick evaluation of the bessel function K2 for emissivity */
// 	for (k = 0; k <= N_ESAMP; k++) {
// 		T = exp(k * dlT + lT_min);
// 		K2[k] = log(gsl_sf_bessel_Kn(2, 1. / T));

// 	}

// 	/* Avoid doing divisions later */
// 	dlK = 1. / dlK;
// 	dlT = 1. / dlT;

// 	fprintf(stderr, "done.\n\n");

// 	return;
// }



// #define KFAC	(9*M_PI*ME*CL/EE)
// double F_eval(double Thetae, double Bmag, double nu)
// {

// 	double K, x;
// 	double linear_interp_F(double);

// 	K = KFAC * nu / (Bmag * Thetae * Thetae);

// 	if (K > KMAX) {
// 		return 0.;
// 	} else if (K < KMIN) {
// 		/* use a good approximation */
// 		x = pow(K, 0.333333333333333333);
// 		return (x * (37.67503800178 + 2.240274341836 * x));
// 	} else {
// 		return linear_interp_F(K);
// 	}
// }





// double linear_interp_F(double K)
// {

// 	int i;
// 	double di, lK;

// 	lK = log(K);

// 	di = (lK - lK_min) * dlK;
// 	i = (int) di;
// 	di = di - i;

// 	return exp((1. - di) * FF[i] + di * FF[i + 1]);
// }