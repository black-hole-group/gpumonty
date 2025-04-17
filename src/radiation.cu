#include "decs.h"
#include "radiation.h"
#include "jnu_mixed.h"
#include "hotcross.h"
__device__ double GPU_Bnu_inv(const double nu, const double Thetae)
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


__device__ double GPU_jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta, cudaTextureObject_t besselTexObj)
{
	double j;

	#ifdef __CUDA_ARCH__
	j = jnu_synch(nu, Ne, Thetae, B, theta, besselTexObj);
	#else
	j = jnu_synch(nu, Ne, Thetae, B, theta);
	#endif
	//printf("nu = %le, Thetae = %le, Ne = %le, B = %le, result = %le\n", nu, Thetae, Ne, B, j/(nu * nu));
	return (j / (nu * nu));
}

/* return Lorentz invariant scattering opacity */
__device__ double GPU_alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double * __restrict__ d_table_ptr)
{
	double kappa;

	kappa = GPU_kappa_es(nu, Thetae, d_table_ptr);
	return (nu * kappa * Ne * MP);
}
/* return Lorentz invariant absorption opacity */
__device__ double GPU_alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B,
		    double theta, cudaTextureObject_t besselTexObj)
{
	double j, bnu;
	#ifdef SPHERE_TEST
	theta = 1;
	#endif
	j = GPU_jnu_inv(nu, Thetae, Ne, B, theta, besselTexObj);
	bnu = GPU_Bnu_inv(nu, Thetae);
	if (j > 0){
		return (j / (bnu + 1.e-100));
	}
	return 0;
}


/* return electron scattering opacity, in cgs */
__device__ double GPU_kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr)
{
	double Eg;

	/* assume pure hydrogen gas to 
	   convert cross section to opacity */
	Eg = HPL * nu / (ME * CL * CL);
	double result = (total_compton_cross_lkup(Eg, Thetae, d_table_ptr) / MP);
	if (isnan(result)){
		printf("GPU_kappa_es is nan: %le, %le", nu, Thetae);
	}
	return result;
}

/* get frequency in fluid frame, in Hz */
__device__ double GPU_get_fluid_nu(const double X[NDIM]  , const double K[NDIM]  , const double Ucov[NDIM]  )
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

__device__ double GPU_get_bk_angle(const double X[NDIM] , const double K[NDIM]  , const double Ucov[NDIM]  ,
		    const double Bcov[NDIM]  , const double B)
{

	double k, mu;


	if (B == 0.)
		return (M_PI / 2.);

	k = fabs(K[0] * Ucov[0] + K[1] * Ucov[1] + K[2] * Ucov[2] +
		 K[3] * Ucov[3]);

	/* B is in cgs but Bcov is in code units */
	mu = (K[0] * Bcov[0] + K[1] * Bcov[1] + K[2] * Bcov[2] +
	      K[3] * Bcov[3]) / (k * B / B_UNIT);

	if (fabs(mu) > 1.)
		mu /= fabs(mu);

	return (acos(mu));
}