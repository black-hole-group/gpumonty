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


__device__ double jnu_inv(const double nu, const double Thetae, const double Ne, const double B, const double theta)
{
	double j;

	j = jnu_synch(nu, Ne, Thetae, B, theta);

	return (j / (nu * nu));
}

/* return Lorentz invariant scattering opacity */
__device__ double alpha_inv_scatt(const double nu, const double Thetae, const double Ne, const double * __restrict__ d_table_ptr)
{
	return (nu * kappa_es(nu, Thetae, d_table_ptr) * Ne * MP);
}
/* return Lorentz invariant absorption opacity */
__device__ double alpha_inv_abs(const double nu, const double Thetae, const double Ne, const double B,
		    double theta)
{
	double j, bnu;
	j = jnu_inv(nu, Thetae, Ne, B, theta);
	bnu = Bnu_inv(nu, Thetae);
	if (j > 0){
		return (j / (bnu + 1.e-100));
	}
	return 0;
}


/* return electron scattering opacity, in cgs */
__device__ double kappa_es(const double nu, const double Thetae, const double * __restrict__ d_table_ptr)
{
	double Eg;

	/* assume pure hydrogen gas to 
	   convert cross section to opacity */
	Eg = HPL * nu / (ME * CL * CL);
	double result = (total_compton_cross_lkup(Eg, Thetae, d_table_ptr) / MP);
	if (isnan(result)){
		printf("kappa_es is nan: %le, %le", nu, Thetae);
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