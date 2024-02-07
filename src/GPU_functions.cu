#include "gpu_header.h"



// __device__ static void init_zone(int i, int j, int k, double *nz, double *dnmax)
// {
// 	int l;
// 	double Ne, Thetae, Bmag, lbth;
// 	double dl, dn, ninterp, K2;
// 	double Ucon[NDIM], Bcon[NDIM];

// 	get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon);

// 	if (Ne == 0. || Thetae < THETAE_MIN) {
// 		*nz = 0.;
// 		*dnmax = 0.;
// 		return;
// 	}

// 	lbth = log(Bmag * Thetae * Thetae);

// 	dl = (lbth - lb_min) / dlb;
// 	l = (int) dl;
// 	dl = dl - l;
// 	if (l < 0) {
// 		*dnmax = 0.;
// 		*nz = 0.;
// 		return;
// 	} else if (l >= NINT) {
// 		fprintf(stderr,
// 			"warning: outside of nint table range %g...change in harm_utils.c\n",
// 			Bmag * Thetae * Thetae);
// 		fprintf(stderr, "lbth = %le, lb_min = %le, dlb = %le l = %d\n", lbth, lb_min, dlb, l);
// 		ninterp = 0.;
// 		*dnmax = 0.;
// 		for (l = 0; l <= N_ESAMP; l++) {
// 			dn = F_eval(Thetae, Bmag,
// 				    exp(j * dlnu +
// 					lnu_min)) / (exp(wgt[l]) +
// 						     1.e-100);
// 			if (dn > *dnmax)
// 				*dnmax = dn;
// 			ninterp += dlnu * dn;
// 		}
// 		ninterp *= dx[1] * dx[2] * dx[3] * L_UNIT * L_UNIT * L_UNIT
// 		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
// 		    * 1. / HPL;
// 	} else {
// 		if (isinf(nint[l]) || isinf(nint[l + 1])) {
// 			ninterp = 0.;
// 			*dnmax = 0.;
// 		} else {
// 			ninterp =
// 			    exp((1. - dl) * nint[l] + dl * nint[l + 1]);
// 			*dnmax =
// 			    exp((1. - dl) * dndlnu_max[l] +
// 				dl * dndlnu_max[l + 1]);
// 		}
// 	}

// 	K2 = K2_eval(Thetae);
// 	if (K2 == 0.) {
// 		*nz = 0.;
// 		*dnmax = 0.;
// 		return;
// 	}

// 	*nz = geom[SPATIAL_INDEX2D(i,j)].g * Ne * Bmag * Thetae * Thetae * ninterp / K2;
// 	if (*nz > Ns * log(NUMAX / NUMIN)) {
// 		fprintf(stderr,
// 			"Something very wrong in zone %d %d: \nB=%g  Thetae=%g  K2=%g  ninterp=%g\n\n",
// 			i, j, Bmag, Thetae, K2, ninterp);
// 		*nz = 0.;
// 		*dnmax = 0.;
// 	}

// 	return;
// }

// __device__ double K2_eval(double Thetae)
// {

// 	double linear_interp_K2(double);

// 	if (Thetae < THETAE_MIN)
// 		return 0.;
// 	if (Thetae > TMAX)
// 		return 2. * Thetae * Thetae;

// 	return linear_interp_K2(Thetae);
// }

// __device__ double linear_interp_K2(double Thetae)
// {

// 	int i;
// 	double di, lT;

// 	lT = log(Thetae);

// 	di = (lT - lT_min) * dlT;
// 	i = (int) di;
// 	di = di - i;

// 	return exp((1. - di) * K2[i] + di * K2[i + 1]);
// }