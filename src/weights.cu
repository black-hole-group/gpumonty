/*
 * GPUmonty - weights.cu
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
/*This should be passed to every file*/
#include "decs.h"
#include "weights.h"
#include "jnu_mixed.h"
#include "utils.h"
//#define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
// #define JCST (7.089473804413026e-24) /* √2·e³/(27·mₑ·c²) [CGS] */
// __host__ void init_weight_table()
// {
// 	int k;
// 	int i, j, l, lstart, lend, myid, nthreads;
// 	double Ne, Thetae, B;
// 	//double K2;
// 	double sum[N_ESAMP + 1], nu[N_ESAMP + 1];
// 	double fac, sfac;
// 	double Ucon[NDIM], Bcon[NDIM];

// 	fprintf(stderr, "Building table for superphoton weights\n");
// 	fflush(stderr);

// 	/*      Set up interpolation */
// 	double lnu_min = log(NUMIN);
// 	double lnu_max = log(NUMAX);
// 	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);

// #pragma omp parallel for schedule(static) private(i)
// 	for (i = 0; i <= N_ESAMP; i++) {
// 		sum[i] = 0.;
// 		nu[i] = exp(i * dlnu + lnu_min);
// 	}

// 	sfac = dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit;

// #pragma omp parallel private(i,j,k,Thetae, K2, Ne, B, fac, l, lstart, lend,myid,nthreads,Ucon,Bcon)
// 	{
// 		nthreads = omp_get_num_threads();
// 		myid = omp_get_thread_num();
// 		lstart = myid * (N_ESAMP / nthreads);
// 		lend = (myid + 1) * (N_ESAMP / nthreads);
// 		if (myid == nthreads - 1)
// 			lend = N_ESAMP + 1;

// 		for (i = 0; i < N1; i++)
// 			for (j = 0; j < N2; j++)
// 				for (k = 0; k < N3; k++){
// 						get_fluid_zone(i, j, k, &Ne, &Thetae, &B, Ucon, Bcon, geom, p);
// 						if (Ne == 0. || Thetae < THETAE_MIN)
// 							continue;
// 						// K2 = K2_eval(Thetae);
// 						// fac =
// 						// 	(JCST * Ne * B * Thetae * Thetae /
// 						// 	K2) * sfac * geom[SPATIAL_INDEX2D(i,j)].g;
// 						fac = sfac * geom[SPATIAL_INDEX2D(i,j)].g;
						
// 						for (l = lstart; l < lend; l++){
// 							// sum[l] +=
// 							// 	fac * F_eval(Thetae, B, nu[l]);
// 							sum[l] += int_jnu_total(Ne, Thetae, B, nu[l]) * fac;
// 						}
// 			}
// #pragma omp barrier
// 	}
// #pragma omp parallel for schedule(static) private(i)
// 	for (i = 0; i <= N_ESAMP; i++){
// 		wgt[i] = log(sum[i] / (HPL * (int) params.Ns) + WEIGHT_MIN);
// 	}

	
// 	fprintf(stderr, "done.\n\n");
// 	fflush(stderr);
// // Their equivalent for sum is int_jnu * fac * gdet = (JCST * Ne * B * Thetae * Thetae /K2) * F_eval * sfac * geom[SPATIAL_INDEX2D(i,j)].g
// // So int_jnu = (JCST * Ne * B * Thetae * Thetae /K2) * F_eval
// 	return;
// }

// #undef JCST

__host__ void init_weight_table()
{
	fprintf(stderr, "Building table for superphoton weights\n");
	fflush(stderr);

	double sum[N_ESAMP + 1], nu[N_ESAMP + 1];

	/* Set up frequency grid */
	double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double dlnu    = (lnu_max - lnu_min) / N_ESAMP;

	for (int i = 0; i <= N_ESAMP; i++) {
		sum[i] = 0.;
		nu[i]  = exp(i * dlnu + lnu_min);
	}

	double sfac = dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit;

#pragma omp parallel
	{
		double *sum_local = (double*)calloc(N_ESAMP + 1, sizeof(double));
		if (!sum_local) { fprintf(stderr, "calloc failed\n"); exit(1); }


#pragma omp for collapse(3) schedule(static)
		for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
		for (int k = 0; k < N3; k++) {
			double Ne, Thetae, B;
			double Ucon[NDIM], Bcon[NDIM];
			get_fluid_zone(i, j, k, &Ne, &Thetae, &B, Ucon, Bcon, geom, p);
			if (Ne == 0. || Thetae < THETAE_MIN)
				continue;
			double fac = sfac * geom[SPATIAL_INDEX2D(i, j)].g;
			double K2 = K2_eval(Thetae);

			for (int l = 0; l <= N_ESAMP; l++)
				sum_local[l] += int_jnu_total(Ne, Thetae, B, nu[l],K2 ) * fac;
		}

		for (int l = 0; l <= N_ESAMP; l++)
#pragma omp atomic
			sum[l] += sum_local[l];
		free(sum_local);

	}

	for (int i = 0; i <= N_ESAMP; i++)
		wgt[i] = log(sum[i] / (HPL * (int) params.Ns) + WEIGHT_MIN);

	fprintf(stderr, "done.\n");
	fflush(stderr);
}

__host__ void init_nint_table(void)
{
	/*
	This function represents the integral of the solid angle averaged emissivity
	over frequency.
	*/

	int i, j;
	double Bmag, dn;
	static int firstc = 1;
    double lb_min, dlb;
    double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);
	if (firstc) {
		lb_min = log(BTHSQMIN);
		dlb = log(BTHSQMAX / BTHSQMIN) / NINT;
		firstc = 0;
	}

	for (i = 0; i <= NINT; i++) {
		nint[i] = 0.;
		Bmag = exp(i * dlb + lb_min);
		dndlnu_max[i] = 0.;
		for (j = 0; j < N_ESAMP; j++) {
			dn = F_eval(1., Bmag,
				    exp(j * dlnu +
					lnu_min)) / (exp(wgt[j]) +
						     1.e-100);
			if (dn > dndlnu_max[i])
				dndlnu_max[i] = dn;
			nint[i] += dlnu * dn;
		}
		nint[i] *= dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit
		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		    * 1. / HPL;
		nint[i] = log(nint[i]);
		dndlnu_max[i] = log(dndlnu_max[i]);

	}

	return;
}


__device__ double linear_interp_weight(const double nu)
{
	int i;
	double di, lnu;
	double lnu_max = log(NUMAX);
	double lnu_min = log(NUMIN);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);
	lnu = log(nu);

	di = (lnu - lnu_min) / dlnu;
	i = (int) di;
	di = di - i;
	return exp((1. - di) * d_wgt[i] + di * d_wgt[i + 1]);
}