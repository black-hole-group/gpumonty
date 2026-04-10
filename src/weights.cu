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
#include "radiation.h"
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
			const double kappa = get_model_kappa_ijk(i, j, k, p);
			if (Ne == 0. || Thetae < THETAE_MIN)
				continue;
			double fac = sfac * geom[SPATIAL_INDEX2D(i, j)].g;
			double K2 = K2_eval(Thetae);

			for (int l = 0; l <= N_ESAMP; l++){
				sum_local[l] += int_jnu_total(Ne, Thetae, B, nu[l],K2, kappa) * fac;
			}
		}

		for (int l = 0; l <= N_ESAMP; l++)
#pragma omp atomic
			sum[l] += sum_local[l];
		free(sum_local);

	}

	for (int i = 0; i <= N_ESAMP; i++){
		wgt[i] = log(sum[i] / (HPL * (int) params.Ns) + WEIGHT_MIN);
	}

	fprintf(stderr, "done.\n");
	fflush(stderr);
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