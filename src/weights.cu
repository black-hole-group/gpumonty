/*This should be passed to every file*/
#include "weights.h"
#include "decs.h"
#include "jnu_mixed.h"
#define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
void init_weight_table(void)
{
	int k;
	int i, j, l, lstart, lend, myid, nthreads;
	double Ne, Thetae, B, K2;
	double sum[N_ESAMP + 1], nu[N_ESAMP + 1];
	double fac, sfac;
	double Ucon[NDIM], Bcon[NDIM];

	fprintf(stderr, "Building table for superphoton weights\n");
	fflush(stderr);

	/*      Set up interpolation */
	double lnu_min = log(NUMIN);
	double lnu_max = log(NUMAX);
	double dlnu = (lnu_max - lnu_min) / (N_ESAMP);

#pragma omp parallel for schedule(static) private(i)
	for (i = 0; i <= N_ESAMP; i++) {
		sum[i] = 0.;
		nu[i] = exp(i * dlnu + lnu_min);
	}

	sfac = dx[1] * dx[2] * dx[3] * L_UNIT * L_UNIT * L_UNIT;

#pragma omp parallel private(i,j,k,Thetae, K2, Ne, B, fac, l, lstart, lend,myid,nthreads,Ucon,Bcon)
	{
		nthreads = omp_get_num_threads();
		myid = omp_get_thread_num();
		lstart = myid * (N_ESAMP / nthreads);
		lend = (myid + 1) * (N_ESAMP / nthreads);
		if (myid == nthreads - 1)
			lend = N_ESAMP + 1;

		for (i = 0; i < N1; i++)
			for (j = 0; j < N2; j++)
				for (k = 0; k < N3; k++){
						get_fluid_zone(i, j, k, &Ne, &Thetae, &B, Ucon, Bcon, geom, p);
						if (Ne == 0. || Thetae < THETAE_MIN)
							continue;
						#ifdef __CUDA_ARCH__
						K2 =K2_eval(Thetae, NULL);
						#else
						K2 = K2_eval(Thetae);
						#endif
						fac =
							(JCST * Ne * B * Thetae * Thetae /
							K2) * sfac * geom[SPATIAL_INDEX2D(i,j)].g;
						for (l = lstart; l < lend; l++){
							sum[l] +=
								fac * F_eval(Thetae, B, nu[l]);
						}
			}
#pragma omp barrier
	}
#pragma omp parallel for schedule(static) private(i)
	for (i = 0; i <= N_ESAMP; i++){
		wgt[i] = log(sum[i] / (HPL * Ns) + WEIGHT_MIN);
	}

	
	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	return;
}

#undef JCST


void init_weight_table_blackbody(void)
{
    int i, j, k, l;
    double ThetaS = 1.e-8;
	int lstart, lend, myid, nthreads;
    double sum[N_ESAMP + 1], nu[N_ESAMP + 1];
    double temperature = ThetaS * ME * CL * CL / KBOL;
    
    fprintf(stderr, "Building weight table for blackbody photons\n");
    fflush(stderr);

    /* Set up frequency grid */
    double lnu_min = log(NUMIN);
    double lnu_max = log(NUMAX);
    double dlnu = (lnu_max - lnu_min) / N_ESAMP; // This is Δln ν

    /* Initialize arrays */
    for (int i = 0; i <= N_ESAMP; i++) {
        sum[i] = 0.0;
        nu[i] = exp(i * dlnu + lnu_min);
    }

    /* Volume element factor √(-g)ΔtΔ²x */
    double dt = 1.0; // Time step
    double area_element = dt * dx[2] * dx[3] * L_UNIT * L_UNIT;

    /* Sequential computation of emission */
    //I'm setting i = 200 as R = 1./L_UNIT
	#pragma omp parallel private(i, j,k, l, lstart, lend,myid,nthreads)
	{
		nthreads = omp_get_num_threads();
		myid = omp_get_thread_num();
		lstart = myid * (N_ESAMP / nthreads);
		lend = (myid + 1) * (N_ESAMP / nthreads);
		if (myid == nthreads - 1)
			lend = N_ESAMP + 1;
		int index = N1 - 1;

		for (j = 0; j < N2; j++) {
			for (k = 0; k < N3; k++) {
				
				/* Get metric determinant for area*/
				double g = sqrt(geom[SPATIAL_INDEX2D(index, j)].gcov[2][2] * geom[SPATIAL_INDEX2D(index,j)].gcov[3][3]);

				/* Calculate emission for each frequency */
				for (l = lstart; l < lend; l++){	
					double dS = (M_PI) * 2.0 * HPL * nu[l] * nu[l] * nu[l] / 
								(CL * CL) * 1.0 / (exp(HPL * nu[l] / (KBOL * temperature)) - 1.0);

					/* Add to sum with proper weight formula components */
					sum[l] += g * area_element * dlnu * dS;
				}
			}
		}

		#pragma omp barrier
	}

    /* Calculate final weights */
#pragma omp parallel for schedule(static) private(i)
	for (i = 0; i <= N_ESAMP; i++){
		wgt[i] = log(sum[i] / (HPL * Ns));
	}
    fprintf(stderr, "done.\n\n");
    fflush(stderr);
    return;
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
		nint[i] *= dx[1] * dx[2] * dx[3] * L_UNIT * L_UNIT * L_UNIT
		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		    * 1. / HPL;
		nint[i] = log(nint[i]);
		dndlnu_max[i] = log(dndlnu_max[i]);
		//printf("%d %e %e\n", i, nint[i], dndlnu_max[i]);

	}

	return;
}


__device__ double GPU_linear_interp_weight(const double nu)
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