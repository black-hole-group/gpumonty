
extern "C"
{
#include "decs.h"
}

/*This should be passed to every file*/
#include "defs_CUDA.h"

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
					get_fluid_zone(i, j, k, &Ne, &Thetae, &B,
							Ucon, Bcon, geom, p);
					if (Ne == 0. || Thetae < THETAE_MIN)
						continue;
					K2 = K2_eval(Thetae);
					fac =
						(JCST * Ne * B * Thetae * Thetae /
						K2) * sfac * geom[SPATIAL_INDEX2D(i,j)].g;
					for (l = lstart; l < lend; l++){
						//fprintf(stderr, "external B = %le\n", B);
						//if(l==0 && ((i == 0 && j == 255) || (i == 1 && j == 0))){
						// if(l==0 && (i == 0 || i == 1 ||i == 2 || i == 3 || i == 4)){
						// 	printf("(%d, %d, %d), fac = %.15e, B = %.15e, Thetae = %.15e, nu = %.15e, sum = %.15e\n", i, j, k, fac, B, Thetae, nu[0], sum[0]);
						// 	printf("Ne = %.15e, K2 = %.15e, K2_eval = %.15e, sfac = %.15e, geom = %.15e\n", Ne, K2, K2_eval(Thetae), sfac, geom[SPATIAL_INDEX2D(i,j)].g);
						// }
						sum[l] +=
							fac * F_eval(Thetae, B, nu[l]);	
				}
			}
#pragma omp barrier
	}
#pragma omp parallel for schedule(static) private(i)
	for (i = 0; i <= N_ESAMP; i++){
		wgt[i] = log(sum[i] / (HPL * Ns));
	}

	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	return;
}

#undef JCST


__host__ void init_nint_table(void)
{

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
			// if(i == 12218){
			// 	printf("wgt[%d] = %.12e\n", j, wgt[j]);
			// }
			nint[i] += dlnu * dn;
		}
		nint[i] *= dx[1] * dx[2] * dx[3] * L_UNIT * L_UNIT * L_UNIT
		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		    * 1. / HPL;
		nint[i] = log(nint[i]);

		dndlnu_max[i] = log(dndlnu_max[i]);
	}

	return;
}