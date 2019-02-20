#include "decs.h"
#include "harm_model.h"
#include <cuda.h>
#include "gpu_utils.h"

#define SPECTRUM_FILE_NAME	"grmonty.spec"

unsigned long long N_superph_recorded = 0;
struct of_spectrum spect[N_THBINS][N_EBINS];

/*******************************************************************************
* Host-only Functions
*
*******************************************************************************/

void init_spectrum () {
	for (int i = 0; i < N_THBINS; i++)
		for (int j = 0; j < N_EBINS; j++) {
			spect[i][j].dNdlE = 0.0;
			spect[i][j].dEdlE = 0.0;
			spect[i][j].nph = 0.0;
			spect[i][j].nscatt = 0.0;
			spect[i][j].X1iav = 0.0;
			spect[i][j].X2isq = 0.0;
			spect[i][j].X3fsq = 0.0;
			spect[i][j].tau_abs = 0.0;
			spect[i][j].tau_scatt = 0.0;
			spect[i][j].ne0 = 0.0;
			spect[i][j].thetae0 = 0.0;
			spect[i][j].b0 = 0.0;
			spect[i][j].E0 = 0.0;;
		}
}

/* output spectrum to file */
void report_spectrum(unsigned long long N_superph_made)
{
	int i, j;
	double dx2, dOmega, nuLnu, tau_scatt, L;
	FILE *fp;

	double nu0,nu1,nu,fnu ;
	double dsource = 8000*PC ;

	fp = fopen(SPECTRUM_FILE_NAME, "w");
	if (fp == NULL) {
		fprintf(stderr, "trouble opening spectrum file\n");
		exit(0);
	}

	/* output */
	max_tau_scatt = 0.;
	L = 0.;
	for (i = 0; i < N_EBINS; i++) {

		/* output log_10(photon energy/(me c^2)) */
		fprintf(fp, "%10.5g ", (i * dlE + lE0) / M_LN10);

		for (j = 0; j < N_THBINS; j++) {

			/* convert accumulated photon number in each bin
			   to \nu L_\nu, in units of Lsun */
			dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);

			/* factor of 2 accounts for folding around equator */
			dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);

			nuLnu =
			    (ME * CL * CL) * (4. * M_PI / dOmega) * (1. /
								     dlE);

			nuLnu *= spect[j][i].dEdlE;
			nuLnu /= LSUN;

			tau_scatt =
			    spect[j][i].tau_scatt / (spect[j][i].dNdlE +
						     SMALL);
			fprintf(fp,
				"%10.5g %10.5g %10.5g %10.5g %10.5g %10.5g ",
				nuLnu,
				spect[j][i].tau_abs / (spect[j][i].dNdlE +
						       SMALL), tau_scatt,
				spect[j][i].X1iav / (spect[j][i].dNdlE +
						     SMALL),
				sqrt(fabs
				     (spect[j][i].X2isq /
				      (spect[j][i].dNdlE + SMALL))),
				sqrt(fabs
				     (spect[j][i].X3fsq /
				      (spect[j][i].dNdlE + SMALL)))
			    );


			nu0 = ME * CL * CL * exp((i - 0.5) * dlE + lE0) / HPL ;
			nu1 = ME * CL * CL * exp((i + 0.5) * dlE + lE0) / HPL ;

			if(nu0 < 230.e9 && nu1 > 230.e9) {
				nu = ME * CL * CL * exp(i * dlE + lE0) / HPL ;
				fnu = nuLnu*LSUN/(4.*M_PI*dsource*dsource*nu*JY) ;
				fprintf(stderr,"fnu: %10.5g\n",fnu) ;
			}

			/* added to give average # scatterings */
			fprintf(fp,"%10.5g ",spect[j][i].nscatt/ (
				spect[j][i].dNdlE + SMALL)) ;

			if (tau_scatt > max_tau_scatt)
				max_tau_scatt = tau_scatt;

			L += nuLnu * dOmega * dlE / (4. * M_PI);
		}
		fprintf(fp, "\n");
	}
	fprintf(stderr,
		"luminosity %g, dMact %g, efficiency %g, L/Ladv %g, max_tau_scatt %g\n",
		L, dMact * M_unit / T_unit / (MSUN / YEAR),
		L * LSUN / (dMact * M_unit * CL * CL / T_unit),
		L * LSUN / (Ladv * M_unit * CL * CL / T_unit),
		max_tau_scatt);
	fprintf(stderr, "\n");
	fprintf(stderr, "N_superph_made: %llu\n", N_superph_made);
	fprintf(stderr, "N_superph_recorded: %llu\n", N_superph_recorded);

	fclose(fp);

}

/*
	record contribution of super photon to spectrum.

	This routine should make minimal assumptions about the
	coordinate system.
*/
void record_super_photon(struct of_photon *ph)
{
	double lE, dx2;
	int iE, ix2;

	if (isnan(ph->w) || isnan(ph->E)) {
		// fprintf(stderr, "record isnan: %g %g\n", ph->w, ph->E);
		return;
	}

	if (ph->tau_scatt > max_tau_scatt) {
		#pragma omp critical (MAXTAUSCATT_UPDATE)
		if (ph->tau_scatt > max_tau_scatt) {
			max_tau_scatt = ph->tau_scatt;
			CUDASAFE(cudaMemcpyToSymbolAsync(d_max_tau_scatt, &max_tau_scatt,
											 sizeof(double), 0,
											 cudaMemcpyHostToDevice,
											 max_tau_scatt_stream));
		}
	}
	/* currently, bin in x2 coordinate */

	/* get theta bin, while folding around equator */
	dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);
	if (ph->X[2] < 0.5 * (startx[2] + stopx[2]))
		ix2 = (int) (ph->X[2] / dx2);
	else
		ix2 = (int) ((stopx[2] - ph->X[2]) / dx2);

	/* check limits */
	if (ix2 < 0 || ix2 >= N_THBINS)
		return;

	/* get energy bin */
	lE = log(ph->E);
	iE = (int) ((lE - lE0) / dlE + 2.5) - 2;	/* bin is centered on iE*dlE + lE0 */

	/* check limits */
	if (iE < 0 || iE >= N_EBINS)
		return;

	#pragma omp atomic update
	N_superph_recorded += 1;
	// #atomic
	// N_scatt += ph->nscatt;

	/* sum in photon */
	#pragma omp atomic update
	spect[ix2][iE].dNdlE += ph->w;
	#pragma omp atomic update
	spect[ix2][iE].dEdlE += ph->w * ph->E;
	#pragma omp atomic update
	spect[ix2][iE].tau_abs += ph->w * ph->tau_abs;
	#pragma omp atomic update
	spect[ix2][iE].tau_scatt += ph->w * ph->tau_scatt;
	#pragma omp atomic update
	spect[ix2][iE].X1iav += ph->w * ph->X1i;
	#pragma omp atomic update
	spect[ix2][iE].X2isq += ph->w * (ph->X2i * ph->X2i);
	#pragma omp atomic update
	spect[ix2][iE].X3fsq += ph->w * (ph->X[3] * ph->X[3]);
	#pragma omp atomic update
	spect[ix2][iE].ne0 += ph->w * (ph->ne0);
	#pragma omp atomic update
	spect[ix2][iE].b0 += ph->w * (ph->b0);
	#pragma omp atomic update
	spect[ix2][iE].thetae0 += ph->w * (ph->thetae0);
	#pragma omp atomic update
	spect[ix2][iE].nscatt += ph->w * ph->nscatt;
	#pragma omp atomic update
	spect[ix2][iE].nph += 1.;

}
