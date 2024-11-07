#include "defs.h"
#include "functions.h"
#include "model.h"


int main(int argc, char *argv[])
{
	clock_t start, end;
    start = clock();

	double Ntot;
	int quit_flag;
	struct of_photon ph = {0}; 
	const char *spect_file_name = argv[3];
	

	if (argc < 3) {
		fprintf(stderr, "usage: gpumonty, Ns, path_to_data, filename\n");
		fprintf(stderr, "example: ./gpumonty 8000000 ./data/SANE_0.9.bin SANE.spec \n");
		exit(0);
	}
	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;


	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_recorded = 0;
	N_scatt = 0;
	quit_flag = 0;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

    launch_loop(ph, quit_flag, time(NULL), p, spect_file_name);
	end = clock();
    printf("Time spent running the full code: %f seconds. Ntot = %d\n", ((double)(end - start)) / CLOCKS_PER_SEC, Ns);
	printf("%.12e\n", int_jnu(1e13, 100, 1, 1e12));

	return (0);

}

__host__ void init_model(char *args[])
{
	/* This will tell the units defined in decs.h. 
	There used to be a function here for this, but it's extremely 
	unecessary as well as taking M_UNIT as an argument*/
	fprintf(stderr, "\nUNITS\n");
	fprintf(stderr, "L,T,M: %g %g %g\n", L_UNIT, T_UNIT, M_UNIT);
	fprintf(stderr, "rho,u,B: %g %g %g\n", RHO_UNIT, U_UNIT, B_UNIT);
	max_tau_scatt = (6. * L_UNIT) * RHO_UNIT * 0.4;
	fprintf(stderr, "Initial max_tau_scatt: %g\n", max_tau_scatt);


	fprintf(stderr, "getting simulation data...\n");
	init_data(args[2]);
	/* initialize the metric */
	fprintf(stderr, "initializing geometry...\n");
	fflush(stderr);
	init_geometry();
	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	/* make look-up table for hot cross sections */
	init_hotcross();

	/* make table for solid angle integrated emissivity and K2 */
	init_emiss_tables();

	/* make table for superphoton weights */
	init_weight_table();

	/* make table for quick evaluation of ns_zone */
	init_nint_table();

}
/* set up all grid functions */
__host__ void init_geometry()
{
	int i, j, k;
	double X[NDIM];

	for (i = 0; i < N1; i++) {
		for (j = 0; j < N2; j++) {
			for (k = 0; k < N3; k++) {

			/* zone-centered */
			coord(i, j, X);
			gcov_func(X, geom[SPATIAL_INDEX2D(i,j)].gcov);


			geom[SPATIAL_INDEX2D(i,j)].g = gdet_func(geom[SPATIAL_INDEX2D(i,j)].gcov);


			gcon_func(X, geom[SPATIAL_INDEX2D(i,j)].gcov, geom[SPATIAL_INDEX2D(i,j)].gcon);
			}
		}
	}
	//here:
	/* done! */
}

__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum spect[N_THBINS][N_EBINS], const char * filename)
{
	int i, j;
	double dx2, dOmega, nuLnu, tau_scatt, L;
	FILE *fp;
    char filepath[256]; // Adjust the size as needed

    // Construct the file path
    snprintf(filepath, sizeof(filepath), "./output/%s", filename);
	fp = fopen(filepath, "w");
	if (fp == NULL) {
		fprintf(stderr, "trouble opening spectrum file\n");
		exit(0);
	}

	/* output */
	max_tau_scatt = 0.;
	L = 0.;
	//Running through all energy bins. The frequency interval will correspond to:
	//i * dlE + lE0 but this is in natural logarithm to go to log10 divide by ln(10)
	for (i = 0; i < N_EBINS; i++) {

		/* output log_10(photon energy/(me c^2))*/
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

			if (tau_scatt > max_tau_scatt){
				max_tau_scatt = tau_scatt;
			}
			L += nuLnu * dOmega * dlE;
		}
		fprintf(fp, "\n");
	}
	fprintf(stderr,
		"luminosity %g, dMact %g, efficiency %g, L/Ladv %g, max_tau_scatt %g\n",
		L, dMact * M_UNIT / T_UNIT / (MSUN / YEAR),
		L * LSUN / (dMact * M_UNIT * CL * CL / T_UNIT),
		L * LSUN / (Ladv * M_UNIT * CL * CL / T_UNIT),
		max_tau_scatt);
	for (j = 0; j < N_THBINS; j++) {
	/* convert accumulated photon number in each bin 
			to \nu L_\nu, in units of Lsun */
		dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);

		/* factor of 2 accounts for folding around equator */
		dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);
		fprintf(stderr, "dOmega for thetabin (%d) = %le\n",j,dOmega);
	}
	fprintf(stderr, "\n");
	fprintf(stderr, "N_superph_made: %llu\n", N_superph_made);
	fprintf(stderr, "N_superph_recorded: %llu\n", N_superph_recorded);
	fprintf(stderr, "Data saved in %s\n", filepath);
	fclose(fp);

}
