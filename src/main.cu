#include "defs.h"
#include "decs.h"
#include "harm_model.h"
#include "gpu_header.h"

#include <time.h>

double table[NW + 1][NT + 1];
double dlw, dlT, lminw, lmint; 
double nint[NINT + 1];
double K2[N_ESAMP + 1];
double dndlnu_max[NINT + 1];


__device__ double d_table[NW + 1][NT + 1];
__device__ double d_maximum_w = 0;

__device__ unsigned long long photon_count = 0;
__device__ unsigned long long generated_sphotons, d_N_superph_recorded;
__device__ int d_N1, d_N2, d_N3, d_Ns, d_N_scatt;
__device__ double d_a, d_thetae_unit, d_startx[NDIM], d_dx[NDIM], d_wgt[N_ESAMP + 1], d_F[N_ESAMP + 1], d_K2[N_ESAMP + 1], d_bias_norm, d_stopx[NDIM], d_Rh, d_max_tau_scatt;
	

__device__ unsigned long long scattering_counter = 0;
__device__ unsigned long long d_num_scat_phs[MAX_LAYER_SCA];
__device__ unsigned long long tracking_counter = 0;
__device__ double d_nint[NINT + 1];
__device__ double d_dndlnu_max[NINT + 1];
__device__ double d_hslope = 0;
__device__ double d_R0 = 0;
__device__ int total_sca = 0;





int main(int argc, char *argv[])
{
	double Ntot;
	int quit_flag;
	struct of_photon ph;
	const char *spect_file_name = argv[3];

	if (argc < 3) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit\n");
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

    //launch_loop(ph, quit_flag, time(NULL), p, spect_file_name);

	return (0);

}

void init_model(char *args[])
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
	a = 0.9375;
	Rh = 1 + sqrt(1. - a * a);

	/* make look-up table for hot cross sections */
	init_hotcross();

	/* make table for solid angle integrated emissivity and K2 */
	init_emiss_tables();

	/* make table for superphoton weights */
	init_weight_table();

	/* make table for quick evaluation of ns_zone */
	//init_nint_table();

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


			gcon_func(geom[SPATIAL_INDEX2D(i,j)].gcov, geom[SPATIAL_INDEX2D(i,j)].gcon);
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
	fprintf(stderr, "\n");
	fprintf(stderr, "N_superph_made: %llu\n", N_superph_made);
	fprintf(stderr, "N_superph_recorded: %llu\n", N_superph_recorded);
	fprintf(stderr, "Data saved in %s\n", filepath);
	fclose(fp);

}
double dOmega_func(double x2i, double x2f)
{
	double dO;

	dO = 2. * M_PI *
	    (-cos(M_PI * x2f + 0.5 * (1. - hslope) * sin(2 * M_PI * x2f))
	     + cos(M_PI * x2i + 0.5 * (1. - hslope) * sin(2 * M_PI * x2i))
	    );

	return (dO);
}