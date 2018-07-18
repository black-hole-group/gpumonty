/*
   Using monte carlo method, estimate spectrum of an appropriately
   scaled axisymmetric GRMHD simulation as a function of
   latitudinal viewing angle.

   Input simulation data is assumed to be in dump format provided by
   HARM code.  Location of input file is, at present, hard coded
   (see init_sim_data.c).

   Nph super-photons are generated in total and then allowed
   to propagate.  They are weighted according to the emissivity.
   The photons are pushed by the geodesic equation.
   Their weight decays according to the local absorption coefficient.
   The photons also scatter with probability related to the local
   scattering opacity.

   The electrons are assumed to have a thermal distribution
   function, and to be at the same temperature as the protons.
 */

#include "decs.h"
#include <time.h>

// External global variables to be send to GPU (used in other .c's)
extern double h_dlT, dlT, lT_min, lminw, dlw, lmint, Rh, dlE;
extern double table[(NW + 1)*(NT + 1)];
extern double K2[N_ESAMP + 1];
extern double ***p;
extern struct of_spectrum spect[N_THBINS*N_EBINS];

gsl_integration_workspace *w;

int main(int argc, char *argv[]) {
	unsigned long long Ns, N_superph_made, N_superph_recorded;
	int quit_flag, myid;
	struct of_photon *phs;
	unsigned long int seed;
	time_t currtime, starttime;

	if (argc < 4) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit [seed]\nWhere seed >= 1\n");
		exit(0);
	}
	if (argc > 4) {
		sscanf(argv[4], "%lu", &seed);
		if (seed < 1) {
			fprintf(stderr, "error: seed must be >= 1\nusage: grmonty Ns infilename M_unit [seed]\n");
			exit(0);
		}
	}
	else seed = 139 + time(NULL); /* Arbitrarily picked initial seed */
	sscanf(argv[1], "%llu", &Ns);

	cpu_rng_init(seed);

	/* spectral bin parameters */
	dlE = 0.25;		/* bin width */
	lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */

	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_made = 0;
	N_superph_recorded = 0;
	// N_scatt = 0;
	starttime = time(NULL);
	quit_flag = 0;

	fprintf(stderr, "Generating photons...\n");
	fflush(stderr);

	unsigned long long phs_max = Ns;
	unsigned long long ph_count = 0;
	phs = malloc(phs_max * sizeof(struct of_photon));
	while (!quit_flag) {
		if (ph_count == phs_max) {
			phs_max = 2*phs_max;
			phs = realloc(phs, phs_max*sizeof(struct of_photon));
		}
		make_super_photon(&phs[ph_count], &quit_flag, Ns);
		ph_count++;
	}
	ph_count--;
	phs = realloc(phs, ph_count*sizeof(struct of_photon)); //trim excedent memory
	N_superph_made = ph_count;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

	curandState_t curandstate;

	// #pragma acc enter data copyin(startx, stopx, B_unit, dlT, lT_min, K2, lminw, dlw,\
	// 	 lmint, table, L_unit, max_tau_scatt, p, Ne_unit, Thetae_unit, lE0, Rh, dlE,\
	// 	 N_superph_recorded, N_scatt, spect, dx, N1, N2, N3, n_within_horizon)
	#pragma acc parallel copyin(startx, stopx, B_unit, dlT, h_dlT, lT_min, K2,\
		 lminw, dlw, lmint, table, L_unit, max_tau_scatt, p[:NPRIM][:N1][:N1*N2],\
		 Ne_unit, Thetae_unit, lE0, Rh, dlE, dx, N1, N2, N3, n_within_horizon) \
		 copy(spect[:N_THBINS*N_EBINS], N_superph_recorded) private(curandstate)
	{

		gpu_rng_init (&curandstate, seed);

		#pragma acc loop
		for (unsigned long long i = 0; i < ph_count; i++) {
			/* push ph around */

			track_super_photon(&curandstate, &phs[i], &N_superph_recorded);

			// /* give interim reports on rates */
			// if (((int) (N_superph_made)) % 100000 == 0
			//     && N_superph_made > 0) {
			// 	currtime = time(NULL);
			// 	fprintf(stderr, "time %g, rate %g ph/s\n",
			// 		(double) (currtime - starttime),
			// 		N_superph_made / (currtime -
			// 				  starttime));
			// }
		}
	}
	currtime = time(NULL);
	fprintf(stderr, "Final time %g, rate %g ph/s\n",
		(double) (currtime - starttime),
		(double) N_superph_made / (currtime - starttime));

	report_spectrum(N_superph_made, N_superph_recorded);

	/* done! */
	return (0);

}
