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


void malloc_spect (struct of_spectrum ***spect) {
	*spect = malloc(N_THBINS * sizeof(struct of_spectrum *));
	for (int i = 0; i < N_THBINS; i++) {
		(*spect)[i] = malloc(N_EBINS * sizeof(struct of_spectrum));
		for (int j = 0; j < N_EBINS; j++) {
			(*spect)[i][j].dNdlE = 0.0;
			(*spect)[i][j].dEdlE = 0.0;
			(*spect)[i][j].nph = 0.0;
			(*spect)[i][j].nscatt = 0.0;
			(*spect)[i][j].X1iav = 0.0;
			(*spect)[i][j].X2isq = 0.0;
			(*spect)[i][j].X3fsq = 0.0;
			(*spect)[i][j].tau_abs = 0.0;
			(*spect)[i][j].tau_scatt = 0.0;
			(*spect)[i][j].ne0 = 0.0;
			(*spect)[i][j].thetae0 = 0.0;
			(*spect)[i][j].b0 = 0.0;
			(*spect)[i][j].E0 = 0.0;;
		}
	}
}

unsigned long long generate_photons (unsigned long long Ns, struct of_photon **phs) {
	unsigned long long phs_max = Ns;
	unsigned long long ph_count = 0;
	unsigned long long n2gen;
	double dnmax;

	*phs = malloc(phs_max * sizeof(struct of_photon));

	for (int zi = 0; zi < N1; zi++) {
		for (int zj = 0; zj < N2; zj++) {
			init_zone(zi, zj, &n2gen, &dnmax, Ns);
			if (ph_count + n2gen >= phs_max) {
				phs_max = 2*(ph_count + n2gen);
				*phs = realloc(*phs, phs_max*sizeof(struct of_photon));
			}
			for (unsigned long long gen = 0; gen < n2gen; gen++) {
				sample_zone_photon(zi, zj, dnmax, &((*phs)[ph_count]), !gen);
				// !gen is a small trick to say if this is the first photon of this zone. !gen equals (gen == 0 ? 1 : 0)
				ph_count++;
			}
		}
	}
	*phs = realloc(*phs, ph_count*sizeof(struct of_photon)); //trim excedent memory
	return ph_count;
}


void check_args (int argc, char *argv[], unsigned long long *Ns, unsigned long long *seed) {
	if (argc < 4 || argc > 5) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit [seed]\nWhere seed >= 1\n");
		exit(0);
	}
	if (argc == 5) {
		sscanf(argv[4], "%lu", seed);
		if (*seed < 1) {
			fprintf(stderr, "error: seed must be >= 1\nusage: grmonty Ns infilename M_unit [seed]\n");
			exit(0);
		}
	}
	else *seed = 139 + time(NULL); /* Arbitrarily picked initial seed */
	sscanf(argv[1], "%llu", Ns);
}

int main(int argc, char *argv[]) {
	struct of_spectrum **spect;
	unsigned long long Ns, N_superph_made, N_superph_recorded;
	struct of_photon *phs;
	unsigned long int seed;
	time_t currtime, starttime;
	curandState_t curandstate;

	check_args(argc, argv, &Ns, &seed);
	cpu_rng_init(seed);
	malloc_spect(&spect);

	/* spectral bin parameters */
	dlE = 0.25;		/* bin width */
	lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */
	/* initialize model data, auxiliary variables */
	init_model(argv);
	N_superph_recorded = 0;
	starttime = time(NULL);
	//N_scatt = 0;

	fprintf(stderr, "Generating photons...\n");
	fflush(stderr);
	N_superph_made = generate_photons(Ns, &phs);

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

	#pragma acc update device(startx[:NDIM], stopx[:NDIM], B_unit,  L_unit, max_tau_scatt, Ne_unit, Thetae_unit, lE0, dx, N1, N2, N3, n_within_horizon, h_dlT, dlT, lT_min, lminw, dlw, lmint, Rh, dlE, table, K2, p[:NPRIM][:N1][:N2])
	#pragma acc parallel copyin (phs[:N_superph_made]) private(curandstate) copy(spect[:N_THBINS][:N_EBINS], N_superph_recorded)
	{

		gpu_rng_init (&curandstate, seed);

		#pragma acc loop
		for (unsigned long long i = 0; i < N_superph_made; i++) {
			/* push ph around */

			track_super_photon(&curandstate, &phs[i], &N_superph_recorded, spect);

			// /* give interim reports on rates */
			// if (((int) (N_superph_made)) % 100000 == 0  && N_superph_made > 0) {
			// 	currtime = time(NULL);
			// 	fprintf(stderr, "time %g, rate %g ph/s\n",	(double) (currtime - starttime), N_superph_made / (currtime -
			// 				  starttime));
			// }
		}
	}

	currtime = time(NULL);
	fprintf(stderr, "Final time %g, rate %g ph/s\n",
		(double) (currtime - starttime),
		(double) N_superph_made / (currtime - starttime));

	report_spectrum(N_superph_made, N_superph_recorded, spect);

	for (int i = 0; i < N_THBINS; i++) free(spect[i]);
	free(spect);
	free(phs);

	/* done! */
	return (0);

}
