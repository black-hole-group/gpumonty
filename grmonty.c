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

/* defining declarations for global variables */
struct of_geom **geom;
int N1, N2, N3, n_within_horizon;
double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
int Ns, N_superph_recorded, N_scatt;

/* some coordinate parameters */
double a;
double R0, Rin, Rh, Rout, Rms;
double hslope;
double startx[NDIM], stopx[NDIM], dx[NDIM];

double dlE, lE0;
double gam;
double dMsim;
double M_unit, L_unit, T_unit;
double RHO_unit, U_unit, B_unit, Ne_unit, Thetae_unit;
double max_tau_scatt, Ladv, dMact, bias_norm;

gsl_rng *r;
gsl_integration_workspace *w;

#pragma omp threadprivate(r)
#include <time.h>

int main(int argc, char *argv[])
{
	double Ntot, N_superph_made;
	int quit_flag, myid;
	struct of_photon *phs;
	long int seed;
	time_t currtime, starttime;
	double N_superph_tracked = 0; // Does not include scatters

	if (argc != 4 && argc != 5) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit [seed]\nWhere seed >= 1\n");
		exit(0);
	}
	if (argc > 4) {
		sscanf(argv[4], "%ld", &seed); //user given seed
		if (seed < 1) {
			fprintf(stderr, "seed must be >= 1\nusage: grmonty Ns infilename M_unit [seed]\n");
			exit(0);
		}
		fprintf(stderr, "Using given rng seed: %lu\n", seed);
	}
	else {
		fprintf(stderr, "Using time as rng seed\n");
		seed = time(NULL) + 1; //arbitrary seed
	}

	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;

	/* initialize random number generator */
	#pragma omp parallel private(myid)
	{
		myid = omp_get_thread_num();
		 init_monty_rand(139 * myid + seed);
	}

	/* spectral bin parameters */
	dlE = 0.25;		/* bin width */
	lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */

	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_made = 0;
	N_superph_recorded = 0;
	N_scatt = 0;
	starttime = time(NULL);
	quit_flag = 0;

	fprintf(stderr, "Generating photons...\n");
	fflush(stderr);

	int phs_max = Ns;
	int ph_count = 0;
	phs = malloc(phs_max * sizeof(struct of_photon));
	while (!quit_flag) {
		if (ph_count == phs_max) {
			phs_max = 2*phs_max;
			phs = realloc(phs, phs_max*sizeof(struct of_photon));
		}
		make_super_photon(&phs[ph_count], &quit_flag);
		ph_count++;
	}
	ph_count--;
	phs = realloc(phs, ph_count*sizeof(struct of_photon)); //trim excedent memory
	N_superph_made = ph_count;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < ph_count; i++) {
		/* push ph around */

		track_super_photon(&phs[i]);

		#pragma omp atomic
		N_superph_tracked++;

		/* give interim reports on rates */
		if (((int) (N_superph_tracked)) % 100000 == 0 && N_superph_tracked > 0) {
			currtime = time(NULL);
			fprintf(stderr, "%03.2f%%: time %gs, rate %g ph/s\n",
				(N_superph_tracked / N_superph_made)*100,
				(double) (currtime - starttime),
				N_superph_tracked / (currtime -
						  starttime));
		}
	}
	currtime = time(NULL);
	fprintf(stderr, "Final time %gs, rate %g ph/s\n",
		(double) (currtime - starttime),
		N_superph_tracked / (currtime - starttime));

#ifdef _OPENMP
#pragma omp parallel
	{
		omp_reduce_spect();
	}
#endif
	report_spectrum((int) N_superph_made);

	/* done! */
	return (0);

}
