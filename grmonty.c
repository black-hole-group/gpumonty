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

int main(int argc, char *argv[]) {
	double Ntot, N_superph_made;
	int quit_flag, myid;
	struct of_photon *phs;
	unsigned long int seed;
	time_t currtime, starttime;

	if (argc < 4) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit [seed]\nWhere seed > 0\n");
		exit(0);
	}
	if (argc > 4) {
		sscanf(argv[4], "%lu", &seed);
	}
	else seed = -1;
	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;

	/* initialize random number generator */
	#pragma omp parallel private(myid)
	{
		if (seed > 0) init_monty_rand(seed);
		else {
			myid = omp_get_thread_num();
			init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */
		}
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

	#pragma omp parallel for
	for (int i = 0; i < ph_count; i++) {
		/* push ph around */

		track_super_photon(&phs[i]);

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
	currtime = time(NULL);
	fprintf(stderr, "Final time %g, rate %g ph/s\n",
		(double) (currtime - starttime),
		N_superph_made / (currtime - starttime));

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
