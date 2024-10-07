#include "defs.h"
#include "decs.h"
#include <time.h>

int main(int argc, char *argv[])
{
	double Ntot, N_superph_made;
	int quit_flag, myid;
	struct of_photon ph;
	time_t currtime, starttime;
	const char *spect_file_name = argv[3];

	if (argc < 3) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit\n");
		exit(0);
	}
	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;

	/* initialize random number generator */
// #pragma omp parallel private(myid)
// 	{
// 		myid = omp_get_thread_num();
// 		init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */
// 	}

	/* spectral bin parameters */
	//dlE = 0.25;		/* bin width */
	//lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */

	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_made = 0;
	N_superph_recorded = 0;
	N_scatt = 0;
	starttime = time(NULL);
	quit_flag = 0;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

    launch_loop(ph, quit_flag, time(NULL), p, spect_file_name);

// #pragma omp parallel private(ph)
// 	{

// 		while (1) {

// 			/* get pseudo-quanta */
// #pragma omp critical (MAKE_SPHOT)
// 			{
// 				if (!quit_flag)
// 					make_super_photon(&ph, &quit_flag);
// 			}
// 			if (quit_flag)
// 				break;

// 			/* push them around */
// 			track_super_photon(&ph);

// 			/* step */
// #pragma omp atomic
// 			N_superph_made += 1;

// 			/* give interim reports on rates */
// 			if (((int) (N_superph_made)) % 100000 == 0
// 			    && N_superph_made > 0) {
// 				currtime = time(NULL);
// 				fprintf(stderr, "time %g, rate %g ph/s\n",
// 					(double) (currtime - starttime),
// 					N_superph_made / (currtime -
// 							  starttime));
// 			}
// 		}
// 	}
// 	currtime = time(NULL);
// 	fprintf(stderr, "Final time %g, rate %g ph/s\n",
// 		(double) (currtime - starttime),
// 		N_superph_made / (currtime - starttime));

// #ifdef _OPENMP
// #pragma omp parallel
// 	{
// 		omp_reduce_spect();
// 	}
// #endif
// 	report_spectrum((int) N_superph_made);

	/* done! */
	return (0);

}