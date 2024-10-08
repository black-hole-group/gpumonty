#include "defs.h"
#include "decs.h"
#include "harm_model.h"

#include <time.h>

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

    launch_loop(ph, quit_flag, time(NULL), p, spect_file_name);

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
	#if(HAMR)
		#if(HAMR3D)
		init_hamr3D_data(args[2]);/*PEDRO EDIT -> file to read H-AMR 3D data*/
		#else
		init_hamr_data(args[2]); /*PEDRO EDIT -> file to read H-AMR 2D data*/
		#endif
	#else
	init_harm_data(args[2]);	/* read in HARM simulation data */
	#endif
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
	init_nint_table();

}