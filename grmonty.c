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
	struct of_photon ph;
	time_t currtime, starttime;

	if (argc < 3) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit\n");
		exit(0);
	}
	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;

	// gets max number of photons GPU can hold at once
	int nmaxgpu=get_max_photons(N1,N2,N3);
	if (Ntot<nmaxgpu) nmaxgpu=(int)Ntot;

	/* initialize random number generator */
#pragma omp parallel private(myid)
	{
		myid = omp_get_thread_num();
		init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */
	}

	/* spectral bin parameters */
	dlE = 0.25;		/* bin width */
	lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */

	/* initialize model data, auxiliary variables */
	init_model(argv);

	// send HARM arrays to device
	wrapper that calls
	cudaMalloc
	cudaMemcpy xxxxxxxxxxxxx

	/* 
	Photon generation loop
	==========================
	Loop that generates enough photons to fill the GPU
	memory (or less, if so specified)
	*/
	N_superph_made = 0;
	N_superph_recorded = 0;
	N_scatt = 0;
	starttime = time(NULL);
	quit_flag = 0;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

	for (int i=0; i<nmaxgpu; i++) {
		/* get pseudo-quanta */
		make_super_photon(&ph, &quit_flag);


		/* push them around */
		//track_super_photon(&ph);

		/* step */
		N_superph_made += 1;
	}

	printf("Nph = %f\n", N_superph_made);

	// gets results back from device
	cudaFree device arrays

#ifdef _OPENMP
#pragma omp parallel
	{
		//omp_reduce_spect();
	}
#endif
	//report_spectrum((int) N_superph_made);

	/* done! */
	return (0);

}
