/*

   grmonty Nph

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

   CFG 8-17-06

   Implemented synchrotron sampling, 22 Jan 07

   fixed bugs in tetrad/compton scattering routines, 31 Jan 07

   Implemented normalization for output, 6 Feb 07

   Separated out different synchrotron sampling routines
   into separate files, 8 Mar 07

   fixed bug in energy recording; bug used Kcon[0] rather than
   Kcov[0] as energy, 18 Mar 07

   major reorganization to encapsulate problem-dependent parts 5-6 Nov 07

 */

#include "decs.h"

#define BLOCK_DIM 1
#define GRID_DIM 1
#define TOTAL_THREADS 1

/* defining declarations for global variables */
struct of_geom *geom;
int N1, N2, N3, n_within_horizon;
double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
int Ns, N_superph_recorded, N_scatt;

__device__ int N1_device, N2_device;
__device__ int N_superph_recorded_device, N_scatt_device;


/* some coordinate parameters */
double a;
double R0, Rin, Rh, Rout, Rms;
double hslope;
double startx[NDIM], stopx[NDIM], dx[NDIM];

__device__ double a_device;
__device__ double R0_device, Rh_device;
__device__ double hslope_device;
__device__ double startx_device[NDIM], stopx_device[NDIM], dx_device[NDIM];

double dlE, lE0;
__device__ double dlE_device, lE0_device;
double gam;
double dMsim;
double M_unit, L_unit, T_unit;
__device__ double L_unit_device;
double RHO_unit, U_unit, B_unit, Ne_unit, Thetae_unit;
__device__ double B_unit_device;
double max_tau_scatt, Ladv, dMact, bias_norm;
__device__ double max_tau_scatt_device, bias_norm_device;
__device__ double Ne_unit_device, Thetae_unit_device;
gsl_rng *r;
gsl_integration_workspace *w;

#pragma omp threadprivate(r)
#include <time.h>

int main(int argc, char *argv[])
{
	double Ntot, N_superph_made;
	int quit_flag, myid, i;
	struct of_photon ph;
	time_t currtime, starttime;

	size_t alloc_size = sizeof(struct of_photon) * TOTAL_THREADS;
	struct of_photon photons[TOTAL_THREADS];
	struct of_photon * photons_device;
	cudaMalloc((void **)&photons_device, alloc_size);

	if (argc < 3) {
		fprintf(stderr, "usage: grmonty Ns infilename M_unit\n");
		exit(0);
	}
	sscanf(argv[1], "%lf", &Ntot);
	Ns = (int) Ntot;

	/* initialize random number generator */
#pragma omp parallel private(myid)
	{
		myid = omp_get_thread_num();
		init_monty_rand(139 * myid + time(NULL));	/* Arbitrarily picked initial seed */
	}

	/* spectral bin parameters */
	dlE = 0.25;		/* bin width */
	cudaMemset(&dlE_device, dlE, sizeof(double));
	lE0 = log(1.e-12);	/* location of first bin, in electron rest-mass units */
	cudaMemset(&lE0_device, lE0, sizeof(double));

	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_made = 0;
	N_superph_recorded = 0; /* cudaMemcpyFromSymbol(x_h, x_d, size, 0, cudaMemcpyDeviceToHost); */
	N_scatt = 0; /* cudaMemcpyFromSymbol(x_h, x_d, size, 0, cudaMemcpyDeviceToHost); */
	starttime = time(NULL);
	quit_flag = 0;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

#pragma omp parallel private(ph)
	{

		while (1) {

			/* get pseudo-quanta */
#pragma omp critical (MAKE_SPHOT)
			{
				for(i=0; i<TOTAL_THREADS; i++){
					if (!quit_flag){
						make_super_photon(&photons[i], &quit_flag);
						// make_super_photon(&ph, &quit_flag);
					}
				}
					cudaMemcpy(photons_device, photons, alloc_size, cudaMemcpyHostToDevice);
			}
			if (quit_flag)
				break;

			/* push them around */
			track_super_photon<<<GRID_DIM,BLOCK_DIM>>>(photons_device);

			/* step */
#pragma omp atomic
			N_superph_made += 1;

			/* give interim reports on rates */
			if (((int) (N_superph_made)) % 100000 == 0 && N_superph_made > 0) {
				currtime = time(NULL);
				fprintf(
					stderr, "time %g, rate %g ph/s\n",
					(double) (currtime - starttime),
					N_superph_made / (currtime - starttime)
				);
			}
		}
	}
	currtime = time(NULL);
	fprintf(
		stderr, "Final time %g, rate %g ph/s\n",
		(double) (currtime - starttime),
		N_superph_made / (currtime - starttime)
	);

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
