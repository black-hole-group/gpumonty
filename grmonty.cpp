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
#include "harm_model.h"
#include "kernel.h"

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
	double Ntot;
	int myid;

	//time_t currtime, starttime;

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

	/* initialize model data, auxiliary variables 
	   ==========================================
	*/
	init_model(argv);

	// global vars packaged for device
	struct simvars sim;
	getGlobals(&sim);
	// units packaged for device
	struct allunits units;
	getUnits(&units);
	printf("host=%lf struct=%lf\n", Ne_unit, units.Ne_unit);



    /* generate photons (host)
       =========================
	*/
	// photons array that will be sent to device
	double *pharr=(double *)malloc(NPHVARS*nmaxgpu*sizeof(double));
    // photon generation, host
    genPhotons(pharr, nmaxgpu);

    /* propagate photons (device)
       ==========================
    */
    //xxxxxxlaunchKernel(p, sim, units, pharr, nmaxgpu);


	// gets results back from device


    // open file for writing
    /*
    FILE *f = fopen("photons.dat", "w");
    for (int i=0; i<nmaxgpu*NPHVARS; i++) {
        fprintf(f, "%lf ", pharr[i]);
        if (!(i % NPHVARS)) {
			fprintf(f, "\n");
		}
    }
    fclose(f);
    */
    


	/*double *out;
	out=(double *)malloc(NPRIM*N1*N2*sizeof(double));
	cudaMemcpy(out, d_p, NPRIM*N1*N2*sizeof(double), cudaMemcpyDeviceToHost);	
    // open file for writing
    f = fopen("p_d.dat", "w");
    for (int i=0; i<NPRIM*N1*N2; i++) {
        fprintf(f, "%lf ", out[i]);
    }
    fclose(f);
    */

	// releases memory
	free(pharr);

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
