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

#include "host-device.h"
#include "host.h"
#include "harm_model.h"


#pragma omp threadprivate(r)
#include <time.h>

int main(int argc, char *argv[])
{
	double Ntot;
	int myid;

	//time_t currtime, starttime;

	if (argc < 4) {
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
	// other misc. values
	struct settings setup;
	getSettings(&setup);




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
    launchKernel(p, sim, units, setup, cross, pharr, nmaxgpu);


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
