#include "decs.h"
#include "harm_model.h"

/* 
Photon generation function

Loop that generates enough photons to fill the GPU
memory (or less, if so specified)
*/
void genPhotons(double *pharr, int nmaxgpu) {
	int N_superph_made = 0;
	int quit_flag	
	//N_superph_recorded = 0;
	//N_scatt = 0;
	//starttime = time(NULL);
	//quit_flag = 0;
	struct of_photon ph;

	fprintf(stderr, "Photon generation...\n");
	fflush(stderr);

	// index i goes through each photon
	for (int i=0; i<nmaxgpu; i++) {
		/* generates one superphoton */
		make_super_photon(&ph, &quit_flag);

		// adds photon information to array
		pharr[i*NPHVARS+X0]=ph.X[0];
		pharr[i*NPHVARS+X1]=ph.X[1]; 
		pharr[i*NPHVARS+X2]=ph.X[2]; 
		pharr[i*NPHVARS+X3]=ph.X[3];
		pharr[i*NPHVARS+K0_]=ph.K[0];
		pharr[i*NPHVARS+K1_]=ph.K[1];
		pharr[i*NPHVARS+K2_]=ph.K[2];
		pharr[i*NPHVARS+K3_]=ph.K[3];
		pharr[i*NPHVARS+D0]=ph.dKdlam[0];
		pharr[i*NPHVARS+D1]=ph.dKdlam[1];
		pharr[i*NPHVARS+D2]=ph.dKdlam[2];
		pharr[i*NPHVARS+D3]=ph.dKdlam[3];
		pharr[i*NPHVARS+W]=ph.w;
		pharr[i*NPHVARS+E_]=ph.E;
		pharr[i*NPHVARS+L_]=ph.L;
		pharr[i*NPHVARS+X1I]=ph.X1i;
		pharr[i*NPHVARS+X2I]=ph.X2i;
		pharr[i*NPHVARS+TAUA]=ph.tau_abs;
		pharr[i*NPHVARS+TAUS]=ph.tau_scatt;
		pharr[i*NPHVARS+NE0]=ph.ne0;
		pharr[i*NPHVARS+TH0]=ph.thetae0;
		pharr[i*NPHVARS+B0]=ph.b0;
		pharr[i*NPHVARS+E0_]=ph.E0;
		pharr[i*NPHVARS+E0S]=ph.E0s;
		pharr[i*NPHVARS+NS]=(double)ph.nscatt;
		
		/* step */
		N_superph_made += 1;
	}

	printf("Nph = %f\n", N_superph_made);

}