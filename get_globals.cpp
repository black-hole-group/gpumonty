#include "constants.h"
#include "decs.h"
#include "kernel.h"

/* 
Photon generation function

Loop that generates enough photons to fill the GPU
memory (or less, if so specified)
*/
void genPhotons(struct simvars *sim) {
	// harm dimensions
	sim->N1=N1;
	sim->N2=N2; 
	sim->N3=N3; 
	sim->n_within_horizon=n_within_horizon;

	/* some coordinate parameters */
	sim->a=a;
	sim->R0=R0; 
	sim->Rin=Rin;
	sim->Rh=Rh; 
	sim->Rout=Rout;
	sim->Rms=Rms;
	sim->hslope;
	sim->startx[0]=startx[0];
	sim->startx[1]=startx[1];
	sim->startx[2]=startx[2];
	sim->startx[3]=startx[3];
	sim->stopx[0]=stopx[0];
	sim->stopx[1]=stopx[1];
	sim->stopx[2]=stopx[2];
	sim->stopx[3]=stopx[3];
	sim->dx[0]=dx[0];
	sim->dx[1]=dx[1];
	sim->dx[2]=dx[2];
	sim->dx[3]=dx[3];
	sim->dlE=dlE; 
	sim->lE0=lE0;
	sim->gam=gam;
	sim->dMsim=dMsim;
}