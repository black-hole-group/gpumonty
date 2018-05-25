#include "constants.h"
#include "decs.h"
//#include "kernel.h"

/* 
  Routines that create structs containing all variables that define
  the GRMHD simulation properties and units. These will be passed
  to the device, since we cannot define global variables visible
  in the host and device as previously done.
*/



void getGlobals(struct simvars *sim) {
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




void getUnits(struct allunits *units) {
	units->M_unit=M_unit;
	units->L_unit=L_unit;
	units->T_unit=T_unit;
	units->RHO_unit=RHO_unit;
	units->U_unit=U_unit;
	units->B_unit=B_unit;
	units->Ne_unit=Ne_unit;
	units->Thetae_unit=Thetae_unit;	
}



void getSettings(struct misc *setup) {
	setup->max_tau_scatt=max_tau_scatt;
	setup->RMAX=att;
}