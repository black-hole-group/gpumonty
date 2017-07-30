//#include "decs.h"
//#include "harm_model.h"
#include <stdio.h>
#include <stdlib.h>

/*

get HARM simulation data from fname

checks for consistency of coordinates in data file with
values of coordinate parameters 

Uses standard HARM data file format

*/


int main(int argc, char *argv[])
{
	FILE *fp;
	char *fname;
	double x[4];
	//double rp, hp, V, dV, two_temp_gam;
	int i, j, k, N1, N2, N3, nx, ny, nz, N1G, N2G, N3G, NPR, DOKTOT, BL;  
	static const int NDIM=4;
	double startx[NDIM], dx[NDIM];
	double a, gam, Rin, Rout, hslope, R0, fractheta, fracphi, rbr, npow2, cpow2, DTr;

	/* header variables not used except locally */
	double t, tf, cour, DTd, DTl, DTi, DTr01, dt;
	int nstep, dump_cnt, rdump01_cnt, image_cnt, rdump_cnt, lim, failed;
	double r, h, divb, vmin, vmax, gdet;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

    // handle command-line argument
    if ( argc != 2 ) {
        printf( "usage: %s filename \n", argv[0] );
        exit(0);
    } 
    
    //sscanf(argv[1], "%s", &fname); // reads command-line argument

	fp = fopen(argv[1], "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* get HARMPI header */
	fscanf(fp, "%lf ", &t);
	// per tile resolution
	fscanf(fp, "%d ", &N1);
	fscanf(fp, "%d ", &N2);
	fscanf(fp, "%d ", &N3);	
	// total resolution
	fscanf(fp, "%d ", &nx);
	fscanf(fp, "%d ", &ny);
	fscanf(fp, "%d ", &nz);	
	// number of ghost cells
	fscanf(fp, "%d ", &N1G);
	fscanf(fp, "%d ", &N2G);
	fscanf(fp, "%d ", &N3G);	
	// 
	fscanf(fp, "%lf ", &startx[1]);
	fscanf(fp, "%lf ", &startx[2]);
	fscanf(fp, "%lf ", &startx[3]);
	fscanf(fp, "%lf ", &dx[1]);
	fscanf(fp, "%lf ", &dx[2]);
	fscanf(fp, "%lf ", &dx[3]);
	fscanf(fp, "%lf ", &tf);
	fscanf(fp, "%d ", &nstep);
	fscanf(fp, "%lf ", &a);
	fscanf(fp, "%lf ", &gam);
	fscanf(fp, "%lf ", &cour);
	fscanf(fp, "%lf ", &DTd);
	fscanf(fp, "%lf ", &DTl);
	fscanf(fp, "%lf ", &DTi);
	fscanf(fp, "%lf ", &DTr);
	fscanf(fp, "%lf ", &DTr01);	
	fscanf(fp, "%d ", &dump_cnt);
	fscanf(fp, "%d ", &image_cnt);
	fscanf(fp, "%d ", &rdump_cnt);
	fscanf(fp, "%d ", &rdump01_cnt);
	fscanf(fp, "%lf ", &dt);
	fscanf(fp, "%d ", &lim);
	fscanf(fp, "%d ", &failed);
	fscanf(fp, "%lf ", &Rin);
	fscanf(fp, "%lf ", &Rout);
	fscanf(fp, "%lf ", &hslope);
	fscanf(fp, "%lf ", &R0);
	fscanf(fp, "%d ", &NPR);
	fscanf(fp, "%d ", &DOKTOT);
	fscanf(fp, "%lf ", &fractheta);
	fscanf(fp, "%lf ", &fracphi);
	fscanf(fp, "%lf ", &rbr);
	fscanf(fp, "%lf ", &npow2);
	fscanf(fp, "%lf ", &cpow2);
	fscanf(fp, "%d ", &BL);





	/* done! */
	fclose(fp);

}
