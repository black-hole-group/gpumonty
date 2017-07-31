//#include "decs.h"
//#include "harm_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*

get HARM simulation data from fname

checks for consistency of coordinates in data file with
values of coordinate parameters 

Uses standard HARM data file format

*/


// Method to allocate a 2D array of floats
double*** make_3d_array(int nx, int ny, int nz) {
	double*** arr;
	int i,j;

	arr = (double ***) malloc(nx*sizeof(double**));

	for (i = 0; i < nx; i++) {
		arr[i] = (double **) malloc(ny*sizeof(double*));

        for(j = 0; j < ny; j++) {
        	arr[i][j] = (double *) malloc(nz * sizeof(double));
        }
    }

	return arr;
} 




int main(int argc, char *argv[])
{
	FILE *fp;
	char *fname, str[1024];
	static const int NDIM=4;
	double x[4], startx[NDIM], dx[NDIM], stopx[NDIM];
	//double rp, hp, V, dV, two_temp_gam;
	int i, j, k;

	/* header variables */
	int N1, N2, N3, nx, ny, nz, N1G, N2G, N3G, NPR, DOKTOT, BL;  
	double a, gam, Rin, Rout, hslope, R0, fractheta, fracphi, rbr, npow2, cpow2, DTr, t, tf, cour, DTd, DTl, DTi, DTr01, dt, var;
	int nstep, dump_cnt, rdump01_cnt, image_cnt, rdump_cnt, lim, failed;
	//double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

	// HARM arrays
	double z[42];
	double ***ti = make_3d_array(N1, N2, N3);
	double ***tj = make_3d_array(N1, N2, N3);
	double ***tk = make_3d_array(N1, N2, N3);
	double ***x1 = make_3d_array(N1, N2, N3);
	double ***x2 = make_3d_array(N1, N2, N3);
	double ***x3 = make_3d_array(N1, N2, N3);
	double ***r = make_3d_array(N1, N2, N3);
	double ***h = make_3d_array(N1, N2, N3);
	double ***ph = make_3d_array(N1, N2, N3);
	double ***rho = make_3d_array(N1, N2, N3);
	double ***ug = make_3d_array(N1, N2, N3);
	double ***pg = make_3d_array(N1, N2, N3);
	double ***U0 = make_3d_array(N1, N2, N3);
	double ***U1 = make_3d_array(N1, N2, N3);
	double ***U2 = make_3d_array(N1, N2, N3);
	double ***U3 = make_3d_array(N1, N2, N3);
	double ***B0 = make_3d_array(N1, N2, N3);
	double ***B1 = make_3d_array(N1, N2, N3);
	double ***B2 = make_3d_array(N1, N2, N3);
	double ***B3 = make_3d_array(N1, N2, N3);
	double ***ktot = make_3d_array(N1, N2, N3);
	double ***divb = make_3d_array(N1, N2, N3);
	double ***uu0 = make_3d_array(N1, N2, N3);
	double ***uu1 = make_3d_array(N1, N2, N3);
	double ***uu2 = make_3d_array(N1, N2, N3);
	double ***uu3 = make_3d_array(N1, N2, N3);
	double ***ud0 = make_3d_array(N1, N2, N3);
	double ***ud1 = make_3d_array(N1, N2, N3);
	double ***ud2 = make_3d_array(N1, N2, N3);
	double ***ud3 = make_3d_array(N1, N2, N3);
	double ***bu0 = make_3d_array(N1, N2, N3);
	double ***bu1 = make_3d_array(N1, N2, N3);
	double ***bu2 = make_3d_array(N1, N2, N3);
	double ***bu3 = make_3d_array(N1, N2, N3);
	double ***bd0 = make_3d_array(N1, N2, N3);
	double ***bd1 = make_3d_array(N1, N2, N3);
	double ***bd2 = make_3d_array(N1, N2, N3);
	double ***bd3 = make_3d_array(N1, N2, N3);
	double ***v1m = make_3d_array(N1, N2, N3);
	double ***v1p = make_3d_array(N1, N2, N3);
	double ***v2m = make_3d_array(N1, N2, N3);
	double ***v2p = make_3d_array(N1, N2, N3);
	double ***v3m = make_3d_array(N1, N2, N3);
	double ***v3p = make_3d_array(N1, N2, N3);
	double ***gdet = make_3d_array(N1, N2, N3);




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

	fgets (str, 1024, fp);

	/* get HARMPI header */
	/*
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
	// HOW TO SWITCH BETWEEN ASCII READING AND BINARY READING?

	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];

	fprintf(stderr, "Sim range x1, x2, x3:  %g %g, %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2], startx[3], stopx[3]);

*/


	// Reads binary data
	for (i=0; i<N1; i++) {
		for (j=0; j<N2; j++) {
			for (k = 0; k < N3; k++) {
				/* - [ ] to set the types right!
				   - [x] allocate these arrays
				   - [ ] number of arrays must match file!
 				*/
				//fread((void *)(&var), sizeof(double), 1, fp);
				fread(z, sizeof(z), 42, fp);
				// AS SOON AS I READ THE BINARY PART I GET A SEGMENTATION FAULT

				//ti[i][j][k]=var;
				/*
				fread(ti[i][j][k], sizeof(double), 1, fp); 
				fread(tj[i][j][k], sizeof(double), 1, fp); 
				fread(tk[i][j][k], sizeof(double), 1, fp); 
				
				fread(x1[i][j][k], sizeof(double), 1, fp); 
				fread(x2[i][j][k], sizeof(double), 1, fp); 
				fread(x3[i][j][k], sizeof(double), 1, fp); 

				fread(r[i][j][k], sizeof(double), 1, fp); 
				fread(h[i][j][k], sizeof(double), 1, fp); 
				fread(ph[i][j][k], sizeof(double), 1, fp); 

				fread(rho[i][j][k], sizeof(double), 1, fp); 
				fread(ug[i][j][k], sizeof(double), 1, fp); 

				pg[i][j][k] = (gam-1.)*ug[i][j][k]

				fread(U0[i][j][k], sizeof(double), 1, fp);
				fread(U1[i][j][k], sizeof(double), 1, fp); 
				fread(U2[i][j][k], sizeof(double), 1, fp);
				fread(U3[i][j][k], sizeof(double), 1, fp);

				fread(B0[i][j][k], sizeof(double), 1, fp);
				fread(B1[i][j][k], sizeof(double), 1, fp);
				fread(B2[i][j][k], sizeof(double), 1, fp);
				fread(B3[i][j][k], sizeof(double), 1, fp);

				fread(ktot[i][j][k], sizeof(double), 1, fp);
				fread(divb[i][j][k], sizeof(double), 1, fp);

				fread(uu0[i][j][k], sizeof(double), 1, fp);
				fread(uu1[i][j][k], sizeof(double), 1, fp);
				fread(uu2[i][j][k], sizeof(double), 1, fp);
				fread(uu3[i][j][k], sizeof(double), 1, fp);
				fread(ud0[i][j][k], sizeof(double), 1, fp);
				fread(ud1[i][j][k], sizeof(double), 1, fp);
				fread(ud2[i][j][k], sizeof(double), 1, fp);
				fread(ud3[i][j][k], sizeof(double), 1, fp);
				fread(bu0[i][j][k], sizeof(double), 1, fp);
				fread(bu1[i][j][k], sizeof(double), 1, fp);
				fread(bu2[i][j][k], sizeof(double), 1, fp);
				fread(bu3[i][j][k], sizeof(double), 1, fp);
				fread(bd0[i][j][k], sizeof(double), 1, fp);
				fread(bd1[i][j][k], sizeof(double), 1, fp);
				fread(bd2[i][j][k], sizeof(double), 1, fp);
				fread(bd3[i][j][k], sizeof(double), 1, fp);

				fread(v1m[i][j][k], sizeof(double), 1, fp);
				fread(v1p[i][j][k], sizeof(double), 1, fp);
				fread(v2m[i][j][k], sizeof(double), 1, fp);
				fread(v2p[i][j][k], sizeof(double), 1, fp);
				fread(v3m[i][j][k], sizeof(double), 1, fp);
				fread(v3p[i][j][k], sizeof(double), 1, fp);

				fread(gdet[i][j][k], sizeof(double), 1, fp);
				
				//rhor = 1+(1-d.a**2)**0.5
			    //alpha = (-d.guu[0,0])**(-0.5)
			    */
				
			}
		}
	}

	for (i=0; i<42; i++) {
		printf("%f\n", z[i]);
	}


	/* done! */
	free(ti);
	fclose(fp);

}
