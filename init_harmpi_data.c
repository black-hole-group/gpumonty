//#include "decs.h"
//#include "harm_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/*

get HARM simulation data from fname

checks for consistency of coordinates in data file with
values of coordinate parameters 

Uses standard HARMPI data file format, which is the following:

Line 1: ASCII, 45 fields
Line 2 and onwards: binary, N * 42 fields, where N=N1*N2*N3 is the
	total number of grid points
*/


// Method to allocate a 3D array of floats
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



/* 
Converts a string into an array of floats. Needs to know beforehand
the number of elements. Assumes the separators are empty spaces. 
*/
double *string2float(int n, char *str) {
	double *x;
	int i;
	char *delim = " "; // input separated by spaces
	char *token = NULL;
	char *unconverted;	

    // Allocates array with n elements
    x = (double *)malloc(sizeof(double)*n); 	

	i=0;
	for (token = strtok(str, delim); token != NULL; token = strtok(NULL, delim)) {
		x[i] = strtod(token, &unconverted);
		if (!isspace(*unconverted) && *unconverted != 0) {
			printf("Input string contains a character that's not valid in a floating point constant\n");
			exit(1);
		}
		i++;
	}

	return x;
}





int main(int argc, char *argv[])
{
	FILE *fp;
	char fname[1024], header_s[1024];
	static const int NDIM=4;
	double x[4], startx[NDIM], dx[NDIM], stopx[NDIM];
	//double rp, hp, V, dV, two_temp_gam;
	int i, j, k, l;
	float var[42];
	double *header_f; // will hold header info

	/* header variables */
	int N1, N2, N3, nx, ny, nz, N1G, N2G, N3G, NPR, DOKTOT, BL;  
	double a, gam, Rin, Rout, hslope, R0, fractheta, fracphi, rbr, npow2, cpow2, DTr, t, tf, cour, DTd, DTl, DTi, DTr01, dt;
	int nstep, dump_cnt, rdump01_cnt, image_cnt, rdump_cnt, lim, failed;
	//double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

    // handle command-line argument
    if ( argc != 2 ) {
        printf( "usage: %s filename \n", argv[0] );
        exit(0);
    } 
    
    // reads filename  
    strncpy(fname, argv[1], 1024); 

	fp = fopen(fname, "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* gets HARMPI header */
    fgets(header_s, 1024, fp);

    // reads array from header string
    header_f=string2float(45,header_s);

    for (i=0; i<45; i++) {
    	printf("%f\n", header_f[i]);
    }
  
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

	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];

	fprintf(stderr, "Sim range x1, x2, x3:  %g %g, %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2], startx[3], stopx[3]);
	*/

	// Declare 3D HARMPI arrays
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
	//double ***B0 = make_3d_array(N1, N2, N3);
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


	// Reads binary data

	for (i=0; i<N1; i++) {
		for (j=0; j<N2; j++) {
			for (k = 0; k < N3; k++) {
				// reads 42 floats from binary data
				fread(var, sizeof(float), 42, fp); 

				// assigns the 3D arrays
				ti[i][j][k]=var[0]; 
				tj[i][j][k]= var[1]; 
				tk[i][j][k]= var[2]; 
				
				x1[i][j][k]= var[3]; 
				x2[i][j][k]= var[4]; 
				x3[i][j][k]= var[5]; 

				r[i][j][k]= var[6]; 
				h[i][j][k]= var[7]; 
				ph[i][j][k]= var[8]; 

				rho[i][j][k]= var[9]; 
				ug[i][j][k]= var[10]; 

				pg[i][j][k] = (gam-1.)*ug[i][j][k];

				U0[i][j][k]= var[11];
				U1[i][j][k]= var[12]; 
				U2[i][j][k]= var[13];
				U3[i][j][k]= var[14];

				//B0[i][j][k]= var[15];
				B1[i][j][k]= var[15];
				B2[i][j][k]= var[16];
				B3[i][j][k]= var[17];

				ktot[i][j][k]= pg[i][j][k]/pow(rho[i][j][k],gam);

				divb[i][j][k]= var[18];

				uu0[i][j][k]= var[19];
				uu1[i][j][k]= var[20];
				uu2[i][j][k]= var[21];
				uu3[i][j][k]= var[22];
				ud0[i][j][k]= var[23];
				ud1[i][j][k]= var[24];
				ud2[i][j][k]= var[25];
				ud3[i][j][k]= var[26];
				bu0[i][j][k]= var[27];
				bu1[i][j][k]= var[28];
				bu2[i][j][k]= var[29];
				bu3[i][j][k]= var[30];
				bd0[i][j][k]= var[31];
				bd1[i][j][k]= var[32];
				bd2[i][j][k]= var[33];
				bd3[i][j][k]= var[34];

				v1m[i][j][k]= var[35];
				v1p[i][j][k]= var[36];
				v2m[i][j][k]= var[37];
				v2p[i][j][k]= var[38];
				v3m[i][j][k]= var[39];
				v3p[i][j][k]= var[40];

				gdet[i][j][k]= var[41];
				
				//rhor = 1+(1-d.a**2)**0.5
			    //alpha = (-d.guu[0,0])**(-0.5)
			
			}
		}
	}

	// for inspecting a given array
	/*
	for (i=0; i<N1; i++) {
		for (j=0; j<N2; j++) {
			for (k = 0; k < N3; k++) {
				printf("%e ", v1m[i][j][k]);
			}
		}
	}*/

	/* done! */
	free(ti);
	fclose(fp);

	return 0;
}
