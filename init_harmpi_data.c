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


/*
Method to allocate a 3D array of floats that can be accessed
as A[i][j][k].
*/
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
	char fname[1024];
	static const int NDIM=4;
	double x[4], startx[NDIM], dx[NDIM], stopx[NDIM];
	//double rp, hp, V, dV, two_temp_gam;
	int i, j, k, l;
	float var[42];

	/* header variables */
	char header_s[1024]; // header string
	double *header_f; // header values
	int N1, N2, N3, nx, ny, nz, N1G, N2G, N3G, NPR, DOKTOT, BL;  
	double a, gam, Rin, Rout, hslope, R0, fractheta, fracphi, rbr, npow2, cpow2, DTr, t, tf, cour, DTd, DTl, DTi, DTr01, dt;
	int nstep, dump_cnt, rdump01_cnt, image_cnt, rdump_cnt, lim, failed;
	//double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

	/*
	=======================
	opens file
	=======================
	*/

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

	/*
	=======================
	gets HARMPI header 
	=======================
	*/
    fgets(header_s, 1024, fp);

    // reads array from header string
    header_f=string2float(45,header_s);
  
	t=header_f[0];
	// per tile resolution
	N1=header_f[1];
	N2=header_f[2];
	N3=header_f[3];
	// total resolution
	nx=header_f[4];
	ny=header_f[5];
	nz=header_f[6];
	// number of ghost cells
	N1G=header_f[7];
	N2G=header_f[8];
	N3G=header_f[9];
	// 
	startx[1]=header_f[10];
	startx[2]=header_f[11];
	startx[3]=header_f[12];
	dx[1]=header_f[13];
	dx[2]=header_f[14];
	dx[3]=header_f[15];
	tf=header_f[16];
	nstep=header_f[17];
	a=header_f[18];
	gam=header_f[19];
	cour=header_f[20];
	DTd=header_f[21];
	DTl=header_f[22];
	DTi=header_f[23];
	DTr=header_f[24];
	DTr01=header_f[25];
	dump_cnt=header_f[26];
	image_cnt=header_f[27];
	rdump_cnt=header_f[28];
	rdump01_cnt=header_f[29];
	dt=header_f[30];
	lim=header_f[31];
	failed=header_f[32];
	Rin=header_f[33];
	Rout=header_f[34];
	hslope=header_f[35];
	R0=header_f[36];
	NPR=header_f[37];
	DOKTOT=header_f[38];
	fractheta=header_f[39];
	fracphi=header_f[40];
	rbr=header_f[41];
	npow2=header_f[42];
	cpow2=header_f[43];
	BL=header_f[44];

	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];

	fprintf(stderr, "Sim range x1, x2, x3:  %g %g, %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2], startx[3], stopx[3]);

	/*
	=====================
	Binary data
	=====================
	*/

	/* Declare 3D HARMPI arrays.
	The meaning of these variables is explained in 
	https://github.com/atchekho/harmpi/blob/master/tutorial.md#understanding-the-output
	*/
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
				// reads 42 floats from binary data in each pass
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
