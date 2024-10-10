
/***********************************************************************************
    Copyright 2013 Joshua C. Dolence, Charles F. Gammie, Monika Mo\'scibrodzka,
                   and Po Kin Leung

                        GRMONTY  version 1.0   (released February 1, 2013)

    This file is part of GRMONTY.  GRMONTY v1.0 is a program that calculates the
    emergent spectrum from a model using a Monte Carlo technique.

    This version of GRMONTY is configured to use input files from the HARM code
    available on the same site.   It assumes that the source is a plasma near a
    black hole described by Kerr-Schild coordinates that radiates via thermal 
    synchrotron and inverse compton scattering.
    
    You are morally obligated to cite the following paper in any
    scientific literature that results from use of any part of GRMONTY:

    Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009,
        Astrophysical Journal Supplement, 184, 387

    Further, we strongly encourage you to obtain the latest version of 
    GRMONTY directly from our distribution website:
    http://rainman.astro.illinois.edu/codelib/

    GRMONTY is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    GRMONTY is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GRMONTY; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/


#include "decs.h"
#include "harm_model.h"
/*

get HARM simulation data from fname

checks for consistency of coordinates in data file with
values of coordinate parameters 

Uses standard HARM data file format

CFG 1 Sept 07

*/

void check_scan_error(int scan_output, int number_of_arguments ) {
	if(scan_output == EOF)fprintf(stderr, "error reading HARM header\n");
	else if(scan_output != number_of_arguments){
		fprintf(stderr, "Not all HARM data could be set\n");
		fprintf(stderr, "Scan Output = %d\n", scan_output);
		fprintf(stderr, "Number of Arguments = %d\n", number_of_arguments);
		exit(1);
	}
}
void init_harm_data(char *fname)
{
	FILE *fp;
	double x[4];
	double rp, hp, V, dV, two_temp_gam;
	int i, j, k;

	/* header variables not used except locally */
	double t, tf, cour, DTd, DTl, DTi, dt;
	int nstep, DTr, dump_cnt, image_cnt, rdump_cnt, lim, failed;
	double r, h, divb, vmin, vmax, gdet;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];

	fp = fopen(fname, "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* get standard HARM header */
	fscanf(fp, "%lf ", &t);
	fscanf(fp, "%d ", &N1);
	fscanf(fp, "%d ", &N2);
	fscanf(fp, "%lf ", &startx[1]);
	fscanf(fp, "%lf ", &startx[2]);
	fscanf(fp, "%lf ", &dx[1]);
	fscanf(fp, "%lf ", &dx[2]);
	fscanf(fp, "%lf ", &tf);
	fscanf(fp, "%d ", &nstep);
	fscanf(fp, "%lf ", &a);
	fscanf(fp, "%lf ", &gam);
	fscanf(fp, "%lf ", &cour);
	fscanf(fp, "%lf ", &DTd);
	fscanf(fp, "%lf ", &DTl);
	fscanf(fp, "%lf ", &DTi);
	fscanf(fp, "%d ", &DTr);
	fscanf(fp, "%d ", &dump_cnt);
	fscanf(fp, "%d ", &image_cnt);
	fscanf(fp, "%d ", &rdump_cnt);
	fscanf(fp, "%lf ", &dt);
	fscanf(fp, "%d ", &lim);
	fscanf(fp, "%d ", &failed);
	fscanf(fp, "%lf ", &Rin);
	fscanf(fp, "%lf ", &Rout);
	fscanf(fp, "%lf ", &hslope);
	fscanf(fp, "%lf ", &R0);

	/* nominal non-zero values for axisymmetric simulations */
	startx[0] = 0.;
	startx[3] = 0.;

	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = 2. * M_PI;

	fprintf(stderr, "Sim range x1, x2:  %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2]);

	dx[0] = 1.;
	dx[3] = 2. * M_PI;

	/* Allocate storage for all model size dependent variables */
	init_storage();

	two_temp_gam =
	    0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) +
		   gam);
	Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);

	dMact = 0.;
	Ladv = 0.;
	bias_norm = 0.;
	V = 0.;
	dV = dx[1] * dx[2] * dx[3];
	for (k = 0; k < N1 * N2; k++) {
		j = k % N2;
		i = (k - j) / N2;
		fscanf(fp, "%lf %lf %lf %lf", &x[1], &x[2], &r, &h);

		/* check that we've got the coordinate parameters right */
		bl_coord(x, &rp, &hp);
		if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {
			fprintf(stderr, "grid setup error\n");
			fprintf(stderr, "rp,r,hp,h: %g %g %g %g\n",
				rp, r, hp, h);
			fprintf(stderr,
				"edit R0, hslope, compile, and continue\n");
			exit(1);
		}

		fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf",
		       &p[KRHO][i][j],
		       &p[UU][i][j],
		       &p[U1][i][j],
		       &p[U2][i][j],
		       &p[U3][i][j],
		       &p[B1][i][j], &p[B2][i][j], &p[B3][i][j]);


		fscanf(fp, "%lf", &divb);

		fscanf(fp, "%lf %lf %lf %lf",
		       &Ucon[0], &Ucon[1], &Ucon[2], &Ucon[3]);
		fscanf(fp, "%lf %lf %lf %lf", &Ucov[0],
		       &Ucov[1], &Ucov[2], &Ucov[3]);
		fscanf(fp, "%lf %lf %lf %lf", &Bcon[0],
		       &Bcon[1], &Bcon[2], &Bcon[3]);
		fscanf(fp, "%lf %lf %lf %lf", &Bcov[0],
		       &Bcov[1], &Bcov[2], &Bcov[3]);
		fscanf(fp, "%lf ", &vmin);
		fscanf(fp, "%lf ", &vmax);
		fscanf(fp, "%lf ", &vmin);
		fscanf(fp, "%lf ", &vmax);
		fscanf(fp, "%lf\n", &gdet);

		bias_norm +=
		    dV * gdet * pow(p[UU][i][j] / p[KRHO][i][j] *
				    Thetae_unit, 2.);
		V += dV * gdet;

		/* check accretion rate */
		if (i <= 20)
			dMact += gdet * p[KRHO][i][j] * Ucon[1];
		if (i >= 20 && i < 40)
			Ladv += gdet * p[UU][i][j] * Ucon[1] * Ucov[0];

	}

	bias_norm /= V;
	dMact *= dx[3] * dx[2];
	dMact /= 21.;
	Ladv *= dx[3] * dx[2];
	Ladv /= 21.;
	fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);


	/* done! */

}
void init_sphere_data(char *fname)
{
	FILE *fp;
	double x[4];
	double rp, hp, ph, V, dV, two_temp_gam;
	int i, j, k;

	/* header variables not used except locally */
	double t, a, gam;
	double r, h, gdet;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
	int int_size = sizeof(int);
	int double_size = sizeof(double);

	fp = fopen(fname, "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* get standard HARM header */
    check_scan_error(fread(&t, sizeof(double), 1, fp), 1);
	check_scan_error(fread(&N1, int_size, 1, fp), 1);
    check_scan_error(fread(&N2, int_size, 1, fp), 1);
    check_scan_error(fread(&N3, int_size, 1, fp), 1);
	check_scan_error(fread(&startx[1], double_size, 1, fp), 1);
	check_scan_error(fread(&startx[2], double_size, 1, fp), 1);
	check_scan_error(fread(&startx[3], double_size, 1, fp), 1);
	check_scan_error(fread(&dx[1], double_size, 1, fp), 1);
	check_scan_error(fread(&dx[2], double_size, 1, fp), 1);
	check_scan_error(fread(&dx[3], double_size, 1, fp), 1);
	check_scan_error(fread(&a, double_size, 1, fp), 1);
	check_scan_error(fread(&gam, double_size, 1, fp), 1);
	R0 = 0;
	hslope = 0;
	fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);


	

	/* nominal non-zero values for axisymmetric simulations */
	startx[0] = 0.;
	startx[3] = 0.;
	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = 2. * M_PI;

	fprintf(stderr, "Sim range x1, x2:  %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2]);

	dx[0] = 1.;
	dx[3] = 2. * M_PI;

	/* Allocate storage for all model size dependent variables */
	init_storage();

	two_temp_gam =
	    0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) +
		   gam);
	Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);

	dMact = 0.;
	Ladv = 0.;
	bias_norm = 0.;
	V = 0.;
	dV = dx[1] * dx[2] * dx[3];
	for (k = 0; k < N1 * N2 * N3; k++) {
		// z = 0;
		j = k % N2;
		i = (k - j) / N2;
		check_scan_error(fread(&x[1], double_size, 1, fp), 1);
		//fprintf(stderr, "x1[%d, %d] = %le\n", i, j, x[1]);
		check_scan_error(fread(&x[2], double_size, 1, fp), 1);
		//fprintf(stderr, "x2[%d, %d] = %le\n", i, j, x[2]);
		check_scan_error(fread(&r, double_size, 1, fp), 1);
		//fprintf(stderr, "r[%d, %d] = %le\n", i, j, r);
		check_scan_error(fread(&h, double_size, 1, fp), 1);
		//fprintf(stderr, "h[%d, %d] = %le\n", i, j, h);

		//fprintf(stderr,"Outside X[1] = %le, X[2] = %le, X[3] = %le \n", x[1], x[2], x[3]);

		/* check that we've got the coordinate parameters right */
		bl_coord(x, &rp, &hp);
		if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {
			fprintf(stderr, "grid setup error\n");
			fprintf(stderr, "rp,r,hp,h: %g %g %g %g\n",
				rp, r, hp, h);
			fprintf(stderr, "X1 = %g, X2 = %g\n", x[1], x[2]);
			fprintf(stderr,
				"edit R0, hslope, compile, and continue\n");
			exit(1);
		}

		check_scan_error(fread(&p[KRHO][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "rho[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(KRHO,k)]);
		check_scan_error(fread(&p[UU][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "UU[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(UU,k)]);
		check_scan_error(fread(&p[U1][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "U1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U1,k)]);
		check_scan_error(fread(&p[U2][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "U2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U2,k)]);
		check_scan_error(fread(&p[U3][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "U3[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U3,k)]);
		check_scan_error(fread(&p[B1][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "B1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B1,k)]);
		check_scan_error(fread(&p[B2][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "B2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B2,k)]);
		check_scan_error(fread(&p[B3][i][j], double_size, 1, fp), 1);
		//fprintf(stderr, "B3[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B3,k)]);
		check_scan_error(fread(&Ucon[0], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
		check_scan_error(fread(&Ucon[1], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon1[%d, %d] = %le\n", i, j, Ucon[1]);
		check_scan_error(fread(&Ucon[2], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon2[%d, %d] = %le\n", i, j, Ucon[2]);
		check_scan_error(fread(&Ucon[3], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon3[%d, %d] = %le\n", i, j, Ucon[3]);
		check_scan_error(fread(&Ucov[0], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucov0[%d, %d] = %le\n", i, j, Ucov[0]);
		check_scan_error(fread(&Ucov[1], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucov1[%d, %d] = %le\n", i, j, Ucov[1]);
		check_scan_error(fread(&Ucov[2], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucov2[%d, %d] = %le\n", i, j, Ucov[2]);
		check_scan_error(fread(&Ucov[3], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucov3[%d, %d] = %le\n", i, j, Ucov[3]);
		check_scan_error(fread(&Bcon[0], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[0]);
		check_scan_error(fread(&Bcon[1], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[1]);
		check_scan_error(fread(&Bcon[2], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[2]);
		check_scan_error(fread(&Bcon[3], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[3]);
		check_scan_error(fread(&Bcov[0], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
		check_scan_error(fread(&Bcov[1], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
		check_scan_error(fread(&Bcov[2], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
		check_scan_error(fread(&Bcov[3], double_size, 1, fp), 1);
		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
		check_scan_error(fread(&gdet, double_size, 1, fp), 1);
		V += dV * gdet;
		bias_norm +=
		    dV * gdet * pow(p[UU][i][j] / p[KRHO][i][j] *
				    Thetae_unit, 2.);
		//fprintf(stderr, "rho[%d][%d] = %le, UU[%d][%d] = %le, gdet = %le, Theunit = %le, dV = %le, bias_norm = %le\n", i, j, p[NPRIM_INDEX(KRHO,k)], i, j, p[NPRIM_INDEX(UU,k)], gdet, Thetae_unit, dV, bias_norm);

		//fprintf(stderr, "V[%d][%d] = %le\n", i, j, V);

		/* check accretion rate */
		if (i <= 20)
			dMact += gdet * dx[2] * dx[3] * p[KRHO][i][j] * Ucon[1];
			Ladv += gdet * dx[2] * dx[3] * p[UU][i][j] * Ucon[1] * Ucon[0];
	}
	bias_norm /= V;
	//bias_norm = 1;
	fprintf(stderr, "bias_norm = %le, V = %le\n", bias_norm, V);
	dMact *= dx[3] * dx[2];
	dMact /= 21.;
	Ladv *= dx[3] * dx[2];
	Ladv /= 21.;
	fprintf(stderr, "Ladv = %le, dMact = %le\n", Ladv, dMact);

    fclose(fp);
}
