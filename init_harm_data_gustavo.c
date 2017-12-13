
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



/*

I modified the header so that it can read data from harmpi-3d.
New variables to store header data were added as needed.

GS

*/


void init_harm_data(char *fname)
{
	FILE *fp;
	double x[4];
	double rp, hp, phip, V, dV, two_temp_gam, beta_lorentz, tp_over_te;
	int i, j, k, kk;
    double pktot;

	/* header variables not used except locally */
	double t, tf, cour, DTd, DTl, DTi, DTr, dt;
	int nstep, rdump_cnt, lim, failed;
	double r, h, phi, divb, vmin, vmax, gdet;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
    int mpi_ntot[NDIM];

    int mpi_startn[NDIM], coord_plus_mpi_startn[NDIM];
    double fractheta, fracphi;
    int npr, DOKTOT, BL, DTr01, rdump01_cnt, dump_cnt, image_cnt;

    int GN1, GN2, GN3, nx, ny, nz;
    double ti, tj, tk;
    
	fp = fopen(fname, "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* get HARMPI header */
    fscanf(fp, "%lf ", &t);
    fscanf(fp, "%d ", &N1);
    fscanf(fp, "%d ", &N2);
    fscanf(fp, "%d ", &N3);
    fscanf(fp, "%d ", &nx);
    fscanf(fp, "%d ", &ny);
    fscanf(fp, "%d ", &nz);
    fscanf(fp, "%d ", &GN1);
    fscanf(fp, "%d ", &GN2);
    fscanf(fp, "%d ", &GN3);
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
	fscanf(fp, "%d ", &DTr01);
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
    fscanf(fp, "%d ", &npr);
    fscanf(fp, "%d ", &DOKTOT);
    fscanf(fp, "%lf ", &fractheta);
    fscanf(fp, "%lf ", &fracphi);
    fscanf(fp, "%lf ", &rbr);
    fscanf(fp, "%lf ", &npow2);
    fscanf(fp, "%lf ", &cpow2);
    fscanf(fp, "%d ", &BL);

	/* nominal non-zero values for axisymmetric simulations */
	startx[0] = 0.;

	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];

	fprintf(stderr, "Sim range x1, x2, x3:  %g %g, %g %g, %g %g\n", startx[1],
		stopx[1], startx[2], stopx[2], startx[3], stopx[3]);

	dx[0] = 1.;

	/* Allocate storage for all model size dependent variables */
	init_storage();

	/*two_temp_gam =
	    0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) + gam);
	Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);*/

	dMact = 0.;
	Ladv = 0.;
	bias_norm = 0.;
	V = 0.;
	dV = dx[1] * dx[2] * dx[3];

    // loop to read ascii data
	for (kk = 0; kk < N1 * N2; kk++) {
	    for (k = 0; k < N3; k++) {
		    j = kk % N2;
		    i = (kk - j) / N2;
    
            fscanf(fp, "%lf %lf %lf ", &ti, &tj, &tk);

            fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &x[1], &x[2], &x[3], &r, &h, &phi);

            /* check that we've got the coordinate parameters right */
		    bl_coord(x, &rp, &hp, &phip);
		    if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {  // add phi test - GS
			    fprintf(stderr, "grid setup error\n");
			    fprintf(stderr, "rp, r, hp, h, phip, phi: %g %g %g %g %g %g\n", rp, r, hp, h, phip, phi);
			    fprintf(stderr,"edit R0, hslope, compile, and continue\n");
			    exit(1);
		    }

            fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf ",
                &p[KRHO][i][j][k],
                &p[UU][i][j][k], &p[U1][i][j][k], &p[U2][i][j][k], &p[U3][i][j][k],
                &p[B1][i][j][k], &p[B2][i][j][k], &p[B3][i][j][k], &pktot);
    
            fscanf(fp, "%lf", &divb);

            fscanf(fp, "%lf %lf %lf %lf ", &Ucon[0], &Ucon[1], &Ucon[2], &Ucon[3]);
            fscanf(fp, "%lf %lf %lf %lf ", &Ucov[0], &Ucov[1], &Ucov[2], &Ucov[3]);
            fscanf(fp, "%lf %lf %lf %lf ", &Bcon[0], &Bcon[1], &Bcon[2], &Bcon[3]);
            fscanf(fp, "%lf %lf %lf %lf ", &Bcov[0], &Bcov[1], &Bcov[2], &Bcov[3]);

            fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &vmin, &vmax, &vmin, &vmax, &vmin, &vmax);

            fscanf(fp, "%lf\n", &gdet);

            beta_lorentz = sqrt(p[U1][i][j][k]*p[U1][i][j][k] + p[U2][i][j][k]*p[U2][i][j][k] + p[U3][i][j][k]*p[U3][i][j][k]);

            if (beta_lorentz < BETA_LORENTZ_MIN)
                tp_over_te = TP_OVER_TE_DISK;
            else
                tp_over_te = TP_OVER_TE_JET;

            two_temp_gam =  0.5 * ((1. + 2. / 3. * (tp_over_te + 1.) / (tp_over_te + 2.)) + gam);
            Thetae_unit[i][j][k] = (two_temp_gam - 1.) * (MP / ME) / (1. + tp_over_te);

		    bias_norm +=
		        dV * gdet * pow(p[UU][i][j][k] / p[KRHO][i][j][k] * Thetae_unit[i][j][k], 2.);
		    V += dV * gdet;

		    /* check accretion rate */
	    	if (i <= 20)
		    	dMact += gdet * p[KRHO][i][j][k] * Ucon[1];
		    if (i >= 20 && i < 40)
			    Ladv += gdet * p[UU][i][j][k] * Ucon[1] * Ucov[0];
        }
	}

	bias_norm /= V;
	dMact *= dx[3] * dx[2];
	dMact /= 21.;
	Ladv *= dx[3] * dx[2];
	Ladv /= 21.;
	fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);


	/* done! */

}
