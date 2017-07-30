
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


void init_harm_data(char *fname)
{
	FILE *fp;
	double x[4];
	double rp, hp, V, dV, two_temp_gam;
	int i, j, k;
	static const int NDIM=3;

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
	fscanf(fp, "%d ", &DTr);
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
	fscanf(fp, "%d ", &NPR);
	fscanf(fp, "%d ", &DOKTOT);
	fscanf(fp, "%lf ", &fractheta);
	fscanf(fp, "%lf ", &fracphi);
	fscanf(fp, "%lf ", &rbr);
	fscanf(fp, "%lf ", &npow2);
	fscanf(fp, "%lf ", &cpow2);
	fscanf(fp, "%lf ", &BL);

	printf("%lf\n", a);
	printf("%lf\n", BL);




	/* done! */
	fclose(fp);

}
