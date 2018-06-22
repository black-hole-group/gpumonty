
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
	}
}

void init_harm_data(char *fname)
{
	FILE *fp;
	double x[4];
	double V, dV, two_temp_gam;
	int i, j, k;

	/* header variables not used except locally */
	double t, tf, cour, DTd, DTl, DTi, dt;
	int nstep, DTr, dump_cnt, image_cnt, rdump_cnt;
	double divb, vmin, vmax, gdet;
	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
	double J ;

	fp = fopen(fname, "r");

	if (fp == NULL) {
		fprintf(stderr, "can't open sim data file\n");
		exit(1);
	} else {
		fprintf(stderr, "successfully opened %s\n", fname);
	}

	/* get standard HARM header */
	check_scan_error(fscanf(fp, "%lf ", &t), 1);
	check_scan_error(fscanf(fp, "%d ", &N1), 1);
	check_scan_error(fscanf(fp, "%d ", &N2), 1);
	check_scan_error(fscanf(fp, "%lf ", &startx[1]), 1);
	check_scan_error(fscanf(fp, "%lf ", &startx[2]), 1);
	check_scan_error(fscanf(fp, "%lf ", &dx[1]), 1);
	check_scan_error(fscanf(fp, "%lf ", &dx[2]), 1);
	check_scan_error(fscanf(fp, "%lf ", &tf), 1);
	check_scan_error(fscanf(fp, "%d ", &nstep), 1);
	check_scan_error(fscanf(fp, "%lf ", &gam), 1);
	check_scan_error(fscanf(fp, "%lf ", &cour), 1);
	check_scan_error(fscanf(fp, "%lf ", &DTd), 1);
	check_scan_error(fscanf(fp, "%lf ", &DTl), 1);
	check_scan_error(fscanf(fp, "%lf ", &DTi), 1);
	check_scan_error(fscanf(fp, "%d ", &DTr), 1);
	check_scan_error(fscanf(fp, "%d ", &dump_cnt), 1);
	check_scan_error(fscanf(fp, "%d ", &image_cnt), 1);
	check_scan_error(fscanf(fp, "%d ", &rdump_cnt), 1);
	check_scan_error(fscanf(fp, "%lf ", &dt), 1);

  /* finish reading out the line */
  fseek(fp, 0, SEEK_SET);
  while ( (i=fgetc(fp)) != '\n' ) ;

  /* not set automatically */
  a = 0.9375;
  Rin = 0.98 * (1. + sqrt(1. - a*a)) ;
  Rout = 40.;
  hslope = 0.3;
  R0 = 0.0;
  fprintf(stderr,"coordinate parameters: Rin,Rout,hslope,R0,dx[1],dx[2]: %g %g %g %g %g %g\n",
          Rin,Rout,hslope,R0,dx[1],dx[2]) ;

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
		check_scan_error(fscanf(fp, "%lf %lf", &x[1], &x[2]), 2);

		check_scan_error(fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf",
		       &p[KRHO][i][j],
		       &p[UU][i][j],
		       &p[U1][i][j], &p[U2][i][j], &p[U3][i][j],
		       &p[B1][i][j], &p[B2][i][j], &p[B3][i][j]), 8);


		check_scan_error(fscanf(fp, "%lf", &divb), 1);

		check_scan_error(fscanf(fp, "%lf %lf %lf %lf",
			&Ucon[0], &Ucon[1], &Ucon[2], &Ucon[3]), 4);
		check_scan_error(fscanf(fp, "%lf %lf %lf %lf",
			&Ucov[0], &Ucov[1], &Ucov[2], &Ucov[3]), 4);
		check_scan_error(fscanf(fp, "%lf %lf %lf %lf",
			&Bcon[0], &Bcon[1], &Bcon[2], &Bcon[3]), 4);
		check_scan_error(fscanf(fp, "%lf %lf %lf %lf",
			&Bcov[0], &Bcov[1], &Bcov[2], &Bcov[3]), 4);

		check_scan_error(fscanf(fp, "%lf ", &vmin), 1);
		check_scan_error(fscanf(fp, "%lf ", &vmax), 1);
		check_scan_error(fscanf(fp, "%lf ", &vmin), 1);
		check_scan_error(fscanf(fp, "%lf ", &vmax), 1);
		check_scan_error(fscanf(fp, "%lf ", &gdet), 1);

	        /* additional stuff: current */
		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf ", &J), 1);

		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf ", &J), 1);
		check_scan_error(fscanf(fp, "%lf\n", &J), 1);

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
