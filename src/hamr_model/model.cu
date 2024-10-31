#include "../decs.h"
#include "model.h"



__host__ void check_scan_error(int scan_output, int number_of_arguments ) {
	if(scan_output == EOF)fprintf(stderr, "error reading HARM header\n");
	else if(scan_output != number_of_arguments){
		fprintf(stderr, "Not all HARM data could be set\n");
		fprintf(stderr, "Scan Output = %d\n", scan_output);
		fprintf(stderr, "Number of Arguments = %d\n", number_of_arguments);
		exit(1);
	}
}
__host__ void init_storage(void)
{

    p = (double *) malloc(NPRIM * N1 * N2 * N3 * sizeof(double *));

    geom = (struct of_geom *) malloc(N1* N2* sizeof(struct of_geom));
    return;
}

__host__ void init_data(char *fname)
{
	FILE *fp;
	double x[4];
	double rp, hp, V, dV, two_temp_gam;
	int i, j, k;

	/* header variables not used except locally */
	double t, gam;
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
	N3 = 1;
	fprintf(stderr, "Changing N3 to 1, 2D version");

	Rh = 1 + sqrt(1. - a * a);
	fprintf(stderr, "Rh = %le\n", Rh);
	

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
	printf("Thetae_unit = %le\n", Thetae_unit);

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

		check_scan_error(fread(&p[NPRIM_INDEX(KRHO,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "rho[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(KRHO,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(UU,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "UU[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(UU,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(U1,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "U1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U1,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(U2,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "U2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U2,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(U3,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "U3[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U3,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(B1,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "B1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B1,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(B2,k)], double_size, 1, fp), 1);
		//fprintf(stderr, "B2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B2,k)]);
		check_scan_error(fread(&p[NPRIM_INDEX(B3,k)], double_size, 1, fp), 1);
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
		bias_norm += dV * gdet * pow(p[NPRIM_INDEX(UU,k)] / p[NPRIM_INDEX(KRHO,k)] * Thetae_unit, 2.);
        V += dV * gdet;
		
		/* check accretion rate */
		if (i <= 20)
			dMact += gdet * p[NPRIM_INDEX(KRHO,k)] * Ucon[1];
		if (i >= 20 && i < 40)
			Ladv += gdet * p[NPRIM_INDEX(UU,k)] * Ucon[1] * Ucov[0];
	}
	fprintf(stderr, "Bias Norm Final = %le, V Final = %le\n", bias_norm, V);
	fprintf(stderr, "dMact before = %le, Ladv before = %le\n", dMact, Ladv);
	bias_norm /= V;
	fprintf(stderr, "bias_norm = %le, V = %le\n", bias_norm, V);
	dMact *= dx[3] * dx[2];
	dMact /= 21.;
	Ladv *= dx[3] * dx[2];
	Ladv /= 21.;
	fprintf(stderr, "Ladv = %le, dMact = %le\n", Ladv, dMact);

    fclose(fp);
}


/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int GPU_record_criterion(struct of_photon *ph)
{
	const double X1max = log(RMAX);
	/* this is coordinate and simulation
	   specific: stop at large distance */
	if (ph->X[1] > X1max)
		return (1);

	else
		return (0);

}
/*Stop the tracking of the photon if it falls in the bh or is far enough to not be affected.*/
__device__ int GPU_stop_criterion(struct of_photon *ph)
{
	double wmin, X1min, X1max;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */
	X1min = log(d_Rh);	/* this is coordinate-specific; stop
				   at event horizon */
	X1max = log(RMAX);	/* this is coordinate and simulation
				   specific: stop at large distance */


	if (ph->X[1] < X1min)
		return 1;

	if (ph->X[1] > X1max) {
		if (ph->w < wmin) {
			if (GPU_monty_rand() <= 1. / ROULETTE) {
				ph->w *= ROULETTE;
			} else
				ph->w = 0.;
		}
		return 1;
	}

	if (ph->w < wmin) {
		if (GPU_monty_rand() <= 1. / ROULETTE) {
			ph->w *= ROULETTE;
		} else {
			ph->w = 0.;
			return 1;
		}
	}

	return (0);
}

/*Given internal coordinates, X[1], X[2], X[3], we can figure out cell indexes: (i, j, k)*/
__device__ void GPU_Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM])
{

	*i = (int) ((X[1] - d_startx[1]) / d_dx[1] - 0.5 + 1000) - 1000;
	*j = (int) ((X[2] - d_startx[2]) / d_dx[2] - 0.5 + 1000) - 1000;
	if (*i < 0) {
		*i = 0;
		del[1] = 0.;
	} else if (*i > d_N1 - 2) {
		*i = d_N1 - 2;
		del[1] = 1.;
	} else {
		del[1] = (X[1] - ((*i + 0.5) * d_dx[1] + d_startx[1])) / d_dx[1];
	}

	if (*j < 0) {
		*j = 0;
		del[2] = 0.;
	} else if (*j > d_N2 - 2) {
		*j = d_N2 - 2;
		del[2] = 1.;
	} else {
		del[2] = (X[2] - ((*j + 0.5) * d_dx[2] + d_startx[2])) / d_dx[2]; //fractional displacement of the center of the grid cell
	}
	*k = 0;
	del[3] = 0;
	return;
}

/*Given cell indexes i and j, we can figure out internal coordinates X[1], X[2], X[3]*/
__host__ __device__  void coord(int i, int j, double * X)
{
	#ifdef __CUDA_ARCH__
		/* returns zone-centered values for coordinates */
		X[0] = d_startx[0];
		X[1] = d_startx[1] + (i + 0.5) * d_dx[1];
		X[2] = d_startx[2] + (j + 0.5) * d_dx[2];
		X[3] = d_startx[3];
	#else
		/* returns zone-centered values for coordinates */
		X[0] = startx[0];
		X[1] = startx[1] + (i + 0.5) * dx[1];
		X[2] = startx[2] + (j + 0.5) * dx[2];
		X[3] = startx[3];
	#endif
	return;
}
__host__ __device__ void gcov_func(double *X, double gcov[][NDIM])
{
	int k, l;
	double sth, cth, s2, rho2;
	double r, th;
	/* required by broken math.h */
	//void sincos(double th, double *sth, double *cth);

	DLOOP gcov[k][l] = 0.;
	bl_coord(X, &r, &th);
	#ifdef __CUDA_ARCH__
	double bhspin = d_a;
	#else
	double bhspin = a;
	#endif
	sth = sin(th);
	cth = cos(th);
	s2 = sth * sth;
	rho2 = r * r + bhspin * bhspin * cth * cth;


	gcov[0][0] = (-1. + 2. * r / rho2);
	gcov[0][1] = (2. * r / rho2) * r;
	gcov[0][3] = (-2. * bhspin * r * s2 / rho2);

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2. * r / rho2) * r * r;
	gcov[1][3] = (-bhspin * s2 * (1. + 2. * r / rho2)) * r;

	gcov[2][2] = rho2 * (M_PI/2) * (M_PI/2);

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] = s2 * (rho2 + bhspin*bhspin * s2 * (1. + 2. * r / rho2));
}

__host__ double dOmega_func(double x2i, double x2f)
{
	double dO;

	dO = 2. * M_PI *
	    (-cos(M_PI * x2f + 0.5 * (1. - hslope) * sin(2 * M_PI * x2f))
	     + cos(M_PI * x2i + 0.5 * (1. - hslope) * sin(2 * M_PI * x2i))
	    );

	return (dO);
}



__host__ __device__ void vofx_matthewcoords(double *X, double *V){
	V[0] = X[0];
	double RTRANS =5000000.;
	double RB = 0.;
	double RADEXP = 1.0;
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	double BRAVO = 0.0;
	double TANGO = 1.0;
	double CHARLIE = 0.0;
	double DELTA = 3.0;
	if (X[1] < Xtrans){
		V[1] = exp(pow(X[1], RADEXP)) + RB;
	}
	else if (X[1] >= Xtrans && X[1]<1.01*Xtrans){
		V[1] = 10.*(X[1] / Xtrans - 1.)*((X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS) +
			(1. - 10.*(X[1] / Xtrans - 1.))*(exp(pow(X[1], RADEXP)) + RB);
	}
	else{
		V[1] = (X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS;
	}
	double A1 = 1. / (1. + pow(CHARLIE*(log(V[1]) / log(10.)), DELTA));
	double A2 = BRAVO*(log(V[1]) / log(10.)) + TANGO;
	double A3 = pow(0.5, 1. - A2);
	double sign = 1.;
	double X_2 =(X[2]+1.0)/2.0;
	double Xc = sqrt(pow(X_2, 2.));

	if (X_2 < 0.0){
		sign = -1.;
	}
	if (X_2 > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X_2 >= 0.5){
		Xc = 1. - Xc;
		V[2] = M_PI - sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	else{
		V[2] = sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	V[3] = X[3];
}

/* return boyer-lindquist coordinate of point */
__host__  __device__ void bl_coord(double * X, double * r, double *th)
{
	double V[4];
	double SINGSMALL = 1.e-20;
  	void (*vofx_function_pointer)(double*, double*);
    vofx_function_pointer = vofx_matthewcoords;
	vofx_function_pointer(X,V);
	// avoid singularity at polar axis
	if (fabs(V[2])<SINGSMALL){
		if (V[2] >= 0.0) V[2] = SINGSMALL;
		if (V[2]<0.0)  V[2] = -SINGSMALL;
	}
	if (fabs(M_PI - V[2]) <SINGSMALL){
		if (V[2] >= M_PI) V[2] = M_PI + SINGSMALL;
		if (V[2]<M_PI)  V[2] = M_PI -  SINGSMALL;
	}
	*r = V[1];
	*th = V[2];
	//*phi = V[3];
	return ;
}