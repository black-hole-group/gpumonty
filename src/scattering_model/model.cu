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



// __host__ void init_data(char *fname)
// {
// 	FILE *fp;
// 	double x[4];
// 	double rp, hp, V, dV, two_temp_gam;
// 	int i, j, k;

// 	/* header variables not used except locally */
// 	double t, a, gam;
// 	double r, h, gdet;
// 	double Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
// 	int int_size = sizeof(int);
// 	int double_size = sizeof(double);

// 	fp = fopen(fname, "r");

// 	if (fp == NULL) {
// 		fprintf(stderr, "can't open sim data file\n");
// 		exit(1);
// 	} else {
// 		fprintf(stderr, "successfully opened %s\n", fname);
// 	}

// 	/* get standard HARM header */
//     check_scan_error(fread(&t, sizeof(double), 1, fp), 1);
// 	check_scan_error(fread(&N1, int_size, 1, fp), 1);
//     check_scan_error(fread(&N2, int_size, 1, fp), 1);
//     check_scan_error(fread(&N3, int_size, 1, fp), 1);
// 	check_scan_error(fread(&startx[1], double_size, 1, fp), 1);
// 	check_scan_error(fread(&startx[2], double_size, 1, fp), 1);
// 	check_scan_error(fread(&startx[3], double_size, 1, fp), 1);
// 	check_scan_error(fread(&dx[1], double_size, 1, fp), 1);
// 	check_scan_error(fread(&dx[2], double_size, 1, fp), 1);
// 	check_scan_error(fread(&dx[3], double_size, 1, fp), 1);
// 	check_scan_error(fread(&a, double_size, 1, fp), 1);
// 	check_scan_error(fread(&gam, double_size, 1, fp), 1);
// 	R0 = 0;
// 	hslope = 0;
// 	fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);


	

// 	/* nominal non-zero values for axisymmetric simulations */
// 	startx[0] = 0.;
// 	startx[3] = 0.;
// 	stopx[0] = 1.;
// 	stopx[1] = startx[1] + N1 * dx[1];
// 	stopx[2] = startx[2] + N2 * dx[2];
// 	stopx[3] = 2. * M_PI;

// 	fprintf(stderr, "Sim range x1, x2:  %g %g, %g %g\n", startx[1],
// 		stopx[1], startx[2], stopx[2]);

// 	dx[0] = 1.;
// 	dx[3] = 2. * M_PI;

// 	/* Allocate storage for all model size dependent variables */
// 	init_storage();

// 	two_temp_gam =
// 	    0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) +
// 		   gam);
// 	Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);
// 	printf("Thetae_unit = %le\n", Thetae_unit);

// 	dMact = 0.;
// 	Ladv = 0.;
// 	bias_norm = 0.;
// 	V = 0.;
// 	dV = dx[1] * dx[2] * dx[3];
// 	for (k = 0; k < N1 * N2 * N3; k++) {
// 		// z = 0;
// 		j = k % N2;
// 		i = (k - j) / N2;
// 		check_scan_error(fread(&x[1], double_size, 1, fp), 1);
// 		//fprintf(stderr, "x1[%d, %d] = %le\n", i, j, x[1]);
// 		check_scan_error(fread(&x[2], double_size, 1, fp), 1);
// 		//fprintf(stderr, "x2[%d, %d] = %le\n", i, j, x[2]);
// 		check_scan_error(fread(&r, double_size, 1, fp), 1);
// 		//fprintf(stderr, "r[%d, %d] = %le\n", i, j, r);
// 		check_scan_error(fread(&h, double_size, 1, fp), 1);
// 		//fprintf(stderr, "h[%d, %d] = %le\n", i, j, h);

// 		//fprintf(stderr,"Outside X[1] = %le, X[2] = %le, X[3] = %le \n", x[1], x[2], x[3]);

// 		/* check that we've got the coordinate parameters right */
// 		bl_coord(x, &rp, &hp);
// 		if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {
// 			fprintf(stderr, "grid setup error\n");
// 			fprintf(stderr, "rp,r,hp,h: %g %g %g %g\n",
// 				rp, r, hp, h);
// 			fprintf(stderr, "X1 = %g, X2 = %g\n", x[1], x[2]);
// 			fprintf(stderr,
// 				"edit R0, hslope, compile, and continue\n");
// 			exit(1);
// 		}

// 		check_scan_error(fread(&p[NPRIM_INDEX(KRHO,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "rho[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(KRHO,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(UU,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "UU[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(UU,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(U1,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "U1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U1,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(U2,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "U2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U2,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(U3,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "U3[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(U3,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(B1,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "B1[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B1,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(B2,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "B2[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B2,k)]);
// 		check_scan_error(fread(&p[NPRIM_INDEX(B3,k)], double_size, 1, fp), 1);
// 		//fprintf(stderr, "B3[%d, %d] = %le\n", i, j, p[NPRIM_INDEX(B3,k)]);
// 		check_scan_error(fread(&Ucon[0], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
// 		check_scan_error(fread(&Ucon[1], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon1[%d, %d] = %le\n", i, j, Ucon[1]);
// 		check_scan_error(fread(&Ucon[2], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon2[%d, %d] = %le\n", i, j, Ucon[2]);
// 		check_scan_error(fread(&Ucon[3], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon3[%d, %d] = %le\n", i, j, Ucon[3]);
// 		check_scan_error(fread(&Ucov[0], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucov0[%d, %d] = %le\n", i, j, Ucov[0]);
// 		check_scan_error(fread(&Ucov[1], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucov1[%d, %d] = %le\n", i, j, Ucov[1]);
// 		check_scan_error(fread(&Ucov[2], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucov2[%d, %d] = %le\n", i, j, Ucov[2]);
// 		check_scan_error(fread(&Ucov[3], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucov3[%d, %d] = %le\n", i, j, Ucov[3]);
// 		check_scan_error(fread(&Bcon[0], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[0]);
// 		check_scan_error(fread(&Bcon[1], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[1]);
// 		check_scan_error(fread(&Bcon[2], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[2]);
// 		check_scan_error(fread(&Bcon[3], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Bcon[3]);
// 		check_scan_error(fread(&Bcov[0], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
// 		check_scan_error(fread(&Bcov[1], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
// 		check_scan_error(fread(&Bcov[2], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
// 		check_scan_error(fread(&Bcov[3], double_size, 1, fp), 1);
// 		//fprintf(stderr, "Ucon0[%d, %d] = %le\n", i, j, Ucon[0]);
// 		check_scan_error(fread(&gdet, double_size, 1, fp), 1);
// 		// if(p[NPRIM_INDEX(KRHO,k)] != 0){
// 		// 	V += dV * gdet;
// 		// 	bias_norm +=
// 		//     dV * gdet * pow((gam - 1)* p[NPRIM_INDEX(UU,k)] / p[NPRIM_INDEX(KRHO,k)], 2.);
// 		// }
// 		V += dV * gdet;
// 		bias_norm += dV * gdet * pow(p[NPRIM_INDEX(UU,k)]/ p[NPRIM_INDEX(KRHO,k)] * Thetae_unit, 2.);
// 		/* check accretion rate */
// 		if (i <= 20)
// 			dMact += gdet * dx[2] * dx[3] * p[NPRIM_INDEX(KRHO,k)] * Ucon[1];
// 			Ladv += gdet * dx[2] * dx[3] * p[NPRIM_INDEX(UU,k)] * Ucon[1] * Ucon[0];
// 	}
// 	bias_norm /= V;
// 	//bias_norm = V/bias_norm;
// 	bias_norm = 0.0/0.0; //producing a nan
// 	fprintf(stderr, "bias_norm = %le, V = %le\n", bias_norm, V);
// 	dMact *= dx[3] * dx[2];
// 	dMact /= 21.;
// 	Ladv *= dx[3] * dx[2];
// 	Ladv /= 21.;
// 	fprintf(stderr, "Ladv = %le, dMact = %le\n", Ladv, dMact);

//     fclose(fp);
// }

__host__ void init_data(char *fname)
{

	/*Resolution*/
	N1 = 1024;
	N2 = 128;
	N3 = 1;
	
	double Rin = 0.01/L_UNIT;
	double Rout = 1.e5/L_UNIT;
	#if(exponential_coordinates)
	double Xin = log(Rin);
	double Xout = log(Rout);
	#else
	double Xin = Rin;
	double Xout = Rout;
	#endif
	double sphere_radius = 1./ L_UNIT;

	#if(exponential_coordinates)
	double sphere_x = log(sphere_radius);
	#else
	double sphere_x = sphere_radius;
	#endif
	//This way sphere_r index i = 10;
	int r_index = 200;
	Xin = (r_index * Xout/N1 - sphere_x) * N1/(r_index - N1);

	double th_in = 0.0001;
	double th_out = M_PI;
	double two_temp_gam;
	double r, h;
	double x[4];
	double Ne_value, B_value, thetae_value;
	int i,j,k;

	/*sphere parameters*/
	gam = 13./9.;
	a = 0.;
	Ne_value = 1.e20/NE_UNIT; /*in 1/cm^3*/
	B_value = 1./B_UNIT; /*in G*/
	thetae_value = 4.;

	/*grid parameters*/
	dx[1] = (Xout - Xin)/N1;
	dx[2] = (th_out - th_in)/N2;
	dx[3] =  2 * M_PI;
	startx[0] = 0.;
	startx[1] = Xin;
	startx[2] = th_in;
	startx[3] = 0.;
	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];
	R0 = 0;
	hslope = 0;
	
	fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);
	fprintf(stderr, "startX (%le, %le), stopX(%le, %le)\n", startx[1], startx[2], stopx[1], stopx[2]);

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
	double tau = 1.e-4;
	double thetae = 4.;
	for (k = 0; k < N1 * N2 * N3; k++) {
		// z = 0;
		j = k % N2;
		i = (k - j) / N2;
		x[1] = startx[1] + i * dx[1];
		x[2] = startx[2] + j * dx[2];
		bl_coord(x, &r, &h);
		if(k == 0){
			fprintf(stderr, "x1 = %le, x2 = %le\n", x[1], x[2]);
			printf("startx[1] = %le, startx[2] = %le\n", startx[1], startx[2]);
		}
		p[NPRIM_INDEX(KRHO,k)] = tau/(((Rout) - (sphere_radius))* L_UNIT * SIGMA_THOMSON) * 1/NE_UNIT;
		p[NPRIM_INDEX(UU,k)] = 1/Thetae_unit* thetae_value * p[NPRIM_INDEX(KRHO,k)];
		p[NPRIM_INDEX(B1,k)] = 0.;
		p[NPRIM_INDEX(B2,k)] = 0.;
		p[NPRIM_INDEX(B3,k)] = 0.;
		p[NPRIM_INDEX(U1,k)] = 0.;
		p[NPRIM_INDEX(U2,k)] = 0.;
		p[NPRIM_INDEX(U3,k)] = 0.;
	}
	bias_norm = 1.; 
	fprintf(stderr, "bias_norm = %le\n", bias_norm);
}



/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int GPU_record_criterion(struct of_photon *ph)
{
	#if(exponential_coordinates)
	const double X1max = log(RMAX);
	#else
	const double X1max = RMAX;
	#endif
	/* this is coordinate and simulation
	   specific: stop at large distance */
	//printf("X[1] coord = %le, X1max = %le\n", ph->X[1], X1max);
	if (ph->X[1] > X1max)
		return (1);

	else
		return (0);

}
/*Stop the tracking of the photon if it falls in the bh or is far enough to not be affected.*/
__device__ int GPU_stop_criterion(struct of_photon *ph)
{
	double wmin, X1max, X1min;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */
	#if(exponential_coordinates)
	X1min = log(RMIN);
	X1max = log(RMAX);
	#else
	X1min = RMIN;
	X1max = RMAX;	/* this is coordinate and simulation specific: stop at large distance */
	#endif				   

	//printf("X1: %le, X1max:%le\n", ph->X[1], X1max);
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

	*i = (int) ((X[1] - d_startx[1]) / d_dx[1] + 1000) - 1000;
	*j = (int) ((X[2] - d_startx[2]) / d_dx[2] + 1000) - 1000;
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
__host__ __device__ void coord(int i, int j, double *X)
{
	#ifdef __CUDA_ARCH__
		/* returns zone-centered values for coordinates */
		X[0] = d_startx[0];
		X[1] = d_startx[1] + (i) * d_dx[1];
		X[2] = d_startx[2] + (j) * d_dx[2];
		X[3] = d_startx[3];
	#else
		/* returns zone-centered values for coordinates */
		X[0] = startx[0];
		X[1] = startx[1] + (i ) * dx[1];
		X[2] = startx[2] + (j ) * dx[2];
		X[3] = startx[3];
	#endif


	return;
}
__host__ __device__ void gcov_func(double *X, double gcov[][NDIM])
{
	int k, l;
	double r,th;
	/* required by broken math.h */
	//void sincos(double th, double *sth, double *cth);
	bl_coord(X, &r, &th);
	DLOOP gcov[k][l] = 0.;
	/*Flat space in spherical coordinates for the test*/							
	gcov[0][0] = -1.;
	gcov[1][1] = 1.;
	gcov[2][2] = r * r;
	gcov[3][3] = r * r * sin(th) * sin(th);

	#if(exponential_coordinates)
		gcov[1][1] = r * r;
	#endif

	//if gcov is inf or is nan, print out the coordinates
	if(isnan(gcov[0][0]) || isnan(gcov[1][1]) || isnan(gcov[2][2]) || isnan(gcov[3][3])){
		printf("Inside gcov_func, NaN quantity!\n");
		printf("r = %le, th = %le\n", r, th);
		printf("gcov[0][0] = %le, gcov[1][1] = %le, gcov[2][2] = %le, gcov[3][3] = %le\n", gcov[0][0], gcov[1][1], gcov[2][2], gcov[3][3]);
	}
	if(isinf(gcov[0][0]) || isinf(gcov[1][1]) || isinf(gcov[2][2]) || isinf(gcov[3][3])){
		printf("Inside gcov_func, infinity quantity\n");
		printf("r = %le, th = %le\n", r, th);
		printf("gcov[0][0] = %le, gcov[1][1] = %le, gcov[2][2] = %le, gcov[3][3] = %le\n", gcov[0][0], gcov[1][1], gcov[2][2], gcov[3][3]);
	}

}

__host__ double dOmega_func(double x2i, double x2f)
{
	double dO;

	dO = 2. * M_PI * (-cos( x2f)+ cos(x2i));

	return (dO);
}

/* return boyer-lindquist coordinate of point */
__host__ __device__ void bl_coord(double *X, double *r, double *th)
{
	
	#if(exponential_coordinates)
		*r = exp(X[1]);
		*th = X[2];
	#else
		*r = X[1];
		*th = X[2];
	#endif
	return;
}