#include "../decs.h"
#include "model.h"

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
    if (fscanf(fp, "%lf", &t) != 1 ||
        fscanf(fp, "%d", &N1) != 1 ||
        fscanf(fp, "%d", &N2) != 1 ||
        fscanf(fp, "%lf", &startx[1]) != 1 ||
        fscanf(fp, "%lf", &startx[2]) != 1 ||
        fscanf(fp, "%lf", &dx[1]) != 1 ||
        fscanf(fp, "%lf", &dx[2]) != 1 ||
        fscanf(fp, "%lf", &tf) != 1 ||
        fscanf(fp, "%d", &nstep) != 1 ||
        fscanf(fp, "%lf", &a) != 1 ||
        fscanf(fp, "%lf", &gam) != 1 ||
        fscanf(fp, "%lf", &cour) != 1 ||
        fscanf(fp, "%lf", &DTd) != 1 ||
        fscanf(fp, "%lf", &DTl) != 1 ||
        fscanf(fp, "%lf", &DTi) != 1 ||
        fscanf(fp, "%d", &DTr) != 1 ||
        fscanf(fp, "%d", &dump_cnt) != 1 ||
        fscanf(fp, "%d", &image_cnt) != 1 ||
        fscanf(fp, "%d", &rdump_cnt) != 1 ||
        fscanf(fp, "%lf", &dt) != 1 ||
        fscanf(fp, "%d", &lim) != 1 ||
        fscanf(fp, "%d", &failed) != 1 ||
        fscanf(fp, "%lf", &Rin) != 1 ||
        fscanf(fp, "%lf", &Rout) != 1 ||
        fscanf(fp, "%lf", &hslope) != 1 ||
        fscanf(fp, "%lf", &R0) != 1) {
        fprintf(stderr, "Error reading HARM header\n");
        fclose(fp);
        exit(1);
    }

    N3 = 1;
    fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);
    fprintf(stderr, "hslope = %le\n", hslope);

    startx[0] = 0.;
    startx[3] = 0.;
    stopx[0] = 1.;
    stopx[1] = startx[1] + N1 * dx[1];
    stopx[2] = startx[2] + N2 * dx[2];
    stopx[3] = 2. * M_PI;

    fprintf(stderr, "Sim range x1, x2:  %g %g, %g %g\n", startx[1], stopx[1], startx[2], stopx[2]);

    dx[0] = 1.;
    dx[3] = 2. * M_PI;

    /* Allocate storage for all model size dependent variables */
    init_storage();

    two_temp_gam = 0.5 * ((1. + 2. / 3. * (TP_OVER_TE + 1.) / (TP_OVER_TE + 2.)) + gam);
    Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + TP_OVER_TE);
    dMact = 0.;
    Ladv = 0.;
    bias_norm = 0.;
    V = 0.;
    dV = dx[1] * dx[2] * dx[3];

    for (k = 0; k < N1 * N2; k++) {
        j = k % N2;
        i = (k - j) / N2;

        if (fscanf(fp, "%lf %lf %lf %lf", &x[1], &x[2], &r, &h) != 4) {
            fprintf(stderr, "Error reading coordinates\n");
            fclose(fp);
            exit(1);
        }

        bl_coord(x, &rp, &hp);
        if (fabs(rp - r) > 1.e-5 * rp || fabs(hp - h) > 1.e-5) {
            fprintf(stderr, "grid setup error\n");
            fprintf(stderr, "rp,r,hp,h: %g %g %g %g\n", rp, r, hp, h);
            fclose(fp);
            exit(1);
        }

        if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf", 
                   &p[NPRIM_INDEX(KRHO,k)],
                   &p[NPRIM_INDEX(UU,k)],
                   &p[NPRIM_INDEX(U1,k)],
                   &p[NPRIM_INDEX(U2,k)],
                   &p[NPRIM_INDEX(U3,k)],
                   &p[NPRIM_INDEX(B1,k)],
                   &p[NPRIM_INDEX(B2,k)],
                   &p[NPRIM_INDEX(B3,k)]) != 8) {
            fprintf(stderr, "Error reading primitive variables\n");
            fclose(fp);
            exit(1);
        }

        if (fscanf(fp, "%lf", &divb) != 1 ||
            fscanf(fp, "%lf %lf %lf %lf", &Ucon[0], &Ucon[1], &Ucon[2], &Ucon[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Ucov[0], &Ucov[1], &Ucov[2], &Ucov[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Bcon[0], &Bcon[1], &Bcon[2], &Bcon[3]) != 4 ||
            fscanf(fp, "%lf %lf %lf %lf", &Bcov[0], &Bcov[1], &Bcov[2], &Bcov[3]) != 4 ||
            fscanf(fp, "%lf", &vmin) != 1 ||
            fscanf(fp, "%lf", &vmax) != 1 ||
            fscanf(fp, "%lf", &vmin) != 1 ||
            fscanf(fp, "%lf", &vmax) != 1 ||
            fscanf(fp, "%lf", &gdet) != 1) {
            fprintf(stderr, "Error reading other data\n");
            fclose(fp);
            exit(1);
        }

        bias_norm += dV * gdet * pow(p[NPRIM_INDEX(UU,k)] / p[NPRIM_INDEX(KRHO,k)] * Thetae_unit, 2.);
        V += dV * gdet;

        if (i <= 20)
            dMact += gdet * p[NPRIM_INDEX(KRHO,k)] * Ucon[1];
        if (i >= 20 && i < 40)
            Ladv += gdet * p[NPRIM_INDEX(UU,k)] * Ucon[1] * Ucov[0];
    }

    bias_norm /= V;
    dMact *= dx[3] * dx[2];
    dMact /= 21.;
    Ladv *= dx[3] * dx[2];
    Ladv /= 21.;
    fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);

    fclose(fp);
}



/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int GPU_record_criterion(struct of_photon *ph)
{
	const double X1max = log(RMAX);
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
	#if(HAMR3D)
	*k= (int) ((X[3] - d_startx[3]) / d_dx[3] - 0.5 + 1000) - 1000;
	if (*k < 0) {
		*k = 0;
		del[3] = 0.;
	} else if (*k > d_N3 - 2) {
		*k = d_N3 - 2;
		del[3] = 1.;
	} else {
		del[3] = (X[3] - ((*k + 0.5) * d_dx[3] + d_startx[3])) / d_dx[3]; //fractional displacement of the center of the grid cell
	}
	#endif
	return;
}

/*Given cell indexes i and j, we can figure out internal coordinates X[1], X[2], X[3]*/
__host__ __device__ void coord(int i, int j, double *X)
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
	double tfac, rfac, hfac, pfac;
	/* required by broken math.h */
	//void sincos(double th, double *sth, double *cth);

	DLOOP gcov[k][l] = 0.;
	bl_coord(X, &r, &th);
	#ifdef __CUDA_ARCH__
	double bhspin = d_a;
	double radius0 = d_R0;
	double theta_slope = d_hslope;
	#else
	double bhspin = a;
	double radius0 = R0;
	double theta_slope = hslope;
	#endif
	//sincos(th, &sth, &cth);
	sth = sin(th);
	cth = cos(th);
	sth = fabs(sth) + SMALL;
	s2 = sth * sth;
	rho2 = r * r + bhspin * bhspin * cth * cth;

	/* transformation for Kerr-Schild -> modified Kerr-Schild */
	tfac = 1.;
	rfac = r - radius0;
	hfac = M_PI + (1. - theta_slope) * M_PI * cos(2. * M_PI * X[2]);
	pfac = 1.;

	#if(HAMR)
	tfac = 1.;
	rfac = 1.;
	hfac = 1.;
	pfac = 1.;
	#endif

	gcov[0][0] = (-1. + 2. * r / rho2) * tfac * tfac;
	gcov[0][1] = (2. * r / rho2) * tfac * rfac;
	gcov[0][3] = (-2. * bhspin * r * s2 / rho2) * tfac * pfac;

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2. * r / rho2) * rfac * rfac;
	gcov[1][3] = (-bhspin * s2 * (1. + 2. * r / rho2)) * rfac * pfac;

	gcov[2][2] = rho2 * hfac * hfac;

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] =
	    s2 * (rho2 + bhspin*bhspin * s2 * (1. + 2. * r / rho2)) * pfac * pfac;
}