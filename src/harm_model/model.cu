#include "../decs.h"
#include "model.h"
#include "../utils.h"
#include "../metrics.h"

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
	Rh = 1 + sqrt(1. - a * a);
	fprintf(stderr, "Rh = %le\n", Rh);
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
__device__ int GPU_stop_criterion(struct of_photon *ph, curandState localState)
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
			if (curand_uniform_double(&localState)<= 1. / ROULETTE) {
				ph->w *= ROULETTE;
			} else
				ph->w = 0.;
		}
		return 1;
	}

	if (ph->w < wmin) {
		if (curand_uniform_double(&localState) <= 1. / ROULETTE) {
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

__host__ double dOmega_func(double x2i, double x2f)
{
	double dO;

	dO = 2. * M_PI *
	    (-cos(M_PI * x2f + 0.5 * (1. - hslope) * sin(2 * M_PI * x2f))
	     + cos(M_PI * x2i + 0.5 * (1. - hslope) * sin(2 * M_PI * x2i))
	    );

	return (dO);
}

/* return boyer-lindquist coordinate of point */
__host__ __device__ void bl_coord(double *X, double *r, double *th)
{
  #ifdef __CUDA_ARCH__
  double theta_slope = d_hslope;
  #else
  double theta_slope = hslope;
  #endif
	//fprintf(stderr,"X[1] = %le, X[2] = %le, X[3] = %le \n", X[1], X[2], X[3]);
	*r = exp(X[1]);
	*th = M_PI * X[2] + ((1. - theta_slope) / 2.) * sin(2. * M_PI * X[2]);

	return;
}


__host__ __device__ void get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
    double Ucon[NDIM], double Bcon[NDIM], struct of_geom * d_geom, double * d_p)
{
    int l, m;
    double Ucov[NDIM], Bcov[NDIM];
    double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
    #ifdef __CUDA_ARCH__
    double thetaeUnit = d_thetae_unit;
    #else
    double thetaeUnit = Thetae_unit;

    #endif

    *Ne = d_p[NPRIM_INDEX3D(KRHO, i, j, k)] * NE_UNIT;
    *Thetae = d_p[NPRIM_INDEX3D(UU, i, j, k)] / (*Ne) * NE_UNIT * thetaeUnit;

    Bp[1] = d_p[NPRIM_INDEX3D(B1, i, j, k)];
    Bp[2] = d_p[NPRIM_INDEX3D(B2, i, j, k)];
    Bp[3] = d_p[NPRIM_INDEX3D(B3, i, j, k)];

    Vcon[1] = d_p[NPRIM_INDEX3D(U1, i, j, k)];
    Vcon[2] = d_p[NPRIM_INDEX3D(U2, i, j, k)];
    Vcon[3] = d_p[NPRIM_INDEX3D(U3, i, j, k)];

    /* Get Ucov */
    VdotV = 0.;
    for (l = 1; l < NDIM; l++)
    for (m = 1; m < NDIM; m++)
        VdotV += d_geom[SPATIAL_INDEX2D(i,j)].gcov[l][m] * Vcon[l] * Vcon[m];
    Vfac = sqrt(-1. / d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][0] * (1. + fabs(VdotV)));
    Ucon[0] = -Vfac * d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][0];
    for (l = 1; l < NDIM; l++){
    Ucon[l] = Vcon[l] - Vfac * d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][l];
    //printf("Ucon[%d] = %le, Vcon[%d] = %le, Vfac = %le, geom[0][%d] = %le\n", l, Ucon[l], l, Vcon[l], Vfac, l, d_geom[SPATIAL_INDEX2D(i,j)].gcon[0][l]);
    }
    lower(Ucon, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Ucov);
    /* Get B and Bcov */
    UdotBp = 0.;
    for (l = 1; l < NDIM; l++)
    UdotBp += Ucov[l] * Bp[l];
    Bcon[0] = UdotBp;
    for (l = 1; l < NDIM; l++){
    Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
    }
    lower(Bcon, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Bcov);
    *B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
    Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;


    #ifdef SCATTERING_TEST
    *Ne = 1.e-4/(SIGMA_THOMSON * (1.e5 - 1.));
    *Thetae = 4.;
    *B = 0;
    return;
    #endif

    if (isnan(*B)){
    printf("i = %d, j = %d, k = %d\n", i, j, k);
    printf( "VdotV = %le\n", VdotV);
    printf( "Vfac = %lf\n", Vfac);
    for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcon[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcon[a][b]);
    for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcov[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcov[a][b]);
    printf( "Thetae: %lf\n", *Thetae);
    printf( "Ne: %lf\n", *Ne);
    printf( "Bp: %lf, %lf, %lf\n", Bp[1], Bp[2], Bp[3]);
    printf( "Vcon: %lf, %lf, %lf\n", Vcon[1], Vcon[2], Vcon[3]);
    printf( "Bcon: %lf, %lf, %lf, %lf\n Bcov: %lf, %lf, %lf %lf\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3], Bcov[0], Bcov[1], Bcov[2], Bcov[3]);
    printf( "Ucon: %lf, %lf, %lf, %lf\n Ucov: %lf, %lf, %lf %lf\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3], Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
    }
}


__device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
    double *Thetae, double *B, double Ucon[NDIM],
    double Ucov[NDIM], double Bcon[NDIM],
    double Bcov[NDIM], double * d_p)
{
    int i, j, k;
    double del[NDIM];
    double rho, uu;
    double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
    double gcon[NDIM][NDIM], coeff[8];

    //checks if it's within the grid
    if (X[1] < d_startx[1] ||
    X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {

    *Ne = 0.;

    return;
    }

    // Finds out i and j index as well as fraction displacement del from the coordinates X[1], X[2], X[3]
    //Xtoij(X, &i, &j, del);
    GPU_Xtoijk(X, &i, &j, &k, del);
    //Xtoijk(X, &i, &j, &k, del);

    //Calculate the coeficient of displacement
    coeff[0] = (1. - del[1]) * (1. - del[2]) * (1. - del[3]);
    coeff[1] = (1. - del[1]) * (1. - del[2]) * del[3];
    coeff[2] = (1. - del[1]) * del[2] * del[3];
    coeff[3] = del[1] * del[2] * del[3];
    coeff[4] = (1. - del[1]) * del[2] * (1. - del[3]);
    coeff[5] = del[1] * (1. - del[2]) * (1. - del[3]);
    coeff[6] = del[1] * (1. - del[2]) * del[3];
    coeff[7] = del[1] * del[2] * (1. - del[3]);



    //interpolate based on the displacement
    rho = GPU_interp_scalar(d_p, KRHO, i, j, k, coeff);
    uu = GPU_interp_scalar(d_p, UU, i, j, k, coeff);
    *Ne = rho * NE_UNIT;
    *Thetae = uu / rho * d_thetae_unit;

    Bp[1] = GPU_interp_scalar(d_p, B1, i, j, k, coeff);
    Bp[2] = GPU_interp_scalar(d_p, B2, i, j, k, coeff);
    Bp[3] = GPU_interp_scalar(d_p, B3, i, j, k, coeff);

    Vcon[1] = GPU_interp_scalar(d_p, U1, i, j, k, coeff);
    Vcon[2] = GPU_interp_scalar(d_p, U2, i, j, k, coeff);
    Vcon[3] = GPU_interp_scalar(d_p, U3, i, j, k, coeff);

    gcon_func(X, gcov, gcon);

    /* Get Ucov */
    VdotV = 0.;
    for (i = 1; i < NDIM; i++)
    for (j = 1; j < NDIM; j++)
    VdotV += gcov[i][j] * Vcon[i] * Vcon[j];
    Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
    Ucon[0] = -Vfac * gcon[0][0];
    for (i = 1; i < NDIM; i++){
    Ucon[i] = Vcon[i] - Vfac * gcon[0][i];
    }
    lower(Ucon, gcov, Ucov);

    /* Get B and Bcov */
    UdotBp = 0.;
    for (i = 1; i < NDIM; i++)
    UdotBp += Ucov[i] * Bp[i];
    Bcon[0] = UdotBp;
    for (i = 1; i < NDIM; i++)
    Bcon[i] = (Bp[i] + Ucon[i] * UdotBp) / Ucon[0];
    lower(Bcon, gcov, Bcov);

    *B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
    Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;

    #ifdef SCATTERING_TEST
        *Ne = 1.e-4/(SIGMA_THOMSON * (1.e5 - 1.));
        *Thetae = 4.;
        *B = 0;
        return;
    #endif
}



__device__ double GPU_bias_func(double Te, double w, int round_scatt)
{
    double bias, max, avg_num_scatt;
    #if(0)
        max = 0.5 * w / WEIGHT_MIN;
        //bias = Te * Te /(5. *d_max_tau_scatt);
        bias = fmax(1., d_bias_norm * Te * Te/d_max_tau_scatt);

        if (bias > max){
        bias = max;
        }

        return bias;
    #else
        //return 1;
        max = 0.5 * w / WEIGHT_MIN;

        avg_num_scatt = d_N_scatt / (1. * d_N_superph_recorded + 1.);
        bias =
        100. * Te * Te / (d_bias_norm * d_max_tau_scatt *
                (avg_num_scatt + 2));

        //bias = Te * Te/(d_bias_norm * d_max_tau_scatt * 2.);

        if (bias < TP_OVER_TE)
        bias = TP_OVER_TE;
        if (bias > max)
        bias = max;
        //printf("bias = %le, max = %le, avg_num_scatt = %le\n", bias, max, avg_num_scatt);
        return bias / TP_OVER_TE;
    #endif
}

__device__ __forceinline__ double atomicMaxdouble(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}



__device__ void GPU_record_super_photon(struct of_photon *ph , struct of_spectrum* d_spect) {
    double lE, dx2;
    int iE, ix2;

    if (isnan(ph->w) || isnan(ph->E)) {
        printf("record isnan: %g %g\n", ph->w, ph->E);
        return;
    }

	d_max_tau_scatt = atomicMaxdouble(&d_max_tau_scatt, ph->tau_scatt);
	#ifdef HAMR
		dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
		ix2 = ((ph->X[2]) < 0) ? (int)((1 +ph->X[2]) / dx2) : (int)((d_stopx[2] - ph->X[2]) / dx2);
	#else
	    dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
    	ix2 = (ph->X[2] < 0.5 * (d_startx[2] + d_stopx[2])) ? (int)(ph->X[2] / dx2) : (int)((d_stopx[2] - ph->X[2]) / dx2);
	#endif
    if (ix2 < 0 || ix2 >= N_THBINS){
        return;
	}

    // Get energy bin
    lE = log(ph->E);
    iE = (int)((lE - lE0) / dlE + 2.5) - 2;

    if (iE < 0 || iE >= N_EBINS){
	    return;
	}

    atomicAdd(&d_N_superph_recorded, 1);
    //atomicAdd(&d_N_scatt, ph->nscatt);

	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].dNdlE), ph->w);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].dEdlE), ph->w * ph->E);
    atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].tau_abs), ph->w * ph->tau_abs);
    atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].tau_scatt), ph->w * ph->tau_scatt);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X1iav), ph->w * ph->X1i);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X2isq), ph->w * (ph->X2i * ph->X2i));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].X3fsq), ph->w * (ph->X[3] * ph->X[3]));
	// atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].ne0),  ph->w * (ph->ne0));
	// atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].b0), ph->w * (ph->b0));
	// atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].thetae0),ph->w * (ph->thetae0));
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].nscatt),  ph->nscatt);
	atomicAdd(&(d_spect[(ix2 * N_EBINS) + iE].nph), 1);
}