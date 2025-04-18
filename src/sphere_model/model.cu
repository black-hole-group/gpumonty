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
	double Rin = 1.e-2/L_UNIT;
	double Rout = 2./L_UNIT;
	#if(EXP_COORDS)
	Rin = log(Rin);
	Rout = log(Rout);
	#endif
	double th_in = 0.0001;
	double th_out = M_PI;
	double two_temp_gam;
	double r, h;
	double x[4];
	double sphere_radius = 1./ L_UNIT;
	double Ne_value, B_value, thetae_value;
	int i,j,k;

	/*sphere parameters*/
	gam = 13./9.;
	Ne_value = NE_VALUE/NE_UNIT; /*in 1/cm^3*/
	B_value = B_VALUE/B_UNIT; /*in G*/
	thetae_value = THETAE_VALUE;

	/*grid parameters*/
	dx[1] = (Rout - Rin)/N1;
	dx[2] = (th_out - th_in)/N2;
	dx[3] =  2 * M_PI;
	startx[0] = 0.;
	startx[1] = Rin;
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
	for (k = 0; k < N1 * N2 * N3; k++) {
		// z = 0;
		j = k % N2;
		i = (k - j) / N2;
		x[1] = startx[1] + i * dx[1];
		x[2] = startx[2] + j * dx[2];
		bl_coord(x, &r, &h);

		if(r < sphere_radius){
			p[NPRIM_INDEX(KRHO,k)] = Ne_value;
			p[NPRIM_INDEX(UU,k)] = 1/Thetae_unit* thetae_value * p[NPRIM_INDEX(KRHO,k)];
			#if(EXP_COORDS)
				p[NPRIM_INDEX(B1,k)] = B_value * cos(h)/r ;
			#else
				p[NPRIM_INDEX(B1,k)] = B_value * cos(h);
			#endif
			p[NPRIM_INDEX(B2,k)] = - B_value * sin(h)/r;
		}else{
			p[NPRIM_INDEX(KRHO,k)] = 0.;
			p[NPRIM_INDEX(UU,k)] = 0.;
			p[NPRIM_INDEX(B1,k)] = 0.;
			p[NPRIM_INDEX(B2,k)] = 0.;

		}
		p[NPRIM_INDEX(B3,k)] = 0.;
		p[NPRIM_INDEX(U1,k)] = 0.;
		p[NPRIM_INDEX(U2,k)] = 0.;
		p[NPRIM_INDEX(U3,k)] = 0.;
	}
	bias_norm = 0.0/0.0; //producing a nan so we don't account for scattering
	fprintf(stderr, "bias_norm = %le\n", bias_norm);
}



/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int GPU_record_criterion(double X1)
{
	#if(EXP_COORDS)
	const double X1max = log(RMAX);
	#else
	const double X1max = RMAX;
	#endif
	/* this is coordinate and simulation
	   specific: stop at large distance */
	//printf("X[1] coord = %le, X1max = %le\n", ph->X[1], X1max);
	if (X1 > X1max)
		return (1);

	else
		return (0);


}
/*Stop the tracking of the photon if it falls in the bh or is far enough to not be affected.*/
__device__ int GPU_stop_criterion(double X1, double * w, curandState localState)
{
	double wmin, X1max, X1min;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */
	#if(EXP_COORDS)
	X1min = log(RMIN);
	X1max = log(RMAX);
	#else
	X1min = RMIN;
	X1max = RMAX;	/* this is coordinate and simulation specific: stop at large distance */
	#endif				   

	if (X1 < X1min)
		return 1;

	if (X1 > X1max) {
		if (*w < wmin) {
			if (curand_uniform_double(&localState)<= 1. / ROULETTE) {
				*w = *w *  ROULETTE;
			} else
				*w = 0.;
		}
		return 1;
	}

	if (*w < wmin) {
		if (curand_uniform_double(&localState) <= 1. / ROULETTE) {
			*w = *w * ROULETTE;
		} else {
			*w = 0.;
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
	} else if (*i > N1 - 2) {
		*i = N1 - 2;
		del[1] = 1.;
	} else {
		del[1] = (X[1] - ((*i + 0.5) * d_dx[1] + d_startx[1])) / d_dx[1];
	}

	if (*j < 0) {
		*j = 0;
		del[2] = 0.;
	} else if (*j > N2 - 2) {
		*j = N2 - 2;
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
__host__ __device__ void gcov_func(const double *X, double gcov[][NDIM])
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
	if(th == 0){
		gcov[3][3] = 0.;
	}else{
		gcov[3][3] = r * r * sin(th) * sin(th);
	}

	#if(EXP_COORDS)
		gcov[1][1] = r * r;
	#endif

	//if gcov is inf or is nan, print out the coordinates
	if(isnan(gcov[0][0]) || isnan(gcov[1][1]) || isnan(gcov[2][2]) || isnan(gcov[3][3])){
		printf("Inside gcov_func, NaN quantity!\n");
		printf("r = %le, th = %le, X[1] = %le, X[2] = %le\n", r, th, X[1], X[2]);
		printf("gcov[0][0] = %le, gcov[1][1] = %le, gcov[2][2] = %le, gcov[3][3] = %le\n", gcov[0][0], gcov[1][1], gcov[2][2], gcov[3][3]);
	}
	if(isinf(gcov[0][0]) || isinf(gcov[1][1]) || isinf(gcov[2][2]) || isinf(gcov[3][3])){
		printf("Inside gcov_func, infinity quantity\n");
		printf("r = %le, th = %le, X[1] = %le, X[2] = %le\n", r, th, X[1], X[2]);
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
__host__ __device__ void bl_coord(const double *X, double *r, double *th)
{
	
	#if(EXP_COORDS)
		*r = exp(X[1]);
		*th = X[2];
	#else
		*r = X[1];
		*th = X[2];
	#endif
	return;
}

__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,
    double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *  d_geom, const double *  d_p)
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
    double Bcov[NDIM], cudaTextureObject_t d_p)
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
    // coeff[0] = (1. - del[1]) * (1. - del[2]) * (1. - del[3]);
    // coeff[1] = (1. - del[1]) * (1. - del[2]) * del[3];
    // coeff[2] = (1. - del[1]) * del[2] * del[3];
    // coeff[3] = del[1] * del[2] * del[3];
    // coeff[4] = (1. - del[1]) * del[2] * (1. - del[3]);
    // coeff[5] = del[1] * (1. - del[2]) * (1. - del[3]);
    // coeff[6] = del[1] * (1. - del[2]) * del[3];
    // coeff[7] = del[1] * del[2] * (1. - del[3]);



    //interpolate based on the displacement
    rho = GPU_interp_scalar(d_p, KRHO, i, j, k, del);
    uu = GPU_interp_scalar(d_p, UU, i, j, k, del);
    *Ne = rho * NE_UNIT;
    *Thetae = uu / rho * d_thetae_unit;

    Bp[1] = GPU_interp_scalar(d_p, B1, i, j, k, del);
    Bp[2] = GPU_interp_scalar(d_p, B2, i, j, k, del);
    Bp[3] = GPU_interp_scalar(d_p, B3, i, j, k, del);

    Vcon[1] = GPU_interp_scalar(d_p, U1, i, j, k, del);
    Vcon[2] = GPU_interp_scalar(d_p, U2, i, j, k, del);
    Vcon[3] = GPU_interp_scalar(d_p, U3, i, j, k, del);

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



__device__ void GPU_record_super_photon(struct of_photonSOA ph, struct of_spectrum* d_spect, unsigned long long photon_index) {
    double lE, dx2;
    int iE, ix2, index;

    if (isnan(ph.w[photon_index]) || isnan(ph.E[photon_index])) {
        printf("record isnan: %g %g\n", ph.w[photon_index], ph.E[photon_index]);
        return;
    }

    d_max_tau_scatt = atomicMaxdouble(&d_max_tau_scatt, ph.tau_scatt[photon_index]);

    #ifdef HAMR
        dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
        ix2 = ((ph.X2[photon_index]) < 0) ? (int)((1 + ph.X2[photon_index]) / dx2) : (int)((d_stopx[2] - ph.X2[photon_index]) / dx2);
    #else
        dx2 = (d_stopx[2] - d_startx[2]) / (2.0 * N_THBINS);
        ix2 = (ph.X2[photon_index] < 0.5 * (d_startx[2] + d_stopx[2])) ? (int)(ph.X2[photon_index] / dx2) : (int)((d_stopx[2] - ph.X2[photon_index]) / dx2);
    #endif

    if (ix2 < 0 || ix2 >= N_THBINS) {
        return;
    }

    // Get energy bin
    lE = log(ph.E[photon_index]);
    iE = (int)((lE - lE0) / dlE + 2.5) - 2;

    if (iE < 0 || iE >= N_EBINS) {
        return;
    }

    // Calculate the index once
    index = ix2 * N_EBINS + iE;

    atomicAdd(&d_N_superph_recorded, 1);
    //atomicAdd(&d_N_scatt, ph.nscatt[photon_index]);

    atomicAdd(&(d_spect[index].dNdlE), ph.w[photon_index]);
    atomicAdd(&(d_spect[index].dEdlE), ph.w[photon_index] * ph.E[photon_index]);
    atomicAdd(&(d_spect[index].tau_abs), ph.w[photon_index] * ph.tau_abs[photon_index]);
    atomicAdd(&(d_spect[index].tau_scatt), ph.w[photon_index] * ph.tau_scatt[photon_index]);
    atomicAdd(&(d_spect[index].X1iav), ph.w[photon_index] * ph.X1i[photon_index]);
    atomicAdd(&(d_spect[index].X2isq), ph.w[photon_index] * (ph.X2i[photon_index] * ph.X2i[photon_index]));
    atomicAdd(&(d_spect[index].X3fsq), ph.w[photon_index] * (ph.X3[photon_index] * ph.X3[photon_index]));
    // atomicAdd(&(d_spect[index].ne0),  ph.w[photon_index] * (ph.ne0[photon_index]));
    // atomicAdd(&(d_spect[index].b0), ph.w[photon_index] * (ph.b0[photon_index]));
    // atomicAdd(&(d_spect[index].thetae0), ph.w[photon_index] * (ph.thetae0[photon_index]));
    atomicAdd(&(d_spect[index].nscatt), ph.nscatt[photon_index]);
    atomicAdd(&(d_spect[index].nph), 1);
}