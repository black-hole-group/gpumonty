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
	double Rin = RMIN;
	double Rout = 2 * RMAX;
	#if(EXP_COORDS)
	Rin = log(Rin);
	Rout = log(Rout);
	#endif
	double th_in = 0.;
	double th_out = M_PI;
	double two_temp_gam;
	double r, h;
	double x[4];
	double sphere_radius = SPHERE_RADIUS;
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
	startx[1] = 0.;
    #if(EXP_COORDS)
    startx[1] = Rin;
    #endif
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
__device__ int GPU_stop_criterion(double X1, double * w, curandState * localState)
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
			if (curand_uniform_double(localState)<= 1. / ROULETTE) {
				*w = *w *  ROULETTE;
			} else
				*w = 0.;
		}
		return 1;
	}

	if (*w < wmin) {
		if (curand_uniform_double(localState) <= 1. / ROULETTE) {
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
	gcov[3][3] = r * r * sin(th) * sin(th);

	#if(EXP_COORDS)
		gcov[1][1] *= r * r;
	#endif
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

__host__ __device__ void ijktoX(){

}

__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,
    double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *  d_geom, const double *  d_p)
{
 
    double X[4] = {0.};
    #ifdef __CUDA_ARCH__
    X[1] = d_startx[1] + (i + 0.5) * d_dx[1];
    X[2] = d_startx[2] + (j + 0.5) * d_dx[2];
    X[3] = d_startx[3] + (k + 0.5) * d_dx[3];
    #else
    X[1] = startx[1] + (i + 0.5) * dx[1];
    X[2] = startx[2] + (j + 0.5) * dx[2];
    X[3] = startx[3] + (k + 0.5) * dx[3];
    #endif

    double gcov[NDIM][NDIM];
    gcov_func(X, gcov);

    double Ucov[4] = {0.};
    double Bcov[4] = {0.};

    GPU_get_fluid_params(X, gcov, Ne, Thetae, B, Ucon, Ucov, Bcon, Bcov);
}


__host__ __device__ void GPU_get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
    double *Thetae, double *B, double Ucon[NDIM],
    double Ucov[NDIM], double Bcon[NDIM],
    double Bcov[NDIM])
{

    #ifdef __CUDA_ARCH__
    if (X[1] < d_startx[1] || X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {
        *Ne = 0;
        *B = 0;
        *Thetae = 0;
        return;
    }
    #else
    if (X[1] < startx[1] || X[1] > stopx[1] || X[2] < startx[2] || X[2] > stopx[2]) {
        *Ne = 0;
        *B = 0;
        *Thetae = 0;
        return;
    }
    #endif

    double r,th;
    bl_coord(X, &r, &th);

    if(r > SPHERE_RADIUS){
        *Ne = 0.;
        *Thetae = 0.;
        *B = 0.;
        return;
    }

    *Ne = NE_VALUE;
    *B = B_VALUE;
    *Thetae = THETAE_VALUE;

    Ucon[0] = 1.;
    Ucon[1] = 0.;
    Ucon[2] = 0.;
    Ucon[3] = 0.;

    Bcon[0] = 0.;
    Bcon[1] = B_VALUE * cos(th)/B_UNIT;
    Bcon[2] = -B_VALUE * sin(th)/(r + 1.e-8) / B_UNIT;
    Bcon[3] = 0.;

    #if(EXP_COORDS)
    Bcon[1] /= r;
    #endif

    lower(Ucon, gcov, Ucov);
    lower(Bcon, gcov, Bcov);
    return;
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