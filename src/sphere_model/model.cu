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

#define MIN(A,B) (A<B?A:B)
__device__ double stepsize(const double X[NDIM], const double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;
	dlx2 = EPS * MIN(X[2], d_stopx[2] - X[2]) / (fabs(K[2]) + SMALL);

	dlx1 = EPS/(fabs(K[1]) + SMALL);
	dlx3 = EPS / (fabs(K[3]) + SMALL);

	idlx1 = 1. / (fabs(dlx1) + SMALL);
	idlx2 = 1. / (fabs(dlx2) + SMALL);
	idlx3 = 1. / (fabs(dlx3) + SMALL);

	dl = 1. / (idlx1 + idlx2 + idlx3);

	return (dl);
}
#undef MIN

__host__ void init_data()
{
	double Rin = RMIN/L_unit;
	double Rout = RMAX/L_unit;

	double th_in = 0.;
	double th_out = M_PI;
	double two_temp_gam;
	double sphere_radius = SPHERE_RADIUS/L_unit;

	gam = 13./9.;

	/*Setting the resolution*/
    // R resolution is high here to properly deal with the sphere edge. The higher it is, the higher the anti-aliasing quality.
    N1 = 30000; 
	N2 = 128;
	N3 = 1;
	/*grid parameters*/
	dx[1] = (Rout - Rin)/N1;
	dx[2] = (th_out - th_in)/N2;
	dx[3] =  2 * M_PI;
	startx[0] = 0.;
	startx[1] = 0.;
	startx[2] = th_in;
	startx[3] = 0.;
	stopx[0] = 1.;
	stopx[1] = startx[1] + N1 * dx[1];
	stopx[2] = startx[2] + N2 * dx[2];
	stopx[3] = startx[3] + N3 * dx[3];
	R0 = 0;
	hslope = 0;

    #if(EXP_COORDS)
    startx[1] = log(Rin);
    dx[1] = (log(Rout) - log(Rin))/N1;
    stopx[1] = startx[1] + N1 * dx[1];
    #endif

	fprintf(stderr, "Resolution: %d, %d, %d\n", N1, N2, N3);
	fprintf(stderr, "startX (%le, %le), stopX(%le, %le)\n", startx[1], startx[2], stopx[1], stopx[2]);
    fprintf(stderr, "dx (%le, %le)\n", dx[1], dx[2]);

	/* Allocate storage for all model size dependent variables */
	init_storage();

	two_temp_gam =
	    0.5 * ((1. + 2. / 3. * (params.tp_over_te + 1.) / (params.tp_over_te + 2.)) +
		   gam);
	Thetae_unit = (two_temp_gam - 1.) * (MP / ME) / (1. + params.tp_over_te);
	printf("Thetae_unit = %le\n", Thetae_unit);

	dMact = 0.;
	Ladv = 0.;

    // Here we don't set up the plasma variables pointer. The values are set up directly in the functions
    // get_fluid_zone and get_fluid_params.
    return;
}



/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int record_criterion(double X1)
{
	#if(EXP_COORDS)
	const double r = exp(X1);
	#else
	const double r = X1;
	#endif
	/* this is coordinate and simulation
	   specific: stop at large distance */
	if (r > R_RECORD/d_L_unit){
		return (1);
    }else{
        return (0);
    }



}
/*Stop the tracking of the photon if it falls in the bh or is far enough to not be affected.*/
__device__ int stop_criterion(double X1, double * w, curandState * localState)
{
	#if(EXP_COORDS)
	const double r = exp(X1);
	#else
	const double r = X1;
	#endif				   
    if(*w < WEIGHT_MIN){
        if (curand_uniform_double(localState)<= 1. / ROULETTE) {
            *w = *w *  ROULETTE;
        } else{
            *w = 0.;
            return 1;
        }
    }
	if (r < RMIN/d_L_unit || r > R_RECORD/d_L_unit){
		return 1;
    }


	return (0);
}

/*Given internal coordinates, X[1], X[2], X[3], we can figure out cell indexes: (i, j, k)*/
__device__ void Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM])
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
__host__ __device__ void coord(int i, int j, int k, double *X)
{
	/*Sphere test is 2D*/
    k = 0;
	#ifdef __CUDA_ARCH__
		/* returns zone-centered values for coordinates */
		X[0] = d_startx[0];
		X[1] = d_startx[1] + (i + 0.5) * d_dx[1];
		X[2] = d_startx[2] + (j + 0.5) * d_dx[2];
		X[3] = d_startx[3] + (k + 0.5) * d_dx[3];
	#else
		/* returns zone-centered values for coordinates */
		X[0] = startx[0];
		X[1] = startx[1] + (i + 0.5) * dx[1];
		X[2] = startx[2] + (j + 0.5) * dx[2];
		X[3] = startx[3] + (k + 0.5) * dx[3];
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

    get_fluid_params(X, gcov, Ne, Thetae, B, Ucon, Ucov, Bcon, Bcov);
}


__host__ __device__ void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
                                          double *Thetae, double *B, double Ucon[NDIM],
                                          double Ucov[NDIM], double Bcon[NDIM],
                                          double Bcov[NDIM]) {
    
    double local_dx[4] = {0.};

    #ifdef __CUDA_ARCH__
        // Device Globals
        local_dx[1] = d_dx[1];
        local_dx[2] = d_dx[2];
        local_dx[3] = d_dx[3];

        if (X[1] < d_startx[1] || X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {
            *Ne = 0; *B = 0; *Thetae = 0; return;
        }
        double local_L_unit = d_L_unit;
        double local_B_unit = d_B_unit;
    #else
        // Host Globals
        local_dx[1] = dx[1];
        local_dx[2] = dx[2];
        local_dx[3] = dx[3];

        if (X[1] < startx[1] || X[1] > stopx[1] || X[2] < startx[2] || X[2] > stopx[2]) {
            *Ne = 0; *B = 0; *Thetae = 0; return;
        }
        double local_L_unit = L_unit;
        double local_B_unit = B_unit;
    #endif

    // Sub-sampling for Anti-Aliasing
    int n_sub = 5; 
    int inside_count = 0;
    double X_sub[4] = {0.};
    double r_sub, th_sub;
    
    // Calculate target radius once
    double target_radius = SPHERE_RADIUS / local_L_unit;

    // Loop over sub-grid in X1 (radial) and X2 (theta)
    for (int s1 = 0; s1 < n_sub; s1++) {
        for (int s2 = 0; s2 < n_sub; s2++) {
            
            // Calculate sub-point coordinate using local_dx
            // Centered logic: (Center - Half Cell) + (Step Index + 0.5) * Step Size
            X_sub[1] = (X[1] - 0.5 * local_dx[1]) + (s1 + 0.5) * (local_dx[1] / n_sub);
            X_sub[2] = (X[2] - 0.5 * local_dx[2]) + (s2 + 0.5) * (local_dx[2] / n_sub);
            X_sub[3] = X[3]; 

            // Convert to physical radius
            bl_coord(X_sub, &r_sub, &th_sub);

            if (r_sub <= target_radius) {
                inside_count++;
            }
        }
    }

    double vol_frac = (double)inside_count / (double)(n_sub * n_sub);

    // 3. If completely outside, return 0 (Early Exit)
    if (vol_frac <= 0.0) {
        *Ne = 0.;
        *Thetae = 0.;
        *B = 0.;
        return;
    }

    // 4. Calculate Physics at Cell Center
    double r, th;
    bl_coord(X, &r, &th);

    // Apply Volume Fraction Scaling
    *Ne = NE_VALUE * vol_frac;       
    *B  = B_VALUE * vol_frac;        
    *Thetae = THETAE_VALUE * vol_frac;          

    Ucon[0] = 1.;
    Ucon[1] = 0.;
    Ucon[2] = 0.;
    Ucon[3] = 0.;

    Bcon[0] = 0.;
    Bcon[1] = (*B) * cos(th) / local_B_unit; 
    Bcon[2] = -(*B) * sin(th) / (r + 1.e-8) / local_B_unit;
    Bcon[3] = 0.;

    #if(EXP_COORDS)
    Bcon[1] /= r;
    #endif

    lower(Ucon, gcov, Ucov);
    lower(Bcon, gcov, Bcov);
}

__device__ double bias_func(double Te, double w, int round_scatt)
{
    double bias;
    #if(0)
		double max;
        max = 0.5 * w / WEIGHT_MIN;
        //bias = Te * Te /(5. *d_max_tau_scatt);
        bias = fmax(1., d_bias_norm * Te * Te/d_max_tau_scatt);

        if (bias > max){
        bias = max;
        }

        return bias;
    #elif (1)
        double model_tau_0 = NE_VALUE * SIGMA_THOMSON * 1 * d_L_unit;
        bias = (model_tau_0 > 1.0) ? (model_tau_0) : 1.0;
        return bias;
    #else
		double avg_num_scatt, max;
        //return 1;
        max = 0.5 * w / WEIGHT_MIN;

        avg_num_scatt = d_N_scatt / (1. * d_N_superph_recorded + 1.);
        bias =
        100. * Te * Te / (d_bias_norm * d_max_tau_scatt *
                (avg_num_scatt + 2));


        if (bias < d_tp_over_te)
        bias = d_tp_over_te;
        if (bias > max)
        bias = max;
        return bias / d_tp_over_te;
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



__device__ void record_super_photon(struct of_photonSOA ph, struct of_spectrum* d_spect, unsigned long long photon_index) {
    double lE, dx2;
    int iE, ix2, index;

    if (isnan(ph.w[photon_index]) || isnan(ph.E[photon_index])) {
        printf("record isnan: %g %g\n", ph.w[photon_index], ph.E[photon_index]);
        return;
    }

    d_max_tau_scatt = atomicMaxdouble(&d_max_tau_scatt, ph.tau_scatt[photon_index]);

    double r, th;
    double XArray[NDIM] = {ph.X0[photon_index], ph.X1[photon_index], ph.X2[photon_index], ph.X3[photon_index]};
    bl_coord(XArray, &r, &th);
    dx2 = M_PI/2./N_THBINS;
    if (th > M_PI/2.) {
      ix2 = (int)( (M_PI - th) / dx2 );
    } else {
      ix2 = (int)( th / dx2 );
    }
    if (ix2 < 0 || ix2 >= N_THBINS) {
        printf("record ix2 out of bounds: %d %g\n", ix2, th);
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