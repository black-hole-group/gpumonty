#include "../decs.h"
#include "model.h"
#include "../utils.h"
#include "../metrics.h"
#include "../hdf5_utils.h"
#include "../h5io.h"


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
	double Rin = RMIN;
	double Rout = RMAX;

	double th_in = 0.;
	double th_out = M_PI;
	double two_temp_gam;

	double gam = 13./9.;

	/*Setting the resolution*/
    // R resolution is high here to properly deal with the sphere edge. The higher it is, the higher the anti-aliasing quality.
    N1 = 8192; 
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
	if (r > R_RECORD){
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
    
	if (r < RMIN || r > R_RECORD){
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

__host__ __device__ void gcov_func_row0(const double *X, double gcov[NDIM])
{
	gcov[0] = -1;
	gcov[1] = 0;
    gcov[2] = 0.;
	gcov[3] = 0;
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

    get_fluid_params(X, Ne, Thetae, B, Ucon, Ucov, Bcon, Bcov);
}


__host__ __device__ void get_fluid_params(double X[NDIM], double *Ne,
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

        double tp_over_te = d_tp_over_te;
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
        double tp_over_te = params.tp_over_te;
    #endif

    // Sub-sampling for Anti-Aliasing
    int n_sub = 5; 
    int inside_count = 0;
    double X_sub[4] = {0.};
    double r_sub, th_sub;
    
    // Calculate target radius once
    double target_radius = SPHERE_RADIUS;


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
  

    double gam = 13./9;
    double custom_thetae_unit = MP/ME * (gam - 1.)/(1. + tp_over_te);
    *Ne = MODEL_TAU0/SIGMA_THOMSON/SPHERE_RADIUS/local_L_unit * vol_frac;
    *Thetae = THETAE_VALUE * vol_frac;  
    *B = CL* sqrt(8 * M_PI * (gam - 1.) * (MP + ME)/ BETA0) * sqrt(*Ne * *Thetae)/sqrt(custom_thetae_unit) * vol_frac;

    Ucon[0] = 1.;
    Ucon[1] = 0.;
    Ucon[2] = 0.;
    Ucon[3] = 0.;

    Bcon[0] = 0.;
    Bcon[1] = (*B) * cos(th)/local_B_unit; 
    Bcon[2] = -(*B) * sin(th) / (r + 1.e-8)/local_B_unit;
    Bcon[3] = 0.;

    #if(EXP_COORDS)
    Bcon[1] /= r;
    #endif

    double gcov[NDIM][NDIM];
    gcov_func(X, gcov);
    lower(Ucon, gcov, Ucov);
    lower(Bcon, gcov, Bcov);
}



__device__ double bias_func(double Te, double w, int round_scatt)
{
    double bias;
    #if(0)
		double max;
        max = 0.5 * w / WEIGHT_MIN;
        bias = Te * Te /(5. *d_max_tau_scatt);
        //bias = fmax(1., d_bias_norm * Te * Te/d_max_tau_scatt);

        if (bias > max){
        bias = max;
        }

        return bias;
    #elif (1)
        bias = (MODEL_TAU0 > 1.0) ? (MODEL_TAU0) : 1.0;
        return bias * d_bias_guess[round_scatt];
    #else
		double avg_num_scatt, max;
        //return 1;
        max = 0.5 * w / WEIGHT_MIN;

        //avg_num_scatt = d_N_scatt / (1. * d_N_superph_recorded + 1.);
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



__host__ void report_spectrum_h5(unsigned long long N_superph_made, struct of_spectrum ***spect, const char * filename)
{
  hid_t fid = -1;

  if (params.loaded && strlen(params.spectrum) > 0) {
    char path[512];
    snprintf(path, sizeof(path), "./output/%s", params.spectrum);
    fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    fid = H5Fcreate("./output/spectrum.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }

  if (fid < 0) {
    fprintf(stderr, "! unable to open/create spectrum hdf5 file.\n");
    exit(-3);
  }

  h5io_add_attribute_str(fid, "/", "githash", xstr(VERSION));

  h5io_add_group(fid, "/params");

  // h5io_add_data_dbl(fid, "/params/NUCUT", NUCUT);
  h5io_add_data_dbl(fid, "/params/GAMMACUT", MAXGAMMA);
  h5io_add_data_dbl(fid, "/params/NUMAX", NUMAX);
  h5io_add_data_dbl(fid, "/params/NUMIN", NUMIN);
  h5io_add_data_dbl(fid, "/params/LNUMAX", log10(NUMAX));
  h5io_add_data_dbl(fid, "/params/LNUMIN", log10(NUMIN));
  h5io_add_data_dbl(fid, "/params/DLNU", DLNU);
  h5io_add_data_dbl(fid, "/params/THETAE_MIN", THETAE_MIN);
  h5io_add_data_dbl(fid, "/params/THETAE_MAX", THETAE_MAX);
  h5io_add_data_dbl(fid, "/params/TP_OVER_TE", params.tp_over_te);
  h5io_add_data_dbl(fid, "/params/WEIGHT_MIN", WEIGHT_MIN);
  // h5io_add_data_dbl(fid, "/params/KAPPA", model_kappa);
  h5io_add_data_dbl(fid, "/params/L_unit", L_unit);
  h5io_add_data_dbl(fid, "/params/T_unit", T_unit);
  h5io_add_data_dbl(fid, "/params/M_unit", M_unit);
  h5io_add_data_dbl(fid, "/params/B_unit", B_unit);
  h5io_add_data_dbl(fid, "/params/Ne_unit", Ne_unit);
  h5io_add_data_dbl(fid, "/params/RHO_unit", Rho_unit);
  h5io_add_data_dbl(fid, "/params/U_unit", U_unit);
  h5io_add_data_dbl(fid, "/params/Thetae_unit", Thetae_unit);
  h5io_add_data_dbl(fid, "/params/MBH", params.MBH_par);
  h5io_add_data_dbl(fid, "/params/Rin", Rin);
  h5io_add_data_dbl(fid, "/params/Rout", Rout);

  h5io_add_data_int(fid, "/params/THERMAL_SYNCHROTRON", params.thermal_synch);
  h5io_add_data_int(fid, "/params/POWERLAW_SYNCHROTRON", params.powerlaw_synch);
  h5io_add_data_int(fid, "/params/KAPPA_SYNCHROTRON", params.kappa_synch);
 h5io_add_data_int(fid, "/params/BREMSSTRAHLUNG", params.bremsstrahlung);
  h5io_add_data_int(fid, "/params/COMPTON", params.scattering);
  // h5io_add_data_int(fid, "/params/DIST_KAPPA", MODEL_EDF==EDF_KAPPA_FIXED?1:0);
  h5io_add_data_int(fid, "/params/N_ESAMP", N_ESAMP);
  h5io_add_data_int(fid, "/params/N_EBINS", N_EBINS);
  h5io_add_data_int(fid, "/params/N_THBINS", N_THBINS);
  h5io_add_data_int(fid, "/params/N1", N1);
  h5io_add_data_int(fid, "/params/N2", N2);
  h5io_add_data_int(fid, "/params/N3", N3);
  h5io_add_data_int(fid, "/params/Ns", params.Ns);

  h5io_add_data_str(fid, "/params/dump", params.dump);
  h5io_add_data_str(fid, "/params/model", xstr(MODEL));

  // temporary data buffers
  double lnu_buf[N_EBINS];
  double dOmega_buf[N_THBINS];
  double nuLnu_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double tau_abs_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double tau_scatt_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double x1av_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double x2av_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double x3av_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double nscatt_buf[N_TYPEBINS][N_EBINS][N_THBINS];
  double Lcomponent_buf[N_TYPEBINS];

  // normal output routine
  double dOmega, nuLnu, tau_scatt, L, Lcomponent, dL;

  max_tau_scatt = 0.;
  L = 0.;
  dL = 0.;

  double dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);
  for (int j=0; j<N_THBINS; ++j) {
    dOmega_buf[j] = 2. * dOmega_func(j * dx2, (j + 1) * dx2);
  }

  for (int k=0; k<N_TYPEBINS; ++k) {
    Lcomponent = 0.;
    for (int i=0; i<N_EBINS; ++i) {
      lnu_buf[i] = (i * dlE + lE0) / M_LN10;
      for (int j=0; j<N_THBINS; ++j) {

        dOmega = dOmega_buf[j];

        nuLnu = (ME * CL * CL) * (4. * M_PI / dOmega) * (1. / dlE);
        nuLnu *= spect[k][j][i].dEdlE/LSUN;

        tau_scatt = spect[k][j][i].tau_scatt/(spect[k][j][i].dNdlE + SMALL);

        nuLnu_buf[k][i][j] = nuLnu;
        tau_abs_buf[k][i][j] = spect[k][j][i].tau_abs/(spect[k][j][i].dNdlE + SMALL);
        tau_scatt_buf[k][i][j] = tau_scatt;
        x1av_buf[k][i][j] = spect[k][j][i].X1iav/(spect[k][j][i].dNdlE + SMALL);
        x2av_buf[k][i][j] = sqrt(fabs(spect[k][j][i].X2isq/(spect[k][j][i].dNdlE + SMALL)));
        x3av_buf[k][i][j] = sqrt(fabs(spect[k][j][i].X3fsq/(spect[k][j][i].dNdlE + SMALL)));
        nscatt_buf[k][i][j] = spect[k][j][i].nscatt / (spect[k][j][i].dNdlE + SMALL);

        if (tau_scatt > max_tau_scatt) max_tau_scatt = tau_scatt;

        dL += ME * CL * CL * spect[k][j][i].dEdlE;
        L += nuLnu * dOmega * dlE / (4. * M_PI);
        Lcomponent += nuLnu * dOmega * dlE / (4. * M_PI);
      }
    }
    Lcomponent_buf[k] = Lcomponent;
  }

  h5io_add_group(fid, "/output");

  h5io_add_data_dbl_1d(fid, "/output/lnu", N_EBINS, &lnu_buf[0]);
  h5io_add_data_dbl_1d(fid, "/output/dOmega", N_THBINS, &dOmega_buf[0]);
  h5io_add_data_dbl_3d(fid, "/output/nuLnu", N_TYPEBINS, N_EBINS, N_THBINS, &nuLnu_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/tau_abs", N_TYPEBINS, N_EBINS, N_THBINS, &tau_abs_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/tau_scatt", N_TYPEBINS, N_EBINS, N_THBINS, &tau_scatt_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/x1av", N_TYPEBINS, N_EBINS, N_THBINS, &x1av_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/x2av", N_TYPEBINS, N_EBINS, N_THBINS, &x2av_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/x3av", N_TYPEBINS, N_EBINS, N_THBINS, &x3av_buf[0][0][0]);
  h5io_add_data_dbl_3d(fid, "/output/nscatt", N_TYPEBINS, N_EBINS, N_THBINS, &nscatt_buf[0][0][0]);
  h5io_add_data_dbl_1d(fid, "/output/Lcomponent", N_TYPEBINS, &Lcomponent_buf[0]);

  h5io_add_data_int(fid, "/output/Nrecorded", N_superph_recorded);
  h5io_add_data_int(fid, "/output/Nmade", N_superph_made);
  h5io_add_data_int(fid, "/output/Nscattered", N_scatt);

  double LEdd = 4. * M_PI * GNEWT * params.MBH_par * MSUN * MP * CL / SIGMA_THOMSON;
  double MdotEdd = 4. * M_PI * GNEWT * params.MBH_par * MSUN * MP / ( SIGMA_THOMSON * CL * 0.1 );
  double Lum = L * LSUN;
  double lum = Lum / LEdd;
  double Mdot = dMact * M_unit / T_unit;
  double mdot = Mdot / MdotEdd;

  h5io_add_data_dbl(fid, "/output/L", Lum);
  h5io_add_data_dbl(fid, "/output/Mdot", Mdot);
  h5io_add_data_dbl(fid, "/output/LEdd", LEdd);
  h5io_add_data_dbl(fid, "/output/MdotEdd", MdotEdd);
  h5io_add_data_dbl(fid, "/output/efficiency", L * LSUN / (dMact * M_unit * CL * CL / T_unit));

  h5io_add_attribute_str(fid, "/output/L", "units", "erg/s");
  h5io_add_attribute_str(fid, "/output/LEdd", "units", "erg/s");
  h5io_add_attribute_str(fid, "/output/Mdot", "units", "g/s");
  h5io_add_attribute_str(fid, "/output/MdotEdd", "units", "g/s");

  // Standard logging to stderr
  printf("\n\033[1m==================== OUTPUT ====================\033[0m\n");
  fprintf(stderr, "MBH = %g Msun\n", params.MBH_par);
//   fprintf(stderr, "a = %g\n", bhspin);
  fprintf(stderr,
    "luminosity %g erg/s\ndL %g\ndMact %g\nefficiency %g\nL/Ladv %g\nmax_tau_scatt %g\n",
      L * LSUN, dL, dMact * M_unit / T_unit / (MSUN / YEAR),
      L * LSUN / (dMact * M_unit * CL * CL / T_unit),
      L * LSUN / (Ladv * M_unit * CL * CL / T_unit),
      max_tau_scatt);
  fprintf(stderr, "Mdot = %g g/s, MdotEdd = %g g/s, mdot = %g MdotEdd\n", Mdot, MdotEdd, mdot);

  printf("\n");
  fprintf(stderr, "Number of superphotons made: %llu\n", N_superph_made);
  fprintf(stderr, "Number of superphotons scattered: %llu\n", N_scatt);
  fprintf(stderr, "Number of superphotons recorded: %llu\n", N_superph_recorded);
  fprintf(stderr, "Data saved in %s\n", filename);
	printf("\n\033[1m================================================\033[0m\n");
  H5Fclose(fid);
}
