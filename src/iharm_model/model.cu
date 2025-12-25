#include "../decs.h"
#include "model.h"
#include "../utils.h"
#include "../metrics.h"
#include "../hdf5_utils.h"

static int with_electrons;

// static hdf5_blob fluid_header = { 0 };

static int with_radiation;

__host__ void init_storage(void)
{

    p = (double *) malloc(NPRIM * N1 * N2 * N3 * sizeof(double *));
    geom = (struct of_geom *) malloc(N1* N2* sizeof(struct of_geom));
    return;
}


#include <hdf5.h>
#include <hdf5_hl.h>


void init_data()
{
  double dV, V;
  unsigned long long nprims = 0;
  // 1. Checking if file exists

  if ( hdf5_open((char *)params.dump) < 0 ) {
    exit(-1);
  }

  // get dump info to copy to grmonty output
  // fluid_header = hdf5_get_blob("/header");

  // read header
  hdf5_set_directory("/header/");
  // flag reads
  with_electrons = 0;
  with_radiation = 0;
  if ( hdf5_exists("has_electrons") )
    hdf5_read_single_val(&with_electrons, "has_electrons", H5T_STD_I32LE);
  if ( hdf5_exists("has_radiation") )
    hdf5_read_single_val(&with_radiation, "has_radiation", H5T_STD_I32LE);

  // read geometry
  //int N1_local, N2_local, N3_local;
  hdf5_read_single_val(&nprims, "n_prim", H5T_STD_I32LE);
  hdf5_read_single_val(&N1, "n1", H5T_STD_I32LE);
  hdf5_read_single_val(&N2, "n2", H5T_STD_I32LE);
  hdf5_read_single_val(&N3, "n3", H5T_STD_I32LE);
  hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);


  // conditional reads
  double game = 4./3;
  double gamp = 5./3;
  if (with_electrons) {
    fprintf(stderr, "custom electron model loaded...\n");
    hdf5_read_single_val(&game, "gam_e", H5T_IEEE_F64LE);
    hdf5_read_single_val(&gamp, "gam_p", H5T_IEEE_F64LE);
  }else{
    fprintf(stderr, "no electron model loaded, assuming single fluid...\n");
  }

  if(game != GAME || gamp != GAMP){
    fprintf(stderr, "Code electron/proton gamma does not match the simulation data gammas\n");
    fprintf(stderr, "Code gam_e, gam_p: %lf, %lf\n", GAME, GAMP);
    fprintf(stderr, "Simulation gam_e, gam_p: %lf, %lf\n", game, gamp);
    fprintf(stderr, "Change macros in model.h file");
    exit(-3);
  }

  if (!USE_FIXED_TPTE && !USE_MIXED_TPTE) {
    if (with_electrons != 1) {
      fprintf(stderr, "! no electron temperature model specified in iharm_model\n");
      exit(-3);
    }
    with_electrons = 1;
    Thetae_unit = MP/ME;
  } else if (USE_FIXED_TPTE && !USE_MIXED_TPTE) {
    with_electrons = 0; // force TP_OVER_TE to overwrite electron temperatures
    fprintf(stderr, "using fixed tp_over_te ratio = %g\n", params.tp_over_te);
    Thetae_unit = MP/ME * (gam-1.) / (1. + params.tp_over_te);
    Thetae_unit = 2./3. * MP/ME / (2. + params.tp_over_te);
  } else if (USE_MIXED_TPTE && !USE_FIXED_TPTE) {
    Thetae_unit = 2./3. * MP/ME / 5.;
    with_electrons = 2;
    fprintf(stderr, "using mixed tp_over_te with trat_small = %g and trat_large = %g\n", params.trat_small, params.trat_large);
  } else {
    fprintf(stderr, "! please change electron model in model/iharm.c\n");
    exit(-3);
  }

//   if (with_radiation) {
//     fprintf(stderr, "custom radiation field tracking information loaded...\n");
//     hdf5_set_directory("/header/units/");
//     hdf5_read_single_val(&M_unit, "M_unit", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&T_unit, "T_unit", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&L_unit, "L_unit", H5T_IEEE_F64LE);
//     if (!USE_FIXED_TPTE && !USE_MIXED_TPTE) {
//       hdf5_read_single_val(&Thetae_unit, "Thetae_unit", H5T_IEEE_F64LE);
//     }
//     hdf5_read_single_val(&MBH, "Mbh", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&TP_OVER_TE, "tp_over_te", H5T_IEEE_F64LE);
//   } else {
//     if (! params->loaded) {
//       report_bad_input(argc);
//       sscanf(argv[3], "%lf", &M_unit);
//       sscanf(argv[4], "%lf", &MBH);
//       sscanf(argv[5], "%lf", &TP_OVER_TE);
//     } else {
//       M_unit = params->M_unit;
//       MBH = params->MBH;
//       TP_OVER_TE = params->TP_OVER_TE;
//     }
//     MBH *= MSUN;
//     L_unit = GNEWT*MBH/(CL*CL);
//     T_unit = L_unit/CL;
//   }

  hdf5_set_directory("/header/geom/");
  hdf5_read_single_val(&startx[1], "startx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[2], "startx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[3], "startx3", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[1], "dx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[2], "dx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[3], "dx3", H5T_IEEE_F64LE);

  hdf5_set_directory("/header/geom/mks/");
  double a;
  hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
  hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);

  if (a != BHSPIN){
      fprintf(stderr, "BH spin does not match the simulation data BH spin\n");
      fprintf(stderr, "Code BH spin: %lf, Simulation BH spin: %lf\n", BHSPIN, a);
      fprintf(stderr, "Change macros in model.h file");
      exit(-2);
  }
  hdf5_read_single_val(&Rin, "r_in", H5T_IEEE_F64LE);
  hdf5_read_single_val(&Rout, "r_out", H5T_IEEE_F64LE);
//   if ( METRIC_MKS3 ) {
//     hdf5_set_directory("/header/geom/mks3/");
//     hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&mks3R0, "R0", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&mks3H0, "H0", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&mks3MY1, "MY1", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&mks3MY2, "MY2", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&mks3MP0, "MP0", H5T_IEEE_F64LE);
//     Rout = 100.;
//   } else {
//     hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
//     hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
//     if (hdf5_exists("Rin")) {
//       hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
//       hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
//     } else {
//       hdf5_read_single_val(&Rin, "r_in", H5T_IEEE_F64LE);
//       hdf5_read_single_val(&Rout, "r_out", H5T_IEEE_F64LE);
//     }
//     if (with_derefine_poles) {
//       fprintf(stderr, "custom refinement at poles loaded...\n");
//       hdf5_read_single_val(&poly_xt, "poly_xt", H5T_IEEE_F64LE);
//       hdf5_read_single_val(&poly_alpha, "poly_alpha", H5T_IEEE_F64LE);
//       hdf5_read_single_val(&mks_smooth, "mks_smooth", H5T_IEEE_F64LE);
//       poly_norm = 0.5*M_PI*1./(1. + 1./(poly_alpha + 1.)*1./pow(poly_xt, poly_alpha));
//     }
//   }

  // Set other geometry
  stopx[0] = 1.;
  stopx[1] = startx[1]+N1*dx[1];
  stopx[2] = startx[2]+N2*dx[2];
  stopx[3] = startx[3]+N3*dx[3];

  // Set remaining units and constants
  max_tau_scatt = (6.*L_UNIT)*RHO_UNIT*0.4; // this doesn't make sense ...
  max_tau_scatt = 0.0001; // TODO look at this in the future and figure out a smarter general value

  // Horizon and "max R for geodesic tracking" in KS coordinates
  Rh = 1. + sqrt(1. - a * a);

  fprintf(stderr, "L_unit, T_unit, M_unit = %g %g %g\n", L_UNIT, T_UNIT, M_UNIT);
  fprintf(stderr, "B_unit, Ne_unit, RHO_unit = %g %g %g\n", B_UNIT, NE_UNIT, RHO_UNIT);
  fprintf(stderr, "Thetae_unit = %g\n", Thetae_unit);

  // Allocate storage and set geometry
  init_storage();
  init_geometry();
  // Read prims.
  // Assume standard ordering in iharm dump file, especially for
  // electron variables...
  hdf5_set_directory("/");
 hsize_t fdims[] = { (hsize_t)N1, (hsize_t)N2, (hsize_t)N3, (hsize_t)nprims};
  hsize_t fstart[] = { 0, 0, 0, 0 };
  hsize_t fcount[] = { (hsize_t)N1, (hsize_t)N2, (hsize_t)N3, 1 };
  hsize_t mstart[] = { 0, 0, 0, 0 };

for (int var = 0; var < nprims; var++) {
    fstart[3] = var;
    double *p_var = &p[var * N1 * N2 * N3];
    hdf5_read_array(p_var, "prims", 4, fdims, fstart, fcount, fcount, mstart, H5T_IEEE_F64LE);
}



if (with_electrons == 1) {
    // KEL is primitive index 8
    fstart[3] = 8;
    double *p_kel = &p[KEL * N1 * N2 * N3];
    hdf5_read_array(p_kel, "prims", 4, fdims, fstart, fcount, fcount, mstart, H5T_IEEE_F64LE);

    // KTOT is primitive index 9
    fstart[3] = 9;
    double *p_ktot = &p[KTOT * N1 * N2 * N3];
    hdf5_read_array(p_ktot, "prims", 4, fdims, fstart, fcount, fcount, mstart, H5T_IEEE_F64LE);
}

  hdf5_close();

  V = dMact = Ladv = 0.;
  dV = dx[1]*dx[2]*dx[3];
  ZLOOP {

    V += dV*geom[SPATIAL_INDEX2D(i,j)].g;

    double Ne, Thetae, Bmag, Ucon[NDIM], Ucov[NDIM], Bcon[NDIM];
    get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, geom, p);
    bias_norm += dV*geom[SPATIAL_INDEX2D(i,j)].g * Thetae*Thetae;
    if (10 <= i && i <= 20) {
      lower(Ucon, geom[SPATIAL_INDEX2D(i,j)].gcov, Ucov);
      dMact += geom[SPATIAL_INDEX2D(i,j)].g*dx[2]*dx[3]*p[NPRIM_INDEX3D(KRHO,i,j,k)]*Ucon[1];
      Ladv += geom[SPATIAL_INDEX2D(i,j)].g*dx[2]*dx[3]*p[NPRIM_INDEX3D(UU,i,j,k)]*Ucon[1]*Ucov[0];
    }

  }

  dMact /= 11.;
  Ladv /= 1.;
  bias_norm /= V;
  fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);

  //init_tetrads();
}


/*Criterion whether or not to record the photon once it has left the zone of interest (reached stop_criterion)*/
__device__ int GPU_record_criterion(double X1)
{
	const double X1max = log(1.1 * RMAX);
	/* this is coordinate and simulation
	   specific: stop at large distance */
	//printf("X[1] coord = %le, X1max = %le\n", ph->X[1], X1max);
	if (X1 > X1max)
		return (1);

	else
		return (0);

}


#define MIN(A,B) (A<B?A:B)
__device__ double GPU_stepsize(const double X[NDIM], const double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;

  dlx2 = EPS * MIN(X[2], 1. - X[2]) / (fabs(K[2]) + SMALL);
	dlx1 = EPS / (fabs(K[1]) + SMALL);
	dlx3 = EPS / (fabs(K[3]) + SMALL);

	idlx1 = 1. / (fabs(dlx1) + SMALL);
	idlx2 = 1. / (fabs(dlx2) + SMALL);
	idlx3 = 1. / (fabs(dlx3) + SMALL);

	dl = 1. / (idlx1 + idlx2 + idlx3);

	return (dl);
}
#undef MIN



/*Stop the tracking of the photon if it falls in the bh or is far enough to not be affected.*/
__device__ int GPU_stop_criterion(double X1, double * w, curandState * localState)
{
	double wmin, X1min, X1max;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */
	
    
	if (*w < wmin) {
		if (curand_uniform_double(localState) <= 1. / ROULETTE) {
			*w = *w * ROULETTE;
		} else {
			*w = 0.;
			return 1;
		}
	}
    
  X1min = log(d_Rh * 1.05);	/* this is coordinate-specific; stop
				   at event horizon */
	X1max = log(RMAX * 1.1);	/* this is coordinate and simulation
				   specific: stop at large distance */

  
	if (X1 < X1min || X1 > X1max) {
		return 1;
	}

	return (0);
}

/*Given internal coordinates, X[1], X[2], X[3], we can figure out cell indexes: (i, j, k)*/
__device__ void GPU_Xtoijk(const double X[NDIM], int *i, int *j, int *k, double del[NDIM])
{
  double phi;
  double XG[NDIM];
  for (int mu = 0; mu < NDIM; mu++) XG[mu] = X[mu];
  phi = fmod(X[3], d_stopx[3]);
  if (phi < 0.0) phi = d_stopx[3]+phi;

	*i = (int) ((X[1] - d_startx[1]) / d_dx[1] - 0.5 + 1000.) - 1000;
	*j = (int) ((X[2] - d_startx[2]) / d_dx[2] - 0.5 + 1000.) - 1000;
  *k = (int) ((phi  - d_startx[3]) / d_dx[3] - 0.5 + 1000) - 1000;  
 // don't allow "center zone" to be outside of [0,N*-1]. this will often fire
  // for exotic corodinate systems and occasionally for normal ones. wrap x3.
  if (*i < 0) *i = 0;
  if (*j < 0) *j = 0;
  if (*k < 0) *k = 0;
  if (*i > d_N1-2) *i = d_N1-2; 
  if (*j > d_N2-2) *j = d_N2-2; 
  if (*k > d_N3-1) *k = d_N3-1; 

  // now construct del
  del[1] = (XG[1] - ((*i + 0.5) * d_dx[1] + d_startx[1])) / d_dx[1];
  del[2] = (XG[2] - ((*j + 0.5) * d_dx[2] + d_startx[2])) / d_dx[2];
  del[3] = (phi - ((*k + 0.5) * d_dx[3] + d_startx[3])) / d_dx[3];

  // finally enforce limits on del
  if (del[1] > 1.) del[1] = 1.;
  if (del[1] < 0.) del[1] = 0.;
  if (del[2] > 1.) del[2] = 1.;
  if (del[2] < 0.) del[2] = 0.;
  if (del[3] > 1.) del[3] = 1.;
  if (del[3] < 0.) {
    int oldk = *k;
    *k = d_N3-1;
    del[3] += 1.;
    if (del[3] < 0) {
      printf(" ! unable to resolve X[3] coordinate to zone %d %d %g %g\n", oldk, *k, del[3], XG[3]);
    }
  }
	return;
}



/*Given cell indexes i and j, we can figure out internal coordinates X[1], X[2], X[3]*/
__host__ __device__ void coord(const int i, const int j, const int k, double *X)
{
	#ifdef __CUDA_ARCH__
		/* returns zone-centered values for coordinates */
		X[0] = d_startx[0];
    X[1] = d_startx[1] + (i+0.5)*d_dx[1];
    X[2] = d_startx[2] + (j+0.5)*d_dx[2];
    X[3] = d_startx[3] + (k+0.5)*d_dx[3];
	#else
		/* returns zone-centered values for coordinates */
		X[0] = startx[0];
		X[1] = startx[1] + (i + 0.5) * dx[1];
		X[2] = startx[2] + (j + 0.5) * dx[2];
		X[3] = startx[3] + (k + 0.5) * dx[3];
	#endif


	return;
}

__host__ __device__ void gcov_func(const double *X , double gcov[][NDIM])
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
	double radius0 = d_R0;
	double theta_slope = d_hslope;
	#else
	double radius0 = R0;
	double theta_slope = hslope;
	#endif
	//sincos(th, &sth, &cth);
	sth = sin(th);
	cth = cos(th);
	sth = fabs(sth) + SMALL;
	s2 = sth * sth;
	rho2 = r * r + BHSPIN * BHSPIN * cth * cth;

	/* transformation for Kerr-Schild -> modified Kerr-Schild */
	tfac = 1.;
	rfac = r - radius0;
	hfac = M_PI + (1. - theta_slope) * M_PI * cos(2. * M_PI * X[2]);
	pfac = 1.;

	gcov[0][0] = (-1. + 2. * r / rho2) * tfac * tfac;
	gcov[0][1] = (2. * r / rho2) * tfac * rfac;
	gcov[0][3] = (-2. * BHSPIN * r * s2 / rho2) * tfac * pfac;

	gcov[1][0] = gcov[0][1];
	gcov[1][1] = (1. + 2. * r / rho2) * rfac * rfac;
	gcov[1][3] = (-BHSPIN * s2 * (1. + 2. * r / rho2)) * rfac * pfac;

	gcov[2][2] = rho2 * hfac * hfac;

	gcov[3][0] = gcov[0][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] =
	    s2 * (rho2 + BHSPIN*BHSPIN * s2 * (1. + 2. * r / rho2)) * pfac * pfac;
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
__host__ __device__ void bl_coord(const double *X, double *r, double *th)
{
  #ifdef __CUDA_ARCH__
  const double theta_slope = d_hslope;
  #else
  const double theta_slope = hslope;
  #endif
	*r = exp(X[1]);
	*th = M_PI * X[2] + ((1. - theta_slope) / 2.) * sin(2. * M_PI * X[2]);

	return;
}


__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,
    double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *  d_geom, const double *  d_p)
{
    int l, m;
    double Ucov[NDIM], Bcov[NDIM];
    double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;

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
    *Thetae = thetae_func(d_p[NPRIM_INDEX3D(UU, i, j, k)], d_p[NPRIM_INDEX3D(KRHO, i, j, k)] , (*B)/B_UNIT, d_p[NPRIM_INDEX3D(KEL, i, j, k)]);
    *Ne = d_p[NPRIM_INDEX3D(KRHO, i, j, k)] * NE_UNIT;

    if (*Thetae > THETAE_MAX) *Thetae = THETAE_MAX;

    double sig = pow(*B/B_UNIT,2)/(*Ne/NE_UNIT);
    if(sig > 1. || i < 9) {
        *Thetae = SMALL;
    }

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
    double rho, uu, kel;
    double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
    double gcon[NDIM][NDIM], coeff[8];

    //checks if it's within the grid
    if (X[1] < d_startx[1] ||
    X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {
    *Ne = 0.;

    return;
    }

    // Finds out i and j index as well as fraction displacement del from the coordinates X[1], X[2], X[3]
    GPU_Xtoijk(X, &i, &j, &k, del);
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
    rho = GPU_interp_scalar_pointer(d_p, KRHO, i, j, k, coeff);
    uu = GPU_interp_scalar_pointer(d_p, UU, i, j, k, coeff);
    kel = GPU_interp_scalar_pointer(d_p, KEL, i,j,k, coeff);
    

    *Ne = rho * NE_UNIT;

    Bp[1] = GPU_interp_scalar_pointer(d_p, B1, i, j, k, coeff);
    Bp[2] = GPU_interp_scalar_pointer(d_p, B2, i, j, k, coeff);
    Bp[3] = GPU_interp_scalar_pointer(d_p, B3, i, j, k, coeff);

    Vcon[1] = GPU_interp_scalar_pointer(d_p, U1, i, j, k, coeff);
    Vcon[2] = GPU_interp_scalar_pointer(d_p, U2, i, j, k, coeff);
    Vcon[3] = GPU_interp_scalar_pointer(d_p, U3, i, j, k, coeff);

    gcon_func(X, gcov, gcon);

    /* Get Ucov */
    VdotV = 0.;
    for (int i = 1; i < NDIM; i++)
      for (int j = 1; j < NDIM; j++)
        VdotV += gcov[i][j] * Vcon[i] * Vcon[j];

    //printf("gcov = %.15e, %.15e, %.15e, %.15e\n%.15e, %.15e, %.15e, %.15e\n%.15e, %.15e, %.15e, %.15e\n%.15e, %.15e, %.15e, %.15e\n", gcov[0][0], gcov[0][1], gcov[0][2], gcov[0][3], gcov[1][0], gcov[1][1], gcov[1][2], gcov[1][3], gcov[2][0], gcov[2][1], gcov[2][2], gcov[2][3], gcov[3][0], gcov[3][1], gcov[3][2], gcov[3][3]);
    Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
    Ucon[0] = -Vfac * gcon[0][0];
    for (int i = 1; i < NDIM; i++){
    Ucon[i] = Vcon[i] - Vfac * gcon[0][i];
    }
    lower(Ucon, gcov, Ucov);

    /* Get B and Bcov */
    UdotBp = 0.;
    for (int i = 1; i < NDIM; i++)
    UdotBp += Ucov[i] * Bp[i];
    Bcon[0] = UdotBp;
    for (int i = 1; i < NDIM; i++)
    Bcon[i] = (Bp[i] + Ucon[i] * UdotBp) / Ucon[0];
    lower(Bcon, gcov, Bcov);

    *B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
    Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_UNIT;

    *Thetae = thetae_func(uu, rho, (*B)/B_UNIT, kel);
    if(*Thetae > THETAE_MAX) *Thetae = THETAE_MAX;

    double sig = pow(*B/B_UNIT, 2.)/(*Ne/NE_UNIT);
    if(sig > 1.) *Thetae = SMALL;
}



__device__ double GPU_bias_func(double Te, double w, int round_scatt)
{
  double bias, max;
  max = 0.5 * w / WEIGHT_MIN;

  if (Te > 1000.) Te = 1000.;
  bias = 16. * Te * Te / (5. * 1e-4);

  if (bias > max) bias = max;

  return bias * 30. * 1./2.;
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

    double r, th;
    double X_Array[NDIM] = {ph.X0[photon_index], ph.X1[photon_index], ph.X2[photon_index], ph.X3[photon_index]};
    bl_coord(X_Array, &r, &th);
    dx2 = M_PI/2./N_THBINS;
    if (th > M_PI/2.) {
        ix2 = (int)( (M_PI - th) / dx2 );
    } else {
        ix2 = (int)( th / dx2 );
    }
    if (ix2 < 0 || ix2 >= N_THBINS) return;
    // Get energy bin
    lE = log(ph.E[photon_index]);
    iE = (int)((lE - lE0) / dlE + 2.5) - 2;
    if (iE < 0 || iE >= N_EBINS) return;
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
    atomicAdd(&(d_spect[index].nscatt), ph.nscatt[photon_index]);
    atomicAdd(&(d_spect[index].nph), 1);
}

__host__ __device__ double thetae_func(double uu, double rho, double B, double kel)
{
    // assumes uu, rho, B, kel in code units
    double thetae = 0.;
    #ifdef __CUDA_ARCH__
    double theta_unit = d_thetae_unit;
    double local_trat_small = d_trat_small;
    double local_trat_large = d_trat_large;
    double local_beta_crit = d_beta_crit;
    double thetae_local_max = d_thetae_max;
    #else
    double theta_unit = Thetae_unit;
    double local_trat_small = params.trat_small;
    double local_trat_large = params.trat_large;
    double local_beta_crit = params.beta_crit;
    double thetae_local_max = params.Thetae_max;
    #endif
    // Gotta save d_Thetae_unit, game, gamp, beta, beta_crit, trat_large, trat_small to device memory

    if (WITH_ELECTRONS == 0) {
    //fixed tp/te ratio
    thetae = uu / rho * theta_unit;
    } else if (WITH_ELECTRONS == 1) {
    // howes/kawazura model from IHARM electron thermodynamics
    //thetae = kel * pow(rho, GAME-1.) * Thetae_unit;
    } else if (WITH_ELECTRONS == 2 ) {
    double beta = uu * (GAM -1.) / 0.5 / B / B;
    double b2 = beta*beta / local_beta_crit/local_beta_crit;
    double trat = local_trat_large * b2/(1.+b2) + local_trat_small /(1.+b2);
    if (B == 0) trat = local_trat_large;
    thetae = (MP/ME) * (GAME-1.) * (GAMP-1.) / ( (GAMP-1.) + (GAME-1.)*trat ) * uu / rho;
    }

    return 1./(1./thetae + 1./Thetae_MAX2);
}
