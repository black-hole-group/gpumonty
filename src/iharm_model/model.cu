/*
 * GPUmonty - /iharm_model/model.cu
 * Copyright (C) 2026 Pedro Naethe Motta
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
 */
#include "../decs.h"
#include "model.h"
#include "../utils.h"
#include "../metrics.h"
#include "../hdf5_utils.h"
#include "../h5io.h"
#include "../main.h"


/**
 * Time coordinate of the model.
 */
static double t;

__host__ void init_storage(void)
{

    // These + 2 comes from getting beta_array and sigma_array into the p structure
    p = (double *) malloc((NPRIM) * N1 * N2 * N3 * sizeof(double *));
    geom = (struct of_geom *) malloc(N1* N2* sizeof(struct of_geom));
    return;
}


#include <hdf5.h>
#include <hdf5_hl.h>


static hdf5_blob fluid_header = { 0 };
double poly_norm, poly_xt, poly_alpha, mks_smooth; // mmks
double mks3R0, mks3H0, mks3MY1, mks3MY2, mks3MP0; // mks3

int METRIC;
__device__ int d_METRIC;

//FMKS
__device__ double d_poly_norm, d_poly_xt, d_poly_alpha, d_mks_smooth; // mmks

//MKS3
__device__ double d_mks3R0, d_mks3H0, d_mks3MY1, d_mks3MY2, d_mks3MP0; // mks3

int with_electrons;
static int with_radiation;

double game, gamp, gam;
__device__ double d_game, d_gamp, d_gam;

static double MBH;

double TP_OVER_TE;

__device__ int d_with_electrons, d_with_radiation;

void init_data()
{
  double dV, V;
  unsigned long long nprims = 0;
  // 1. Checking if file exists

  if ( hdf5_open((char *)params.dump) < 0 ) {
    exit(-1);
  }

  // get dump info to copy to grmonty output
  fluid_header = hdf5_get_blob("/header");

  // read header
  hdf5_set_directory("/header/");
  // flag reads
  with_electrons = 0;
  with_radiation = 0;
  int FMKS = 0;
  if ( hdf5_exists("has_electrons") )
    hdf5_read_single_val(&with_electrons, "has_electrons", H5T_STD_I32LE);
  if ( hdf5_exists("has_radiation") )
    hdf5_read_single_val(&with_radiation, "has_radiation", H5T_STD_I32LE);

  // read geometry
  //int N1_local, N2_local, N3_local;
  char metric_name[20];
  hid_t string_type = hdf5_make_str_type(20);
  hdf5_read_single_val(&metric_name, "metric", string_type);

  int metric_eKS = 0;
  int metric_MKS3 = 0;
  if ( strncmp(metric_name, "MMKS", 19) == 0 || strncmp(metric_name, "FMKS", 19) == 0 ) {
    FMKS = 1;
  } else if ( strncmp(metric_name, "MKS3", 19) == 0 ) {
    metric_eKS = 1;
    metric_MKS3 = 1;
  }

  if(metric_eKS){
    if(metric_MKS3){
        fprintf(stderr, "using MKS3 metric...\n");
        fprintf(stderr, "MKS3 hasn't been tested against igrmonty yet, so use with caution!\n");
        METRIC= METRIC_MKS3;
      }else{
        fprintf(stderr, "using Exponential Kerr-Schild metric\n");
        METRIC = METRIC_eKS;
      }
  }else if (FMKS) {
    fprintf(stderr, "using Funky-Modified Kerr-Schild metric\n");
    METRIC = METRIC_FMKS;
  } else {
    fprintf(stderr, "using Modified Kerr-Schild metric\n");
    METRIC = METRIC_MKS;
  }
  

  hdf5_read_single_val(&nprims, "n_prim", H5T_STD_I32LE);
  hdf5_read_single_val(&N1, "n1", H5T_STD_I32LE);
  hdf5_read_single_val(&N2, "n2", H5T_STD_I32LE);
  hdf5_read_single_val(&N3, "n3", H5T_STD_I32LE);
  hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);

  printf("Resolution = %d x %d x %d, nprims = %d\n", N1, N2, N3, (int)nprims);


  // conditional reads
  game = 4./3;
  gamp = 5./3;
  if (with_electrons) {
    fprintf(stderr, "custom electron model loaded...\n");
    hdf5_read_single_val(&game, "gam_e", H5T_IEEE_F64LE);
    hdf5_read_single_val(&gamp, "gam_p", H5T_IEEE_F64LE);
  }else{
    fprintf(stderr, "no electron model loaded, assuming single fluid...\n");
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

  if (with_radiation) {
    fprintf(stderr, "custom radiation field tracking information loaded...\n");
    hdf5_set_directory("/header/units/");
    hdf5_read_single_val(&M_unit, "M_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&T_unit, "T_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&L_unit, "L_unit", H5T_IEEE_F64LE);
    if (!USE_FIXED_TPTE && !USE_MIXED_TPTE) {
      hdf5_read_single_val(&Thetae_unit, "Thetae_unit", H5T_IEEE_F64LE);
    }
    hdf5_read_single_val(&MBH, "Mbh", H5T_IEEE_F64LE);
    hdf5_read_single_val(&TP_OVER_TE, "tp_over_te", H5T_IEEE_F64LE);
  } else {
    M_unit = params.M_unit;
    MBH = params.MBH_par;
    TP_OVER_TE = params.tp_over_te;
    MBH *= MSUN;
    L_unit = GNEWT*MBH/(CL*CL);
    T_unit = L_unit/CL;
  }

  hdf5_set_directory("/header/geom/");
  hdf5_read_single_val(&startx[1], "startx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[2], "startx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[3], "startx3", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[1], "dx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[2], "dx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[3], "dx3", H5T_IEEE_F64LE);

  hdf5_set_directory("/header/geom/mks/");
  if (FMKS) hdf5_set_directory("/header/geom/mmks/");

  if ( metric_MKS3 ) {
    hdf5_set_directory("/header/geom/mks3/");
    hdf5_read_single_val(&bhspin, "a", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3R0, "R0", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3H0, "H0", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MY1, "MY1", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MY2, "MY2", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MP0, "MP0", H5T_IEEE_F64LE);
    Rout = 100.;
  } else {
    hdf5_read_single_val(&bhspin, "a", H5T_IEEE_F64LE);
    hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
    if (hdf5_exists("Rin")) {
      hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
      hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
    } else {
      hdf5_read_single_val(&Rin, "r_in", H5T_IEEE_F64LE);
      hdf5_read_single_val(&Rout, "r_out", H5T_IEEE_F64LE);
    }
    if (FMKS) {
      hdf5_read_single_val(&poly_xt, "poly_xt", H5T_IEEE_F64LE);
      hdf5_read_single_val(&poly_alpha, "poly_alpha", H5T_IEEE_F64LE);
      hdf5_read_single_val(&mks_smooth, "mks_smooth", H5T_IEEE_F64LE);
      poly_norm = 0.5*M_PI*1./(1. + 1./(poly_alpha + 1.)*1./pow(poly_xt, poly_alpha));
      printf("Using FMKS with poly_norm = %g, poly_xt = %g, poly_alpha = %g, mks_smooth = %g\n", poly_norm, poly_xt, poly_alpha, mks_smooth);
    }
  }

  // Set other geometry
  stopx[0] = 1.;
  stopx[1] = startx[1]+N1*dx[1];
  stopx[2] = startx[2]+N2*dx[2];
  stopx[3] = startx[3]+N3*dx[3];

  // Set remaining units and constants
  // I have to reset the units here in case with_radiation is on.
  Rho_unit = M_unit/pow(L_unit,3);
  U_unit = Rho_unit*CL*CL;
  B_unit = CL*sqrt(4.*M_PI*Rho_unit);
  Ne_unit = Rho_unit/(MP + ME);
  max_tau_scatt = (6.*L_unit)*Rho_unit*0.4; // this doesn't make sense ...
  max_tau_scatt = 0.0001; // TODO look at this in the future and figure out a smarter general value

  // Horizon and "max R for geodesic tracking" in KS coordinates
  Rh = 1. + sqrt(1. - bhspin * bhspin);

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

  // load the time value from the grmhd file
  hdf5_set_directory("/");
  hdf5_read_single_val(&t, "/t", H5T_IEEE_F64LE);


  hdf5_close();

  V = dMact = Ladv = 0.;
  dV = dx[1]*dx[2]*dx[3];

#pragma omp parallel for collapse(2) default(none) \
    shared(N1, N2, N3, dV, geom, p, dx, B_unit, Ne_unit, gam) \
    reduction(+:V, bias_norm, dMact, Ladv)
  for (int i = 0; i < N1; i++) {
      for (int j = 0; j < N2; j++) {
          
          int ij_idx = SPATIAL_INDEX2D(i, j);
          double g_det = geom[ij_idx].g;
          
          V += dV * g_det * N3; 

          int in_active_region = (10 <= i && i <= 20);
          double flux_base = 0.0;
          if (in_active_region) {
              flux_base = g_det * dx[2] * dx[3];
          }

          for (int k = 0; k < N3; k++) {
              double Ne, Thetae, Bmag, Ucon[NDIM], Ucov[NDIM], Bcon[NDIM];
              
              get_fluid_zone(i, j, k, &Ne, &Thetae, &Bmag, Ucon, Bcon, geom, p);
              
              bias_norm += dV * g_det * Thetae * Thetae;

              double bsq = Bmag * Bmag/(B_unit * B_unit);
              p[NPRIM_INDEX3D(SIGMA, i,j,k)] = bsq/(Ne/Ne_unit);
              p[NPRIM_INDEX3D(BETA, i,j,k)] = p[NPRIM_INDEX3D(UU, i, j, k)] * (gam - 1.) * 2./bsq;
              
              if (in_active_region) {
                  lower(Ucon, geom[ij_idx].gcov, Ucov);
                  
                  double flux_factor = flux_base * Ucon[1];
                  dMact += flux_factor * p[NPRIM_INDEX3D(KRHO, i, j, k)];
                  Ladv  += flux_factor * p[NPRIM_INDEX3D(UU, i, j, k)] * Ucov[0];
              }
          }
      }
  }

  dMact /= 11.;
  Ladv /= 1.;
  bias_norm /= V;
  fprintf(stderr, "dMact: %g, Ladv: %g\n", dMact, Ladv);
  

  //init_tetrads();
}


__device__ int record_criterion(double X1)
{
	const double X1max = log(1.1 * RMAX);
	/* this is coordinate and simulation
	   specific: stop at large distance */
	if (X1 > X1max)
		return (1);

	else
		return (0);

}


#define MIN(A,B) (A<B?A:B)
__device__ double stepsize(const double X[NDIM], const double K[NDIM])
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



__device__ int stop_criterion(double X1, double * w, curandState * localState)
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

// /*Given internal coordinates, X[1], X[2], X[3], we can figure out cell indexes: (i, j, k)*/
// __device__ void Xtoijk(const double X[NDIM], int *i, int *j, int *k, double del[NDIM])
// {
//   double phi;
//   double XG[NDIM];
//   if (d_METRIC == METRIC_eKS) {
//     // the geodesics are evolved in eKS so invert through KS -> zone coordinates
//     double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
//     for (int mu = 0; mu < NDIM; mu++) XG[mu] = Xks[mu];
//   }else if (d_METRIC == METRIC_MKS3) {
//     double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
//     double H0 = d_mks3H0, MY1 = d_mks3MY1, MY2 = d_mks3MY2, MP0 = d_mks3MP0;
//     double KSx1 = Xks[1], KSx2 = Xks[2];
//     XG[0] = Xks[0];
//     XG[1] = log(Xks[1] - d_mks3R0);
//     XG[2] = (-(H0*pow(KSx1,MP0)*M_PI) - pow(2.,1. + MP0)*H0*MY1*M_PI + 
//       2.*H0*pow(KSx1,MP0)*MY1*M_PI + pow(2.,1. + MP0)*H0*MY2*M_PI + 
//       2.*pow(KSx1,MP0)*atan(((-2.*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI))/(2.*
//       H0*(-pow(KSx1,MP0) - pow(2.,1 + MP0)*MY1 + 2.*pow(KSx1,MP0)*MY1 + 
//         pow(2.,1. + MP0)*MY2)*M_PI);
//     XG[3] = Xks[3];
//   } else {
//     for (int mu = 0; mu < NDIM; mu++) XG[mu] = X[mu];
//   }
//   phi = fmod(X[3], d_stopx[3]);
//   if (phi < 0.0) phi = d_stopx[3]+phi;

// 	*i = (int) ((X[1] - d_startx[1]) / d_dx[1] - 0.5 + 1000.) - 1000;
// 	*j = (int) ((X[2] - d_startx[2]) / d_dx[2] - 0.5 + 1000.) - 1000;
//   *k = (int) ((phi  - d_startx[3]) / d_dx[3] - 0.5 + 1000) - 1000;  
//  // don't allow "center zone" to be outside of [0,N*-1]. this will often fire
//   // for exotic corodinate systems and occasionally for normal ones. wrap x3.
//   if (*i < 0) *i = 0;
//   if (*j < 0) *j = 0;
//   if (*k < 0) *k = 0;
//   if (*i > d_N1-2) *i = d_N1-2; 
//   if (*j > d_N2-2) *j = d_N2-2; 
//   if (*k > d_N3-1) *k = d_N3-1; 

//   // now construct del
//   del[1] = (XG[1] - ((*i + 0.5) * d_dx[1] + d_startx[1])) / d_dx[1];
//   del[2] = (XG[2] - ((*j + 0.5) * d_dx[2] + d_startx[2])) / d_dx[2];
//   del[3] = (phi - ((*k + 0.5) * d_dx[3] + d_startx[3])) / d_dx[3];

//   // finally enforce limits on del
//   if (del[1] > 1.) del[1] = 1.;
//   if (del[1] < 0.) del[1] = 0.;
//   if (del[2] > 1.) del[2] = 1.;
//   if (del[2] < 0.) del[2] = 0.;
//   if (del[3] > 1.) del[3] = 1.;
//   if (del[3] < 0.) {
//     int oldk = *k;
//     *k = d_N3-1;
//     del[3] += 1.;
//     if (del[3] < 0) {
//       printf(" ! unable to resolve X[3] coordinate to zone %d %d %g %g\n", oldk, *k, del[3], XG[3]);
//     }
//   }
// 	return;
// }


/*Given internal coordinates, X[1], X[2], X[3], we can figure out cell indexes: (i, j, k)*/
__host__ __device__ void Xtoijk(const double X[NDIM], int *i, int *j, int *k, double del[NDIM])
{
  // Route to the correct global variables depending on the compilation pass
#ifdef __CUDA_ARCH__
  const int local_METRIC = d_METRIC;
  const double local_mks3H0 = d_mks3H0;
  const double local_mks3MY1 = d_mks3MY1;
  const double local_mks3MY2 = d_mks3MY2;
  const double local_mks3MP0 = d_mks3MP0;
  const double local_mks3R0 = d_mks3R0;
  const double *local_stopx = d_stopx;
  const double *local_startx = d_startx;
  const double *local_dx = d_dx;
  const int local_N1 = d_N1;
  const int local_N2 = d_N2;
  const int local_N3 = d_N3;
#else
  const int local_METRIC = METRIC;
  const double local_mks3H0 = mks3H0;
  const double local_mks3MY1 = mks3MY1;
  const double local_mks3MY2 = mks3MY2;
  const double local_mks3MP0 = mks3MP0;
  const double local_mks3R0 = mks3R0;
  const double *local_stopx = stopx;
  const double *local_startx = startx;
  const double *local_dx = dx;
  const int local_N1 = N1;
  const int local_N2 = N2;
  const int local_N3 = N3;
#endif

  double phi;
  double XG[NDIM];
  if (local_METRIC == METRIC_eKS) {
    // the geodesics are evolved in eKS so invert through KS -> zone coordinates
    double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
    for (int mu = 0; mu < NDIM; mu++) XG[mu] = Xks[mu];
  } else if (local_METRIC == METRIC_MKS3) {
    double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
    double H0 = local_mks3H0, MY1 = local_mks3MY1, MY2 = local_mks3MY2, MP0 = local_mks3MP0;
    double KSx1 = Xks[1], KSx2 = Xks[2];
    XG[0] = Xks[0];
    XG[1] = log(Xks[1] - local_mks3R0);
    XG[2] = (-(H0*pow(KSx1,MP0)*M_PI) - pow(2.,1. + MP0)*H0*MY1*M_PI + 
      2.*H0*pow(KSx1,MP0)*MY1*M_PI + pow(2.,1. + MP0)*H0*MY2*M_PI + 
      2.*pow(KSx1,MP0)*atan(((-2.*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI))/(2.*
      H0*(-pow(KSx1,MP0) - pow(2.,1 + MP0)*MY1 + 2.*pow(KSx1,MP0)*MY1 + 
        pow(2.,1. + MP0)*MY2)*M_PI);
    XG[3] = Xks[3];
  } else {
    for (int mu = 0; mu < NDIM; mu++) XG[mu] = X[mu];
  }
  
  phi = fmod(X[3], local_stopx[3]);
  if (phi < 0.0) phi = local_stopx[3] + phi;

  *i = (int) ((X[1] - local_startx[1]) / local_dx[1] - 0.5 + 1000.) - 1000;
  *j = (int) ((X[2] - local_startx[2]) / local_dx[2] - 0.5 + 1000.) - 1000;
  *k = (int) ((phi  - local_startx[3]) / local_dx[3] - 0.5 + 1000) - 1000;  
  
  // don't allow "center zone" to be outside of [0,N*-1]. this will often fire
  // for exotic corodinate systems and occasionally for normal ones. wrap x3.
  if (*i < 0) *i = 0;
  if (*j < 0) *j = 0;
  if (*k < 0) *k = 0;
  if (*i > local_N1-2) *i = local_N1-2; 
  if (*j > local_N2-2) *j = local_N2-2; 
  if (*k > local_N3-1) *k = local_N3-1; 

  // now construct del
  del[1] = (XG[1] - ((*i + 0.5) * local_dx[1] + local_startx[1])) / local_dx[1];
  del[2] = (XG[2] - ((*j + 0.5) * local_dx[2] + local_startx[2])) / local_dx[2];
  del[3] = (phi - ((*k + 0.5) * local_dx[3] + local_startx[3])) / local_dx[3];

  // finally enforce limits on del
  if (del[1] > 1.) del[1] = 1.;
  if (del[1] < 0.) del[1] = 0.;
  if (del[2] > 1.) del[2] = 1.;
  if (del[2] < 0.) del[2] = 0.;
  if (del[3] > 1.) del[3] = 1.;
  if (del[3] < 0.) {
    int oldk = *k;
    *k = local_N3-1;
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
    if(d_METRIC == METRIC_eKS){
      // convert from zone coordinates to eKS coordinates
      X[1] = log(X[1]);
      X[2] = X[2] / M_PI;
      
    }else if(d_METRIC == METRIC_MKS3){
        double xKS[4] = { 0 };
        double x0 = X[0];
        double x1 = X[1];
        double x2 = X[2];
        double x3 = X[3];

        double H0 = d_mks3H0;
        double MY1 = d_mks3MY1;
        double MY2 = d_mks3MY2;
        double MP0 = d_mks3MP0;
        
        xKS[0] = x0;
        xKS[1] = exp(x1) + d_mks3R0;
        xKS[2] = (M_PI*(1+1./tan((H0*M_PI)/2.)*tan(H0*M_PI*(-0.5+(MY1+(pow(2,MP0)*(-MY1+MY2))/pow(exp(x1)+d_mks3R0,MP0))*(1-2*x2)+x2))))/2.;
        xKS[3] = x3;
        
        X[0] = xKS[0];
        X[1] = log(xKS[1]);
        X[2] = xKS[2] / M_PI;
        X[3] = xKS[3];
      }
	#else
		/* returns zone-centered values for coordinates */
		X[0] = startx[0];
		X[1] = startx[1] + (i + 0.5) * dx[1];
		X[2] = startx[2] + (j + 0.5) * dx[2];
		X[3] = startx[3] + (k + 0.5) * dx[3];

    if(METRIC == METRIC_eKS){
      // convert from zone coordinates to eKS coordinates
      X[1] = log(X[1]);
      X[2] = X[2]/M_PI;
    }else if(METRIC == METRIC_MKS3){
        double xKS[4] = { 0 };
        double x0 = X[0];
        double x1 = X[1];
        double x2 = X[2];
        double x3 = X[3];

        double H0 = mks3H0;
        double MY1 = mks3MY1;
        double MY2 = mks3MY2;
        double MP0 = mks3MP0;
        
        xKS[0] = x0;
        xKS[1] = exp(x1) + mks3R0;
        xKS[2] = (M_PI*(1+1./tan((H0*M_PI)/2.)*tan(H0*M_PI*(-0.5+(MY1+(pow(2,MP0)*(-MY1+MY2))/pow(exp(x1)+R0,MP0))*(1-2*x2)+x2))))/2.;
        xKS[3] = x3;
        
        X[0] = xKS[0];
        X[1] = log(xKS[1]);
        X[2] = xKS[2] / M_PI;
        X[3] = xKS[3];
      }
	#endif


	return;
}

__device__ void gcov_func_row0(const double * X, double gcov_row0[NDIM])
{
  #ifdef __CUDA_ARCH__
    double local_bhspin = d_bhspin;
  #else
    double local_bhspin = bhspin;
  #endif

  double r, s2, rho2;
  {
    double th;
    bl_coord(X, &r, &th);
    double sth, cth;
    sincos(th, &sth, &cth);
    sth = fabs(sth) + SMALL;
    s2 = sth * sth;
    rho2 = r * r + local_bhspin * local_bhspin * cth * cth;
  }
  
	gcov_row0[0] = (-1. + 2. * r / rho2);
	gcov_row0[1] = (2. * r / rho2);
  gcov_row0[2] = 0.;
	gcov_row0[3] = (-2. * local_bhspin * r * s2 / rho2);

}

__host__ __device__ void gcov_func(const double *X , double gcov[][NDIM])
{
	/* required by broken math.h */
  double gcovks[NDIM][NDIM];
  {
    int k, l;
    DLOOP gcovks[k][l] = 0.;
  }


  #ifdef __CUDA_ARCH__
    double local_bhspin = d_bhspin;
  #else
    double local_bhspin = bhspin;
  #endif

  double r, s2, rho2;
  {
    double th;
    bl_coord(X, &r, &th);
    double sth, cth;
    sincos(th, &sth, &cth);
    sth = fabs(sth) + SMALL;
    s2 = sth * sth;
    rho2 = r * r + local_bhspin * local_bhspin * cth * cth;
  }
  
	/* transformation for Kerr-Schild -> Any other metric FMKS/MKS/MKS3... */
  // tfac and pfac are 1 so in order to reduce register pressure, I'm not defining them.
  double dxdX[NDIM][NDIM] = {0.};
  {
    dxdX[0][0] = 1.;
    dxdX[1][1] = exp(X[1]);
    #ifdef __CUDA_ARCH__
      if(d_METRIC == METRIC_FMKS){
        dxdX[2][1] =  -exp(d_mks_smooth*(d_startx[1]-X[1]))*d_mks_smooth*(
            M_PI/2. -
            M_PI*X[2] +
            d_poly_norm*(2.*X[2]-1.)*(1+(pow((-1.+2*X[2])/d_poly_xt,d_poly_alpha))/(1 + d_poly_alpha)) -
            1./2.*(1. - d_hslope)*sin(2.*M_PI*X[2])
            );
         dxdX[2][2] = M_PI + (1. - d_hslope)*M_PI*cos(2.*M_PI*X[2]) +
            exp(d_mks_smooth*(d_startx[1]-X[1]))*(
            -M_PI +
            2.*d_poly_norm*(1. + pow((2.*X[2]-1.)/d_poly_xt,d_poly_alpha)/(d_poly_alpha+1.)) +
            (2.*d_poly_alpha*d_poly_norm*(2.*X[2]-1.)*pow((2.*X[2]-1.)/d_poly_xt,d_poly_alpha-1.))/((1.+d_poly_alpha)*d_poly_xt) -
            (1.-d_hslope)*M_PI*cos(2.*M_PI*X[2])
            );   
      }else{
        dxdX[2][2] = M_PI - (d_hslope - 1.)*M_PI*cos(2.*M_PI*X[2]);
      }
      
    #else
    	if(METRIC == METRIC_FMKS){
        dxdX[2][1] =  -exp(mks_smooth*(startx[1]-X[1]))*mks_smooth*(
            M_PI/2. -
            M_PI*X[2] +
            poly_norm*(2.*X[2]-1.)*(1+(pow((-1.+2*X[2])/poly_xt,poly_alpha))/(1 + poly_alpha)) -
            1./2.*(1. - hslope)*sin(2.*M_PI*X[2])
            );
         dxdX[2][2] = M_PI + (1. - hslope)*M_PI*cos(2.*M_PI*X[2]) +
            exp(mks_smooth*(startx[1]-X[1]))*(
            -M_PI +
            2.*poly_norm*(1. + pow((2.*X[2]-1.)/poly_xt,poly_alpha)/(poly_alpha+1.)) +
            (2.*poly_alpha*poly_norm*(2.*X[2]-1.)*pow((2.*X[2]-1.)/poly_xt,poly_alpha-1.))/((1.+poly_alpha)*poly_xt) -
            (1.-hslope)*M_PI*cos(2.*M_PI*X[2])
            );   
      }else{
        dxdX[2][2] = M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*X[2]);
      }
    #endif
    dxdX[3][3] = 1.;

  }

	gcovks[0][0] = (-1. + 2. * r / rho2);
	gcovks[0][1] = (2. * r / rho2);
  gcovks[0][2] = 0.;
	gcovks[0][3] = (-2. * local_bhspin * r * s2 / rho2);

	gcovks[1][0] = gcovks[0][1];
	gcovks[1][1] = (1. + 2. * r / rho2);
  gcovks[1][2] = 0.;
	gcovks[1][3] = (-local_bhspin * s2 * (1. + 2. * r / rho2));

  gcovks[2][0] = 0.;
  gcovks[2][1] = 0.;
	gcovks[2][2] = rho2;
  gcovks[2][3] = 0.;

	gcovks[3][0] = gcovks[0][3];
	gcovks[3][1] = gcovks[1][3];
  gcovks[3][2] = 0.;
	gcovks[3][3] =
	    s2 * (rho2 + local_bhspin*local_bhspin * s2 * (1. + 2. * r / rho2));

  for (int k = 0; k < NDIM; k++) {
    for (int l = 0; l < NDIM; l++) {
      gcov[k][l] = 0.;
      for (int m = 0; m < NDIM; m++) {
        for (int n = 0; n < NDIM; n++) {
          gcov[k][l] += dxdX[m][k] * dxdX[n][l] * gcovks[m][n];
        }
      }
    }
  }
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

// returns BL.{r,th} == KS.{r,th} of point with geodesic coordinates X
__host__ __device__ void bl_coord(const double *X, double *r, double *th)
{
  *r = exp(X[1]);

  #ifdef __CUDA_ARCH__
    if (d_METRIC == METRIC_eKS) {
      *r = exp(X[1]);
      *th = M_PI * X[2];
    } else if (d_METRIC == METRIC_MKS3) {
      *r = exp(X[1]) + d_mks3R0;
      *th = (M_PI*(1. + 1./tan((d_mks3H0*M_PI)/2.)*tan(d_mks3H0*M_PI*(-0.5 + (d_mks3MY1 + (pow(2.,d_mks3MP0)*(-d_mks3MY1 + d_mks3MY2))/pow(exp(X[1])+d_mks3R0,d_mks3MP0))*(1. - 2.*X[2]) + X[2]))))/2.;
    } else if (d_METRIC == METRIC_FMKS) {
      double thG = M_PI*X[2] + ((1. - d_hslope)/2.)*sin(2.*M_PI*X[2]);
      double y = 2*X[2] - 1.;
      double thJ = d_poly_norm*y*(1. + pow(y/d_poly_xt,d_poly_alpha)/(d_poly_alpha+1.)) + 0.5*M_PI;
      *th = thG + exp(d_mks_smooth*(d_startx[1] - X[1]))*(thJ - thG);
    } else {
      *th = M_PI*X[2] + ((1. - d_hslope)/2.)*sin(2.*M_PI*X[2]);
      
    }
  #else
    if (METRIC == METRIC_eKS) {
      *r = exp(X[1]);
      *th = M_PI * X[2];
    } else if (METRIC == METRIC_MKS3) {
      *r = exp(X[1]) + mks3R0;
      *th = (M_PI*(1. + 1./tan((mks3H0*M_PI)/2.)*tan(mks3H0*M_PI*(-0.5 + (mks3MY1 + (pow(2.,mks3MP0)*(-mks3MY1 + mks3MY2))/pow(exp(X[1])+mks3R0,mks3MP0))*(1. - 2.*X[2]) + X[2]))))/2.;
    } else if (METRIC == METRIC_FMKS) {
      double thG = M_PI*X[2] + ((1. - hslope)/2.)*sin(2.*M_PI*X[2]);
      double y = 2*X[2] - 1.;
      double thJ = poly_norm*y*(1. + pow(y/poly_xt,poly_alpha)/(poly_alpha+1.)) + 0.5*M_PI;
      *th = thG + exp(mks_smooth*(startx[1] - X[1]))*(thJ - thG);
    } else {
      *th = M_PI*X[2] + ((1. - hslope)/2.)*sin(2.*M_PI*X[2]);
    }
  #endif
    return;
}


__host__ __device__ void get_fluid_zone(const int i, const int j, const int k, double *  Ne, double *  Thetae, double * B,
    double Ucon[NDIM], double Bcon[NDIM], const struct of_geom *  d_geom, const double *  d_p)
{
    int l, m;
    double Ucov[NDIM], Bcov[NDIM];

    #ifdef __CUDA_ARCH__
    double local_B_unit = d_B_unit;
    double local_Ne_unit = d_Ne_unit;
    #else
    double local_B_unit = B_unit;
    double local_Ne_unit = Ne_unit;
    #endif

    {
      double Vcon[NDIM], Vfac, VdotV;

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
    }

    /* Get B and Bcov */
    {
      double UdotBp;
      double Bp[NDIM];
      Bp[1] = d_p[NPRIM_INDEX3D(B1, i, j, k)];
      Bp[2] = d_p[NPRIM_INDEX3D(B2, i, j, k)];
      Bp[3] = d_p[NPRIM_INDEX3D(B3, i, j, k)];
      UdotBp = 0.;
      for (l = 1; l < NDIM; l++)
          UdotBp += Ucov[l] * Bp[l];
      Bcon[0] = UdotBp;
      for (l = 1; l < NDIM; l++){
          Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
      }
      lower(Bcon, d_geom[SPATIAL_INDEX2D(i,j)].gcov, Bcov);
      *B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
      Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * local_B_unit;
    }

    

    *Thetae = thetae_func(d_p[NPRIM_INDEX3D(UU, i, j, k)], d_p[NPRIM_INDEX3D(KRHO, i, j, k)] , (*B)/local_B_unit, d_p[NPRIM_INDEX3D(KEL, i, j, k)]);
    *Ne = d_p[NPRIM_INDEX3D(KRHO, i, j, k)] * local_Ne_unit;

    if (*Thetae > THETAE_MAX) *Thetae = THETAE_MAX;

    double sig = pow(*B/local_B_unit,2)/(*Ne/local_Ne_unit);
    if(sig > 1. || i < 9) {
        *Thetae = SMALL;
    }

    if (isnan(*B)){
    //printf("i = %d, j = %d, k = %d\n", i, j, k);
    printf("B is nan in function get_fluid_zone\n");
    // printf( "VdotV = %le\n", VdotV);
    // printf( "Vfac = %lf\n", Vfac);
    // for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcon[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcon[a][b]);
    // for(int a = 0; a < NDIM; a++) for(int b=0;b<NDIM;b++)printf( "gcov[%d][%d]: %lf\n", a, b, d_geom[SPATIAL_INDEX2D(i,j)].gcov[a][b]);
    // printf( "Thetae: %lf\n", *Thetae);
    // printf( "Ne: %lf\n", *Ne);
    // printf( "Bp: %lf, %lf, %lf\n", Bp[1], Bp[2], Bp[3]);
    // printf( "Vcon: %lf, %lf, %lf\n", Vcon[1], Vcon[2], Vcon[3]);
    // printf( "Bcon: %lf, %lf, %lf, %lf\n Bcov: %lf, %lf, %lf %lf\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3], Bcov[0], Bcov[1], Bcov[2], Bcov[3]);
    // printf( "Ucon: %lf, %lf, %lf, %lf\n Ucov: %lf, %lf, %lf %lf\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3], Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
    }
}

__device__ int X_in_domain(const double X[NDIM])
{
    if (d_METRIC == METRIC_eKS) {
      double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
      if (Xks[1] < d_startx[1] || Xks[1] > d_stopx[1]) return 0;
    }else if (d_METRIC == METRIC_MKS3) {
      double XG[4] = { 0 };
      double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
      // if METRIC_MKS3, ignore theta boundaries
      double H0 = d_mks3H0, MY1 = d_mks3MY1, MY2 = d_mks3MY2, MP0 = d_mks3MP0;
      double KSx1 = Xks[1], KSx2 = Xks[2];
      XG[0] = Xks[0];
      XG[1] = log(Xks[1] - d_mks3R0);
      XG[2] = (-(H0*pow(KSx1,MP0)*M_PI) - pow(2,1 + MP0)*H0*MY1*M_PI +
        2*H0*pow(KSx1,MP0)*MY1*M_PI + pow(2,1 + MP0)*H0*MY2*M_PI +
        2*pow(KSx1,MP0)*atan(((-2*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI))/(2.*
        H0*(-pow(KSx1,MP0) - pow(2,1 + MP0)*MY1 + 2*pow(KSx1,MP0)*MY1 +
          pow(2,1 + MP0)*MY2)*M_PI);
      XG[3] = Xks[3];

      if (XG[1] < d_startx[1] || XG[1] > d_stopx[1]) return 0;
  } else {
    if(X[1] < d_startx[1] ||
       X[1] > d_stopx[1]  ||
       X[2] < d_startx[2] ||
       X[2] > d_stopx[2]) {
      return 0;
    }
  }

  return 1;
}

__device__ void get_fluid_params(double X[NDIM], double *Ne,
    double *Thetae, double *B, double Ucon[NDIM],
    double Ucov[NDIM], double Bcon[NDIM],
    double Bcov[NDIM], double * d_p)
{
    int i, j, k;

    //checks if it's within the grid
    if (!X_in_domain(X)) {
        *Ne = 0.;
        return;
    }


    double coeff[8];
    {
      // Finds out i and j index as well as fraction displacement del from the coordinates X[1], X[2], X[3]
      double del[NDIM];
      Xtoijk(X, &i, &j, &k, del);
      //Calculate the coeficient of displacement
      coeff[0] = (1. - del[1]) * (1. - del[2]) * (1. - del[3]);
      coeff[1] = (1. - del[1]) * (1. - del[2]) * del[3];
      coeff[2] = (1. - del[1]) * del[2] * del[3];
      coeff[3] = del[1] * del[2] * del[3];
      coeff[4] = (1. - del[1]) * del[2] * (1. - del[3]);
      coeff[5] = del[1] * (1. - del[2]) * (1. - del[3]);
      coeff[6] = del[1] * (1. - del[2]) * del[3];
      coeff[7] = del[1] * del[2] * (1. - del[3]);
    }

    double gcov[NDIM][NDIM];
    gcov_func(X, gcov);
    {
    double Vcon[NDIM], Vfac, VdotV;
    Vcon[1] = interp_scalar_pointer(d_p, U1, i, j, k, coeff);
    Vcon[2] = interp_scalar_pointer(d_p, U2, i, j, k, coeff);
    Vcon[3] = interp_scalar_pointer(d_p, U3, i, j, k, coeff);


    double gcon[NDIM][NDIM];
    gcon_func(X, gcov, gcon);

    /* Get Ucov */
    VdotV = 0.;
    for (int i = 1; i < NDIM; i++)
      for (int j = 1; j < NDIM; j++)
        VdotV += gcov[i][j] * Vcon[i] * Vcon[j];

    Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
    Ucon[0] = -Vfac * gcon[0][0];
    for (int i = 1; i < NDIM; i++){
    Ucon[i] = Vcon[i] - Vfac * gcon[0][i];
    }
    lower(Ucon, gcov, Ucov);
    }

    {
      double Bp[NDIM];
      double UdotBp;
      Bp[1] = interp_scalar_pointer(d_p, B1, i, j, k, coeff);
      Bp[2] = interp_scalar_pointer(d_p, B2, i, j, k, coeff);
      Bp[3] = interp_scalar_pointer(d_p, B3, i, j, k, coeff);
      /* Get B and Bcov */
      UdotBp = 0.;
      for (int i = 1; i < NDIM; i++)
      UdotBp += Ucov[i] * Bp[i];
      Bcon[0] = UdotBp;
      for (int i = 1; i < NDIM; i++)
      Bcon[i] = (Bp[i] + Ucon[i] * UdotBp) / Ucon[0];
      lower(Bcon, gcov, Bcov);

      *B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
      Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * d_B_unit;
    }

    //interpolate based on the displacement
    double rho = interp_scalar_pointer(d_p, KRHO, i, j, k, coeff);
    double uu = interp_scalar_pointer(d_p, UU, i, j, k, coeff);
    double kel = interp_scalar_pointer(d_p, KEL, i,j,k, coeff);
    

    *Ne = rho * d_Ne_unit;

    *Thetae = thetae_func(uu, rho, (*B)/d_B_unit, kel);
    if(*Thetae > THETAE_MAX) *Thetae = THETAE_MAX;

    double sig = (*B/d_B_unit) * (*B/d_B_unit)/(*Ne/d_Ne_unit);
    if(sig > 1.) *Thetae = SMALL;
}



__device__ double bias_func(double Te, double w, int round_scatt)
{
  double bias, max;
  max = 0.5 * w / WEIGHT_MIN;

  if (Te > 1000.) Te = 1000.;
  bias = 16. * Te * Te / (5. * d_max_tau_scatt);

  if (bias > max) bias = max;

  return bias * d_bias_guess[round_scatt];
}

__device__ void get_model_sigma_beta(const double X[NDIM], const double * d_p, double * beta, double *sigma)
{
  int i,j,k;
  double coeff[8];
  {
    // Finds out i and j index as well as fraction displacement del from the coordinates X[1], X[2], X[3]
    double del[NDIM];
    Xtoijk(X, &i, &j, &k, del);
    //Calculate the coeficient of displacement
    coeff[0] = (1. - del[1]) * (1. - del[2]) * (1. - del[3]);
    coeff[1] = (1. - del[1]) * (1. - del[2]) * del[3];
    coeff[2] = (1. - del[1]) * del[2] * del[3];
    coeff[3] = del[1] * del[2] * del[3];
    coeff[4] = (1. - del[1]) * del[2] * (1. - del[3]);
    coeff[5] = del[1] * (1. - del[2]) * (1. - del[3]);
    coeff[6] = del[1] * (1. - del[2]) * del[3];
    coeff[7] = del[1] * del[2] * (1. - del[3]);
  }
  *beta = interp_scalar_pointer(d_p, SIGMA, i, j, k, coeff);
  *sigma = interp_scalar_pointer(d_p, BETA, i, j, k, coeff);
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
    double gamma_e = d_game;
    double gamma_p = d_gamp;
    double gamma = d_gam;
    int w_electrons = d_with_electrons;
    #else
    double theta_unit = Thetae_unit;
    double local_trat_small = params.trat_small;
    double local_trat_large = params.trat_large;
    double local_beta_crit = params.beta_crit;
    double thetae_local_max = params.Thetae_max;
    double gamma_e = game;
    double gamma_p = gamp;
    double gamma = gam;
    int w_electrons = with_electrons;
    #endif
    // Gotta save beta, beta_crit, trat_large, trat_small to device memory
 
    if (w_electrons == 0) {
    //fixed tp/te ratio
      thetae = uu / rho * theta_unit;
    } else if (w_electrons == 1) {
    // howes/kawazura model from IHARM electron thermodynamics
      thetae = kel * pow(rho, gamma_e-1.) * theta_unit;
    } else if (w_electrons == 2 ) {
      double beta = uu * (gamma -1.) / 0.5 / B / B;
      double b2 = beta*beta / local_beta_crit/local_beta_crit;
      double trat = local_trat_large * b2/(1.+b2) + local_trat_small /(1.+b2);
      if (B == 0) trat = local_trat_large;
      thetae = (MP/ME) * (gamma_e-1.) * (gamma_p-1.) / ( (gamma_p-1.) + (gamma_e-1.)*trat ) * uu / rho;
    }

    return 1./(1./thetae + 1./thetae_local_max);
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

  h5io_add_blob(fid, "/fluid_header", fluid_header);
  hdf5_close_blob(fluid_header);

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
  h5io_add_data_dbl(fid, "/params/a", bhspin);
  h5io_add_data_dbl(fid, "/params/Rin", Rin);
  h5io_add_data_dbl(fid, "/params/Rout", Rout);
  h5io_add_data_dbl(fid, "/params/hslope", hslope);
  h5io_add_data_dbl(fid, "/params/t", t);
  // h5io_add_data_dbl(fid, "/params/bias", bias_guess);

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

  h5io_add_group(fid, "/params/electrons");
  if (with_electrons == 0) {
    h5io_add_data_dbl(fid, "/params/electrons/tp_over_te", params.tp_over_te);
  } else if (with_electrons == 2) {
    h5io_add_data_dbl(fid, "/params/electrons/rlow", params.trat_small);
    h5io_add_data_dbl(fid, "/params/electrons/rhigh", params.trat_large);
  }
  h5io_add_data_int(fid, "/params/electrons/type", with_electrons);

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
  fprintf(stderr, "a = %g\n", bhspin);
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
