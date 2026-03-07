
/*
 * GPUmonty - simcoords.cu
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


#include "hdf5_utils.h"
#include "decs.h"


static double *simcoords_x1 = NULL;
static double *simcoords_x2 = NULL;
static double *simcoords_gdet = NULL;


static double *ks_r = NULL;
static double *ks_h = NULL;
double minks_r, maxks_r;
static size_t sc_n1 = 1;
static size_t sc_n2 = 1;
static double rmax_geo = 100.;
static double rmin_geo = 1.;

static double er0 = 0;
static double h0 = 0;
static double der = 1.e-5;
static double dh = 1.e-5;
static double x1i_oob = 1.e10;

static double use_simcoords = 1;
int simcoords = 0;

#define ij2oned(i,j) ((size_t)(j+sc_n2*(i)))


// interface functions
void load_simcoord_info_from_file(double * d_ks_r, double * d_ks_h)
{
    ks_r = (double *)calloc(N1 * N2, sizeof(*ks_r));
    ks_h = (double *)calloc(N1 * N2, sizeof(*ks_h));
    simcoords_gdet = (double *)calloc(N1 * N2, sizeof(*simcoords_gdet));
    for(int a = 0; a < N1 * N2; a++) {
        int i = a / N2;
        int j = a % N2;
        double X[NDIM];

        // Coordinate k does not affect ks_r and ks_h, so we can just set it to 0 here.
        // last param set to 0 cause we want to calculate the coordinates here to set the ks_r and ks_h arrays.
        coord_wrapper(i, j, 0, X, 0, NULL, NULL);

        double r,th;
        bl_coord(X, &r, &th);

        ks_r[a] = r;
        ks_h[a] = th;
        if (a == 0 || ks_r[a] < minks_r) minks_r = ks_r[a];
        if (a == 0 || ks_r[a] > maxks_r) maxks_r = ks_r[a];
    }

    // Transfer ks_r and ks_h to device
    gpuErrchk(cudaMalloc(&d_ks_r, N1 * N2 * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_ks_h, N1 * N2 * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_ks_r, ks_r, N1 * N2 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ks_h, ks_h, N1 * N2 * sizeof(double), cudaMemcpyHostToDevice));

}


int rev_MKS3(double *xKS, double *xMKS)
{
  double KSx0=xKS[0];
  double KSx1=xKS[1];
  double KSx2=xKS[2];
  double KSx3=xKS[3];

  double R0 = mks3R0;
  double H0 = mks3H0;
  double MY1 = mks3MY1;
  double MY2 = mks3MY2;
  double MP0 = mks3MP0;
  xMKS[0] = KSx0;

  xMKS[1] = log(KSx1 - R0);

  xMKS[2]  = -H0*M_PI * pow(KSx1, MP0) - H0*MY1*M_PI * pow(2., 1.+MP0);
  xMKS[2] += 2.*H0*MY1*M_PI * pow(KSx1, MP0) + H0*MY2*M_PI*pow(2., 1.+MP0);
  xMKS[2] += 2.*pow(KSx1, MP0)*atan(((-2.*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI);
  xMKS[2] /= 2.*M_PI*H0 * (-pow(KSx1, MP0) - pow(2., 1.+MP0)*MY1 + 2*pow(KSx1, MP0)*MY1 + pow(2., 1.+MP0)*MY2);

  xMKS[3] = KSx3;

  return 0;
}

void initialize_simgrid(size_t interp_n1, size_t interp_n2, double x1i, double x1f, double x2i, double x2f) 
{
  if (use_simcoords == 0) return;

  // if we're using simgrid, the tracing should be done in eKS. verify here.
   if (!(METRIC_MKS && hslope == 1.)) {
        fprintf(stderr,
                "Assertion failed: METRIC_MKS && hslope == 1., file %s, line %d\n",
                __FILE__, __LINE__);
        fprintf(stderr, "Actual values -> METRIC_MKS: %d, hslope: %g\n", METRIC_MKS, hslope);
        exit(EXIT_FAILURE);
    }

  if (ks_r == NULL) {
    fprintf(stderr, "! must call load_ks_rh_from_file(...) before initialize_simgrid(...)\n");
    exit(EXIT_FAILURE);
  }

  sc_n1 = interp_n1;
  sc_n2 = interp_n2;

  simcoords_x1 = (double *) calloc(sc_n1*sc_n2, sizeof(*simcoords_x1));
  simcoords_x2 = (double *) calloc(sc_n1*sc_n2, sizeof(*simcoords_x2));

  // note we've made the assumption that x1i,x2i gives the left edge of the grid and
  // x1f,x2f gives the right edge. this means the range is (n1+1)*dx1,(n2+1)*dx2, so
  // that we cover the full domain. if we have an oob error when trying to determine
  // the ii,jj for interpolating, we return x1f+1, so x1f+1 must not be a valid grid
  // coordinate in the fluid model!
  double Rin = minks_r; // 1.05 * (1. + sqrt(1. - a*a));
  double Rout = maxks_r;
  double h_min = 0.;
  double h_max = M_PI;

  // set limit for tracking geodesic emission
  rmax_geo = fmin(rmax_geo, Rout);
  rmin_geo = fmax(rmin_geo, Rin);

  fprintf(stderr, "Rin Rmax %g %g %g %g  %g\n", rmin_geo, rmax_geo, Rin, Rout,  1. + sqrt(1.-bhspin*bhspin));

  x1i_oob = x1f + 1.;
  er0 = log(Rin);
  der = (log(Rout) - log(Rin)) / sc_n1;
  h0 = h_min;
  dh = (h_max - h_min) / sc_n2;

  // coordinate system is MKS, so set reasonable values here
  cstartx[0] = 0;
  cstartx[1] = log(minks_r);
  cstartx[2] = 0;
  cstartx[3] = 0;
  cstopx[0] = 0;
  cstopx[1] = log(Rout);
  cstopx[2] = 1.0;
  cstopx[3] = 2*M_PI;

#pragma omp parallel for schedule(dynamic,2) collapse(2) shared(simcoords_x1,simcoords_x2)
  for (size_t i=0; i<N1; ++i) {
    for (size_t j=0; j<N2; ++j) {
  
      double eKS[NDIM] = { 0 };
      double gridcoord[NDIM] = { 0 };

      eKS[1] = exp(er0 + der*i);
      eKS[2] = h0 + dh*j;

      int rv = 0;
      // TODO_coords: check here!
      if (METRIC_MKS3) {
        rv = rev_MKS3(eKS, gridcoord);
      } else {
        fprintf(stderr, "Error: simcoords only implemented for MKS3 metric currently.\n");
        exit(EXIT_FAILURE);
      }
      
      // force coordinate out of grid if the reverse solver failed
      if (rv != 0) {
        gridcoord[1] = x1i_oob + (x1f-x1i)*100.;
      }

      simcoords_x1[ij2oned(i,j)] = gridcoord[1];
      simcoords_x2[ij2oned(i,j)] = gridcoord[2];
    }
  }
}


// k is not used here, but we keep it for consistency with the general interface.
__host__ __device__ int simcoordijk_to_eks(const int i, const int j, const int k, double eks[NDIM], double * d_ks_r, double *d_ks_h)
{
    #ifdef __CUDA_ARCH__
        int n1 = d_N1;
        int n2 = d_N2;
    #else
        int n1 = N1;
        int n2 = N2;
    #endif

    if(i < 0 || j < 0 || i>= n1 || j >= n2) return -1;
    
    // return the eks for the gridzone at i,j,k
    eks[1] = log(d_ks_r[n2 * i + j]);
    eks[2] = d_ks_h[n2 * i + j]/M_PI;

    return 0;
}

__host__ void finalize_simgrid()
{ 
  // general housekeeping
  free(simcoords_gdet);
  free(simcoords_x2);
  free(simcoords_x1);
  free(ks_h);
  free(ks_r);
}