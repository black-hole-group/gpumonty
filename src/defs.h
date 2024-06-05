#include "config.h"
struct of_geom *geom;
gsl_rng *r;
gsl_integration_workspace *w;
#pragma omp threadprivate(r)
/*Global Variable Section*/
/* defining declarations for global variables */
int N1, N2, N3, n_within_horizon;
double F[N_ESAMP + 1], wgt[N_ESAMP + 1];
int Ns, N_scatt;
unsigned long long N_superph_recorded;

/* some coordinate parameters */
double a;
double R0, Rin, Rh, Rout, Rms;
double hslope;
double startx[NDIM], stopx[NDIM], dx[NDIM];

double dlE, lE0;
double gam;
double dMsim;
double Thetae_unit;
double max_tau_scatt, Ladv, dMact, bias_norm;

/*Model Dependent*/
double *****econ;
double *****ecov;
double ****bcon;
double ****bcov;
double ****ucon;
double ****ucov;
double * p;
double ***ne;
double ***thetae;
double ***b;



