/*

	HARM model specification routines

*/
#include "decs.h"
#include "harm_model.h"

/* mnemonics for dimensional indices */
#define TT      0
#define RR      1
#define TH      2
#define PH      3

/* EPS really ought to be related to the number of
   zones in the simulation. */
#define EPS	0.04
//#define EPS   0.01

#define RMAX	100.
#define ROULETTE	1.e4


#define Rh (1. + sqrt(1. - a * a))
// TODO:we could use this for better performance
// but when changing 'a', we must change 'Rh'
// #define Rh 1.3479852726768764

// Declare external global variables
double *harm_p;
__device__ double *d_harm_p;


/*******************************************************************************
* Host-only Functions
*
*******************************************************************************/


/* encapsulates all initialization routines */
void init_model(char *args[])
{
	/* find dimensional quantities from black hole
	   mass and its accretion rate */
	set_units(args[3]);

	fprintf(stderr, "getting simulation data...\n");
	init_harm_data(args[2]);	/* read in HARM simulation data */

	/* initialize the metric */
	fprintf(stderr, "initializing geometry...\n");
	fflush(stderr);
	init_geometry();
	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	/* make look-up table for hot cross sections */
	init_hotcross();

	/* make table for solid angle integrated emissivity and K2 */
	init_emiss_tables();

	/* make table for superphoton weights */
	unsigned long long Ns;
	sscanf(args[1], "%llu", &Ns);
	init_weight_table(Ns);

	/* make table for quick evaluation of ns_zone */
	init_nint_table();

}

/* these supply basic model data to grmonty */
void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM])
{

	int l, m;
	double Ucov[NDIM], Bcov[NDIM];
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	double sig ;

	*Ne = HARM_P(KRHO, i, j) * Ne_unit;
	*Thetae = HARM_P(UU, i, j) / (*Ne) * Ne_unit * Thetae_unit;

	Bp[1] = HARM_P(B1, i, j);
	Bp[2] = HARM_P(B2, i, j);
	Bp[3] = HARM_P(B3, i, j);

	Vcon[1] = HARM_P(U1, i, j);
	Vcon[2] = HARM_P(U2, i, j);
	Vcon[3] = HARM_P(U3, i, j);

	/* Get Ucov */
	VdotV = 0.;
	for (l = 1; l < NDIM; l++)
		for (m = 1; m < NDIM; m++)
			VdotV += geom[i][j].gcov[l][m] * Vcon[l] * Vcon[m];
	Vfac = sqrt(-1. / geom[i][j].gcon[0][0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * geom[i][j].gcon[0][0];
	for (l = 1; l < NDIM; l++)
		Ucon[l] = Vcon[l] - Vfac * geom[i][j].gcon[0][l];
	lower(Ucon, geom[i][j].gcov, Ucov);

	/* Get B and Bcov */
	UdotBp = 0.;
	for (l = 1; l < NDIM; l++)
		UdotBp += Ucov[l] * Bp[l];
	Bcon[0] = UdotBp;
	for (l = 1; l < NDIM; l++)
		Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
	lower(Bcon, geom[i][j].gcov, Bcov);


	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_unit;


	if(*Thetae > THETAE_MAX) *Thetae = THETAE_MAX ;

	sig = pow(*B/B_unit,2)/(*Ne/Ne_unit) ;
	if(sig > 1.) *Ne = 1.e-10*Ne_unit ;

}


/******************************************************************************
* Host/Device Functions
*
*******************************************************************************/

/* Current metric: modified Kerr-Schild, squashed in theta to give higher
resolution at the equator */
__host__ __device__
void gcon_func(double *X, double gcon[][NDIM])
{

	int k, l;
	double sth, cth, irho2;
	double r, th;
	double hfac;

	DLOOP gcon[k][l] = 0.;

	bl_coord(X, &r, &th);

	sth =sin(th);
	cth = cos(th);
	sth = fabs(sth) + SMALL;

	irho2 = 1. / (r * r + a * a * cth * cth);

	// transformation for Kerr-Schild -> modified Kerr-Schild
	hfac = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * X[2]);

	gcon[TT][TT] = -1. - 2. * r * irho2;
	gcon[TT][1] = 2. * irho2;

	gcon[1][TT] = gcon[TT][1];
	gcon[1][1] = irho2 * (r * (r - 2.) + a * a) / (r * r);
	gcon[1][3] = a * irho2 / r;

	gcon[2][2] = irho2 / (hfac * hfac);

	gcon[3][1] = gcon[1][3];
	gcon[3][3] = irho2 / (sth * sth);
}

__host__ __device__
void gcov_func(double *X, double gcov[][NDIM])
{
	int k, l;
	double sth, cth, s2, rho2;
	double r, th;
	double tfac, rfac, hfac, pfac;

	DLOOP gcov[k][l] = 0.;

	bl_coord(X, &r, &th);

	sth = sin(th);
	cth = cos(th);
	sth = fabs(sth) + SMALL;
	s2 = sth * sth;
	rho2 = r * r + a * a * cth * cth;

	/* transformation for Kerr-Schild -> modified Kerr-Schild */
	tfac = 1.;
	rfac = r - R0;
	hfac = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * X[2]);
	pfac = 1.;

	gcov[TT][TT] = (-1. + 2. * r / rho2) * tfac * tfac;
	gcov[TT][1] = (2. * r / rho2) * tfac * rfac;
	gcov[TT][3] = (-2. * a * r * s2 / rho2) * tfac * pfac;

	gcov[1][TT] = gcov[TT][1];
	gcov[1][1] = (1. + 2. * r / rho2) * rfac * rfac;
	gcov[1][3] = (-a * s2 * (1. + 2. * r / rho2)) * rfac * pfac;

	gcov[2][2] = rho2 * hfac * hfac;

	gcov[3][TT] = gcov[TT][3];
	gcov[3][1] = gcov[1][3];
	gcov[3][3] =
	    s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2)) * pfac * pfac;
}

/*
   connection calculated analytically for modified Kerr-Schild
   	coordinates

   this gives the connection coefficient
	\Gamma^{i}_{j,k} = conn[..][i][j][k]
   where i = {1,2,3,4} corresponds to, e.g., {t,ln(r),theta,phi}
*/
__host__ __device__
void get_connection(double X[4], double lconn[4][4][4])
{
	double r1, r2, r3, r4, sx, cx;
	double th, dthdx2, dthdx22, d2thdx22, sth, cth, sth2, cth2, sth4,
	    cth4, s2th, c2th;
	double a2, a3, a4, rho2, irho2, rho22, irho22, rho23, irho23,
	    irho23_dthdx2;
	double fac1, fac1_rho23, fac2, fac3, a2cth2, a2sth2, r1sth2,
	    a4cth4;

	r1 = exp(X[1]);
	r2 = r1 * r1;
	r3 = r2 * r1;
	r4 = r3 * r1;

	double angle_x  = 2. * M_PI * X[2];
	sx = sin(angle_x);
	cx = cos(angle_x);

	/* HARM-2D MKS */
	th = M_PI * X[2] + 0.5 * (1 - hslope) * sx;
	dthdx2 = M_PI * (1. + (1 - hslope) * cx);
	d2thdx22 = -2. * M_PI * M_PI * (1 - hslope) * sx;

	dthdx22 = dthdx2 * dthdx2;

	sth = sin(th);
	cth = cos(th);
	sth2 = sth * sth;
	r1sth2 = r1 * sth2;
	sth4 = sth2 * sth2;
	cth2 = cth * cth;
	cth4 = cth2 * cth2;
	s2th = 2. * sth * cth;
	c2th = 2 * cth2 - 1.;

	a2 = a * a;
	a2sth2 = a2 * sth2;
	a2cth2 = a2 * cth2;
	a3 = a2 * a;
	a4 = a3 * a;
	a4cth4 = a4 * cth4;

	rho2 = r2 + a2cth2;
	rho22 = rho2 * rho2;
	rho23 = rho22 * rho2;
	irho2 = 1. / rho2;
	irho22 = irho2 * irho2;
	irho23 = irho22 * irho2;
	irho23_dthdx2 = irho23 / dthdx2;

	fac1 = r2 - a2cth2;
	fac1_rho23 = fac1 * irho23;
	fac2 = a2 + 2 * r2 + a2 * c2th;
	fac3 = a2 + r1 * (-2. + r1);

	lconn[0][0][0] = 2. * r1 * fac1_rho23;
	lconn[0][0][1] = r1 * (2. * r1 + rho2) * fac1_rho23;
	lconn[0][0][2] = -a2 * r1 * s2th * dthdx2 * irho22;
	lconn[0][0][3] = -2. * a * r1sth2 * fac1_rho23;

	//lconn[0][1][0] = lconn[0][0][1];
	lconn[0][1][1] = 2. * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
	lconn[0][1][2] = -a2 * r2 * s2th * dthdx2 * irho22;
	lconn[0][1][3] =
	    a * r1 * (-r1 * (r3 + 2 * fac1) + a4cth4) * sth2 * irho23;

	//lconn[0][2][0] = lconn[0][0][2];
	//lconn[0][2][1] = lconn[0][1][2];
	lconn[0][2][2] = -2. * r2 * dthdx22 * irho2;
	lconn[0][2][3] = a3 * r1sth2 * s2th * dthdx2 * irho22;

	//lconn[0][3][0] = lconn[0][0][3];
	//lconn[0][3][1] = lconn[0][1][3];
	//lconn[0][3][2] = lconn[0][2][3];
	lconn[0][3][3] =
	    2. * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;

	lconn[1][0][0] = fac3 * fac1 / (r1 * rho23);
	lconn[1][0][1] = fac1 * (-2. * r1 + a2sth2) * irho23;
	lconn[1][0][2] = 0.;
	lconn[1][0][3] = -a * sth2 * fac3 * fac1 / (r1 * rho23);

	//lconn[1][1][0] = lconn[1][0][1];
	lconn[1][1][1] =
	    (r4 * (-2. + r1) * (1. + r1) +
	     a2 * (a2 * r1 * (1. + 3. * r1) * cth4 + a4cth4 * cth2 +
		   r3 * sth2 + r1 * cth2 * (2. * r1 + 3. * r3 -
					    a2sth2))) * irho23;
	lconn[1][1][2] = -a2 * dthdx2 * s2th / fac2;
	lconn[1][1][3] =
	    a * sth2 * (a4 * r1 * cth4 + r2 * (2 * r1 + r3 - a2sth2) +
			a2cth2 * (2. * r1 * (-1. + r2) + a2sth2)) * irho23;

	//lconn[1][2][0] = lconn[1][0][2];
	//lconn[1][2][1] = lconn[1][1][2];
	lconn[1][2][2] = -fac3 * dthdx22 * irho2;
	lconn[1][2][3] = 0.;

	//lconn[1][3][0] = lconn[1][0][3];
	//lconn[1][3][1] = lconn[1][1][3];
	//lconn[1][3][2] = lconn[1][2][3];
	lconn[1][3][3] =
	    -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);

	lconn[2][0][0] = -a2 * r1 * s2th * irho23_dthdx2;
	lconn[2][0][1] = r1 * lconn[2][0][0];
	lconn[2][0][2] = 0.;
	lconn[2][0][3] = a * r1 * (a2 + r2) * s2th * irho23_dthdx2;

	//lconn[2][1][0] = lconn[2][0][1];
	lconn[2][1][1] = r2 * lconn[2][0][0];
	lconn[2][1][2] = r2 * irho2;
	lconn[2][1][3] =
	    (a * r1 * cth * sth *
	     (r3 * (2. + r1) +
	      a2 * (2. * r1 * (1. + r1) * cth2 + a2 * cth4 +
		    2 * r1sth2))) * irho23_dthdx2;

	//lconn[2][2][0] = lconn[2][0][2];
	//lconn[2][2][1] = lconn[2][1][2];
	lconn[2][2][2] =
	    -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;
	lconn[2][2][3] = 0.;

	//lconn[2][3][0] = lconn[2][0][3];
	//lconn[2][3][1] = lconn[2][1][3];
	//lconn[2][3][2] = lconn[2][2][3];
	lconn[2][3][3] =
	    -cth * sth * (rho23 +
			  a2sth2 * rho2 * (r1 * (4. + r1) + a2cth2) +
			  2. * r1 * a4 * sth4) * irho23_dthdx2;

	lconn[3][0][0] = a * fac1_rho23;
	lconn[3][0][1] = r1 * lconn[3][0][0];
	lconn[3][0][2] = -2. * a * r1 * cth * dthdx2 / (sth * rho22);
	lconn[3][0][3] = -a2sth2 * fac1_rho23;

	//lconn[3][1][0] = lconn[3][0][1];
	lconn[3][1][1] = a * r2 * fac1_rho23;
	lconn[3][1][2] =
	    -2 * a * r1 * (a2 + 2 * r1 * (2. + r1) +
			   a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
	lconn[3][1][3] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

	//lconn[3][2][0] = lconn[3][0][2];
	//lconn[3][2][1] = lconn[3][1][2];
	lconn[3][2][2] = -a * r1 * dthdx22 * irho2;
	lconn[3][2][3] =
	    dthdx2 * (0.25 * fac2 * fac2 * cth / sth +
		      a2 * r1 * s2th) * irho22;

	//lconn[3][3][0] = lconn[3][0][3];
	//lconn[3][3][1] = lconn[3][1][3];
	//lconn[3][3][2] = lconn[3][2][3];
	lconn[3][3][3] = (-a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;

}


/*******************************************************************************
* Device-only Functions
*
*******************************************************************************/

/* stopping criterion for geodesic integrator */
/* K not referenced intentionally */
__device__
int stop_criterion(curandState_t *curandstate, struct of_photon *ph)
{
	double wmin, X1min, X1max;

	wmin = WEIGHT_MIN;	/* stop if weight is below minimum weight */

	X1min = log(Rh);	/* this is coordinate-specific; stop
				   at event horizon */
	X1max = log(RMAX);	/* this is coordinate and simulation
				   specific: stop at large distance */

	if (ph->X[1] < X1min)
		return 1;

	if (ph->X[1] > X1max) {
		if (ph->w < wmin) {
			if (curand_uniform_double(curandstate) <= 1. / ROULETTE) {
				ph->w *= ROULETTE;
			} else
				ph->w = 0.;
		}
		return 1;
	}

	if (ph->w < wmin) {
		if (curand_uniform_double(curandstate) <= 1. / ROULETTE) {
			ph->w *= ROULETTE;
		} else {
			ph->w = 0.;
			return 1;
		}
	}

	return (0);
}

/* criterion for recording photon */
__device__
int record_criterion(struct of_photon *ph)
{
	const double X1max = log(RMAX);
	/* this is coordinate and simulation
	   specific: stop at large distance */

	if (ph->X[1] > X1max)
		return (1);

	else
		return (0);

}



__device__
double stepsize(double X[NDIM], double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;

	dlx1 = EPS * X[1] / (fabs(K[1]) + SMALL);
	dlx2 = EPS * MIN (X[2], d_stopx[2] - X[2]) / (fabs(K[2]) + SMALL);
	dlx3 = EPS / (fabs(K[3]) + SMALL);

	idlx1 = 1. / (fabs(dlx1) + SMALL);
	idlx2 = 1. / (fabs(dlx2) + SMALL);
	idlx3 = 1. / (fabs(dlx3) + SMALL);

	dl = 1. / (idlx1 + idlx2 + idlx3);

	return (dl);
}

/* produces a bias (> 1) for probability of Compton scattering as a function of
local temperature */
__device__
double bias_func(double Te, double w)
{
	double bias, max ;

	max = 0.5 * w / WEIGHT_MIN;

	bias = Te*Te/(5. * d_max_tau_scatt) ;
	//bias = 100. * Te * Te / (bias_norm * max_tau_scatt);

	if (bias < TP_OVER_TE)
		bias = TP_OVER_TE;
	if (bias > max)
		bias = max;

	return bias / TP_OVER_TE;
}

/* Returns the fluid variables at the location indicated by X */
 __device__
void get_fluid_params(double X[NDIM], double gcov[NDIM][NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM])
{
	int i, j;
	double del[NDIM];
	double rho, uu;
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	double gcon[NDIM][NDIM], coeff[4];
	double sig ;

	if (X[1] < d_startx[1] ||
	    X[1] > d_stopx[1] || X[2] < d_startx[2] || X[2] > d_stopx[2]) {

		*Ne = 0.;

		return;
	}

	Xtoij(X, &i, &j, del);

	coeff[0] = (1. - del[1]) * (1. - del[2]);
	coeff[1] = (1. - del[1]) * del[2];
	coeff[2] = del[1] * (1. - del[2]);
	coeff[3] = del[1] * del[2];

	rho = interp_p_scalar(KRHO, i, j, coeff);
	uu = interp_p_scalar(UU, i, j, coeff);

	*Ne = rho * d_Ne_unit;
	*Thetae = uu / rho * d_Thetae_unit;

	Bp[1] = interp_p_scalar(B1, i, j, coeff);
	Bp[2] = interp_p_scalar(B2, i, j, coeff);
	Bp[3] = interp_p_scalar(B3, i, j, coeff);

	Vcon[1] = interp_p_scalar(U1, i, j, coeff);
	Vcon[2] = interp_p_scalar(U2, i, j, coeff);
	Vcon[3] = interp_p_scalar(U3, i, j, coeff);

	gcon_func(X, gcon);

	/* Get Ucov */
	VdotV = 0.;
	for (i = 1; i < NDIM; i++)
		for (j = 1; j < NDIM; j++)
			VdotV += gcov[i][j] * Vcon[i] * Vcon[j];
	Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * gcon[0][0];
	for (i = 1; i < NDIM; i++)
		Ucon[i] = Vcon[i] - Vfac * gcon[0][i];
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
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * d_B_unit;

	if(*Thetae > THETAE_MAX) *Thetae = THETAE_MAX ;
	sig = pow(*B/d_B_unit,2)/(*Ne/d_Ne_unit) ;
	if(sig > 1.) *Ne = 1.e-10*d_Ne_unit ;
}


#undef TT
#undef RR
#undef TH
#undef PH
