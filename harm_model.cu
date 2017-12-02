/*
	HARM model specification routines
*/

#include "decs.h"
#define global
#include "harm_model.h"
#undef global

#include "gpu_helpers.h"

struct of_spectrum spect[N_THBINS][N_EBINS] = { };

#pragma omp threadprivate(spect)

/*

	encapsulates all initialization routines

*/

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

	Rh = 1 + sqrt(1. - a * a);

	/* make look-up table for hot cross sections */
	init_hotcross();

	/* make table for solid angle integrated emissivity and K2 */
	init_emiss_tables();

	/* make table for superphoton weights */
	init_weight_table();

	/* make table for quick evaluation of ns_zone */
	init_nint_table();

}

/*
	make super photon
*/

int n2gen = -1;
double dnmax;
int zone_i, zone_j;

void make_super_photon(struct of_photon *ph, int *quit_flag)
{

	while (n2gen <= 0) {
		n2gen = get_zone(&zone_i, &zone_j, &dnmax);
	}

	n2gen--;

	if (zone_i == N1)
		*quit_flag = 1;
	else
		*quit_flag = 0;

	if (*quit_flag != 1) {
		/* Initialize the superphoton energy, direction, weight, etc. */
		sample_zone_photon(zone_i, zone_j, dnmax, ph);
	}

	return;
}


__device__ double bias_func(double Te, double w)
{
	double bias, max, avg_num_scatt;

	max = 0.5 * w / WEIGHT_MIN;

	avg_num_scatt = N_scatt_device / (1. * N_superph_recorded_device + 1.);
	bias =
	    100. * Te * Te / (bias_norm * max_tau_scatt_device *
			      (avg_num_scatt + 2));

	if (bias < TP_OVER_TE)
		bias = TP_OVER_TE;
	if (bias > max)
		bias = max;

	return bias / TP_OVER_TE;
}

/*

	these supply basic model data to grmonty

*/

void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM])
{

	int l, m;
	double Ucov[NDIM], Bcov[NDIM];
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;

	*Ne = p[KRHO][i][j] * Ne_unit;
	*Thetae = p[UU][i][j] / (*Ne) * Ne_unit * Thetae_unit;

	Bp[1] = p[B1][i][j];
	Bp[2] = p[B2][i][j];
	Bp[3] = p[B3][i][j];

	Vcon[1] = p[U1][i][j];
	Vcon[2] = p[U2][i][j];
	Vcon[3] = p[U3][i][j];

	/* Get Ucov */
	VdotV = 0.;
	for (l = 1; l < NDIM; l++)
		for (m = 1; m < NDIM; m++)
			VdotV += geom[i * N1 + j].gcov[l * NDIM + m] * Vcon[l] * Vcon[m];
	Vfac = sqrt(-1. / geom[i * N1 + j].gcon[0*NDIM + 0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * geom[i * N1 + j].gcon[0*NDIM + 0];
	for (l = 1; l < NDIM; l++)
		Ucon[l] = Vcon[l] - Vfac * geom[i * N1 + j].gcon[0*NDIM + l];
	lower(Ucon, geom[i * N1 + j].gcov, Ucov);

	/* Get B and Bcov */
	UdotBp = 0.;
	for (l = 1; l < NDIM; l++)
		UdotBp += Ucov[l] * Bp[l];
	Bcon[0] = UdotBp;
	for (l = 1; l < NDIM; l++)
		Bcon[l] = (Bp[l] + Ucon[l] * UdotBp) / Ucon[0];
	lower(Bcon, geom[i * N1 + j].gcov, Bcov);

	*B = sqrt(Bcon[0] * Bcov[0] + Bcon[1] * Bcov[1] +
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_unit;

}

void get_fluid_params(double X[NDIM], double gcov[NDIM * NDIM], double *Ne,
		      double *Thetae, double *B, double Ucon[NDIM],
		      double Ucov[NDIM], double Bcon[NDIM],
		      double Bcov[NDIM])
{
	int i, j;
	double del[NDIM];
	double rho, uu;
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;
	double gcon[NDIM*NDIM], coeff[4];
	double interp_scalar(double **var, int i, int j, double del[4], int N1);

	if (X[1] < startx[1] ||
	    X[1] > stopx[1] || X[2] < startx[2] || X[2] > stopx[2]) {

		*Ne = 0.;

		return;
	}

	Xtoij(X, &i, &j, del, startx, dx, N1, N2);

	coeff[0] = (1. - del[1]) * (1. - del[2]);
	coeff[1] = (1. - del[1]) * del[2];
	coeff[2] = del[1] * (1. - del[2]);
	coeff[3] = del[1] * del[2];

	rho = interp_scalar(p[KRHO], i, j, coeff, N1);
	uu = interp_scalar(p[UU], i, j, coeff, N1);

	*Ne = rho * Ne_unit;
	*Thetae = uu / rho * Thetae_unit;

	Bp[1] = interp_scalar(p[B1], i, j, coeff, N1);
	Bp[2] = interp_scalar(p[B2], i, j, coeff, N1);
	Bp[3] = interp_scalar(p[B3], i, j, coeff, N1);

	Vcon[1] = interp_scalar(p[U1], i, j, coeff, N1);
	Vcon[2] = interp_scalar(p[U2], i, j, coeff, N1);
	Vcon[3] = interp_scalar(p[U3], i, j, coeff, N1);

	gcon_func(X, gcon);

	/* Get Ucov */
	VdotV = 0.;
	for (i = 1; i < NDIM; i++)
		for (j = 1; j < NDIM; j++)
			VdotV += gcov[i * NDIM + j] * Vcon[i] * Vcon[j];
	Vfac = sqrt(-1. / gcon[0*NDIM + 0] * (1. + fabs(VdotV)));
	Ucon[0] = -Vfac * gcon[0*NDIM + 0];
	for (i = 1; i < NDIM; i++)
		Ucon[i] = Vcon[i] - Vfac * gcon[0*NDIM + i];
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
		  Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3]) * B_unit;

}


/*
   Current metric: modified Kerr-Schild, squashed in theta
   to give higher resolution at the equator
*/

/* mnemonics for dimensional indices */
#define TT      0
#define RR      1
#define TH      2
#define PH      3

__device__
void gcon_func(double *X, double gcon[NDIM*NDIM])
{

	int k, l;
	double sth, cth, irho2;
	double r, th;
	double hfac;
	/* required by broken math.h */
	// void sincos(double in, double *sth, double *cth);

	DLOOP gcon[k*NDIM + l] = 0.;

	bl_coord(X, &r, &th);

	sincos(th, &sth, &cth);
	sth = fabs(sth) + SMALL;

	irho2 = 1. / (r * r + a * a * cth * cth);

	// transformation for Kerr-Schild -> modified Kerr-Schild
	hfac = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * X[2]);

	gcon[TT*NDIM + TT] = -1. - 2. * r * irho2;
	gcon[TT*NDIM + 1] = 2. * irho2;

	gcon[1*NDIM + TT] = gcon[TT*NDIM + 1];
	gcon[1*NDIM + 1] = irho2 * (r * (r - 2.) + a * a) / (r * r);
	gcon[1*NDIM + 3] = a * irho2 / r;

	gcon[2*NDIM + 2] = irho2 / (hfac * hfac);

	gcon[3*NDIM + 1] = gcon[1*NDIM + 3];
	gcon[3*NDIM + 3] = irho2 / (sth * sth);
}

__device__
void gcov_func(double *X, double gcov[NDIM * NDIM])
{
	int k, l;
	double sth, cth, s2, rho2;
	double r, th;
	double tfac, rfac, hfac, pfac;
	/* required by broken math.h */
	// void sincos(double th, double *sth, double *cth);

	DLOOP gcov[k*NDIM + l] = 0.;

	bl_coord(X, &r, &th);

	sincos(th, &sth, &cth);
	sth = fabs(sth) + SMALL;
	s2 = sth * sth;
	rho2 = r * r + a * a * cth * cth;

	/* transformation for Kerr-Schild -> modified Kerr-Schild */
	tfac = 1.;
	rfac = r - R0;
	hfac = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * X[2]);
	pfac = 1.;

	gcov[TT * NDIM + TT] = (-1. + 2. * r / rho2) * tfac * tfac;
	gcov[TT * NDIM + 1] = (2. * r / rho2) * tfac * rfac;
	gcov[TT * NDIM + 3] = (-2. * a * r * s2 / rho2) * tfac * pfac;

	gcov[1 * NDIM + TT] = gcov[TT * NDIM + 1];
	gcov[1 * NDIM + 1] = (1. + 2. * r / rho2) * rfac * rfac;
	gcov[1 * NDIM + 3] = (-a * s2 * (1. + 2. * r / rho2)) * rfac * pfac;

	gcov[2 * NDIM + 2] = rho2 * hfac * hfac;

	gcov[3 * NDIM + TT] = gcov[TT * NDIM + 3];
	gcov[3 * NDIM + 1] = gcov[1 * NDIM + 3];
	gcov[3 * NDIM + 3] =
	    s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2)) * pfac * pfac;
}

#undef TT
#undef RR
#undef TH
#undef PH

/*

   connection calculated analytically for modified Kerr-Schild
   	coordinates


   this gives the connection coefficient
	\Gamma^{i}_{j,k} = conn[..][i][j][k]
   where i = {1,2,3,4} corresponds to, e.g., {t,ln(r),theta,phi}
*/

void get_connection(double X[4], double lconn[4][4][4])
{
	cuda_get_connection(X, lconn);
}

/* stopping criterion for geodesic integrator */
/* K not referenced intentionally */

#define RMAX	100.
#define ROULETTE	1.e4
int stop_criterion(struct of_photon *ph)
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
			if (monty_rand() <= 1. / ROULETTE) {
				ph->w *= ROULETTE;
			} else
				ph->w = 0.;
		}
		return 1;
	}

	if (ph->w < wmin) {
		if (monty_rand() <= 1. / ROULETTE) {
			ph->w *= ROULETTE;
		} else {
			ph->w = 0.;
			return 1;
		}
	}

	return (0);
}

/* criterion for recording photon */

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


/* EPS really ought to be related to the number of
   zones in the simulation. */
#define EPS	0.04
//#define EPS   0.01


double stepsize(double X[NDIM], double K[NDIM])
{
	double dl, dlx1, dlx2, dlx3;
	double idlx1, idlx2, idlx3;

	dlx1 = EPS * X[1] / (fabs(K[1]) + SMALL);
	dlx2 = EPS * GSL_MIN(X[2], stopx[2] - X[2]) / (fabs(K[2]) + SMALL);
	dlx3 = EPS / (fabs(K[3]) + SMALL);

	idlx1 = 1. / (fabs(dlx1) + SMALL);
	idlx2 = 1. / (fabs(dlx2) + SMALL);
	idlx3 = 1. / (fabs(dlx3) + SMALL);

	dl = 1. / (idlx1 + idlx2 + idlx3);

	return (dl);
}

/*
	record contribution of super photon to spectrum.

	This routine should make minimal assumptions about the
	coordinate system.

*/
void record_super_photon(struct of_photon *ph)
{
	double lE, dx2;
	int iE, ix2;

	if (isnan(ph->w) || isnan(ph->E)) {
		fprintf(stderr, "record isnan: %g %g\n", ph->w, ph->E);
		return;
	}
#pragma omp critical (MAXTAU)
	{
		if (ph->tau_scatt > max_tau_scatt)
			max_tau_scatt = ph->tau_scatt;
	}
	/* currently, bin in x2 coordinate */

	/* get theta bin, while folding around equator */
	dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);
	if (ph->X[2] < 0.5 * (startx[2] + stopx[2]))
		ix2 = (int) (ph->X[2] / dx2);
	else
		ix2 = (int) ((stopx[2] - ph->X[2]) / dx2);

	/* check limits */
	if (ix2 < 0 || ix2 >= N_THBINS)
		return;

	/* get energy bin */
	lE = log(ph->E);
	iE = (int) ((lE - lE0) / dlE + 2.5) - 2;	/* bin is centered on iE*dlE + lE0 */

	/* check limits */
	if (iE < 0 || iE >= N_EBINS)
		return;

#pragma omp atomic
	N_superph_recorded++;
	/* cudaMemcpyFromSymbol(x_h, x_d, size, 0, cudaMemcpyDeviceToHost); */
#pragma omp atomic
	N_scatt += ph->nscatt;
	/* cudaMemcpyFromSymbol(x_h, x_d, size, 0, cudaMemcpyDeviceToHost); */

	/* sum in photon */
	spect[ix2][iE].dNdlE += ph->w;
	spect[ix2][iE].dEdlE += ph->w * ph->E;
	spect[ix2][iE].tau_abs += ph->w * ph->tau_abs;
	spect[ix2][iE].tau_scatt += ph->w * ph->tau_scatt;
	spect[ix2][iE].X1iav += ph->w * ph->X1i;
	spect[ix2][iE].X2isq += ph->w * (ph->X2i * ph->X2i);
	spect[ix2][iE].X3fsq += ph->w * (ph->X[3] * ph->X[3]);
	spect[ix2][iE].ne0 += ph->w * (ph->ne0);
	spect[ix2][iE].b0 += ph->w * (ph->b0);
	spect[ix2][iE].thetae0 += ph->w * (ph->thetae0);
	spect[ix2][iE].nscatt += ph->nscatt;
	spect[ix2][iE].nph += 1.;

}

struct of_spectrum shared_spect[N_THBINS][N_EBINS] = { };

void omp_reduce_spect()
{
/* Combine partial spectra from each OpenMP process		*
 * Inefficient, but only called once so doesn't matter	*/

	int i, j;

#pragma omp critical (UPDATE_SPECT)
	{
		for (i = 0; i < N_THBINS; i++) {
			for (j = 0; j < N_EBINS; j++) {
				shared_spect[i][j].dNdlE +=
				    spect[i][j].dNdlE;
				shared_spect[i][j].dEdlE +=
				    spect[i][j].dEdlE;
				shared_spect[i][j].tau_abs +=
				    spect[i][j].tau_abs;
				shared_spect[i][j].tau_scatt +=
				    spect[i][j].tau_scatt;
				shared_spect[i][j].X1iav +=
				    spect[i][j].X1iav;
				shared_spect[i][j].X2isq +=
				    spect[i][j].X2isq;
				shared_spect[i][j].X3fsq +=
				    spect[i][j].X3fsq;
				shared_spect[i][j].ne0 += spect[i][j].ne0;
				shared_spect[i][j].b0 += spect[i][j].b0;
				shared_spect[i][j].thetae0 +=
				    spect[i][j].thetae0;
				shared_spect[i][j].nscatt +=
				    spect[i][j].nscatt;
				shared_spect[i][j].nph += spect[i][j].nph;
			}
		}
	}
#pragma omp barrier
#pragma omp master
	{
		for (i = 0; i < N_THBINS; i++) {
			for (j = 0; j < N_EBINS; j++) {
				spect[i][j].dNdlE =
				    shared_spect[i][j].dNdlE;
				spect[i][j].dEdlE =
				    shared_spect[i][j].dEdlE;
				spect[i][j].tau_abs =
				    shared_spect[i][j].tau_abs;
				spect[i][j].tau_scatt =
				    shared_spect[i][j].tau_scatt;
				spect[i][j].X1iav =
				    shared_spect[i][j].X1iav;
				spect[i][j].X2isq =
				    shared_spect[i][j].X2isq;
				spect[i][j].X3fsq =
				    shared_spect[i][j].X3fsq;
				spect[i][j].ne0 = shared_spect[i][j].ne0;
				spect[i][j].b0 = shared_spect[i][j].b0;
				spect[i][j].thetae0 =
				    shared_spect[i][j].thetae0;
				spect[i][j].nscatt =
				    shared_spect[i][j].nscatt;
				spect[i][j].nph = shared_spect[i][j].nph;
			}
		}
	}
}

/*

	output spectrum to file

*/

#define SPECTRUM_FILE_NAME	"grmonty.spec"

void report_spectrum(int N_superph_made)
{
	int i, j;
	double dx2, dOmega, nuLnu, tau_scatt, L;
	FILE *fp;

	fp = fopen(SPECTRUM_FILE_NAME, "w");
	if (fp == NULL) {
		fprintf(stderr, "trouble opening spectrum file\n");
		exit(0);
	}

	/* output */
	max_tau_scatt = 0.;
	L = 0.;
	for (i = 0; i < N_EBINS; i++) {

		/* output log_10(photon energy/(me c^2)) */
		fprintf(fp, "%10.5g ", (i * dlE + lE0) / M_LN10);

		for (j = 0; j < N_THBINS; j++) {

			/* convert accumulated photon number in each bin
			   to \nu L_\nu, in units of Lsun */
			dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);

			/* factor of 2 accounts for folding around equator */
			dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);

			nuLnu =
			    (ME * CL * CL) * (4. * M_PI / dOmega) * (1. /
								     dlE);

			nuLnu *= spect[j][i].dEdlE;
			nuLnu /= LSUN;

			tau_scatt =
			    spect[j][i].tau_scatt / (spect[j][i].dNdlE +
						     SMALL);
			fprintf(fp,
				"%10.5g %10.5g %10.5g %10.5g %10.5g %10.5g ",
				nuLnu,
				spect[j][i].tau_abs / (spect[j][i].dNdlE +
						       SMALL), tau_scatt,
				spect[j][i].X1iav / (spect[j][i].dNdlE +
						     SMALL),
				sqrt(fabs
				     (spect[j][i].X2isq /
				      (spect[j][i].dNdlE + SMALL))),
				sqrt(fabs
				     (spect[j][i].X3fsq /
				      (spect[j][i].dNdlE + SMALL)))
			    );

			if (tau_scatt > max_tau_scatt)
				max_tau_scatt = tau_scatt;

			L += nuLnu * dOmega * dlE;
		}
		fprintf(fp, "\n");
	}
	fprintf(stderr,
		"luminosity %g, dMact %g, efficiency %g, L/Ladv %g, max_tau_scatt %g\n",
		L, dMact * M_unit / T_unit / (MSUN / YEAR),
		L * LSUN / (dMact * M_unit * CL * CL / T_unit),
		L * LSUN / (Ladv * M_unit * CL * CL / T_unit),
		max_tau_scatt);
	printf("\n");
	printf("N_superph_made: %d\n", N_superph_made);
	printf("N_superph_recorded: %d\n", N_superph_recorded);

	fclose(fp);

}
