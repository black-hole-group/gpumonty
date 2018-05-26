/*
	HARM model specification routines 
*/

#include "host-device.h"
#include "host.h"
#define global
#include "harm_model.h"
#undef global

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

	Rh = 1. + sqrt(1. - a * a);

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


//double bias_func(double Te, double w)



/* 

	these supply basic model data to grmonty

*/

void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM])
{

	int l, m;
	double Ucov[NDIM], Bcov[NDIM];
	double Bp[NDIM], Vcon[NDIM], Vfac, VdotV, UdotBp;

	*Ne = p[KRHO*N1*N2+i*N2+j] * Ne_unit;
	*Thetae = p[UU*N1*N2+i*N2+j] / (*Ne) * Ne_unit * Thetae_unit;

	Bp[1] = p[B1*N1*N2+i*N2+j];
	Bp[2] = p[B2*N1*N2+i*N2+j];
	Bp[3] = p[B3*N1*N2+i*N2+j];

	Vcon[1] = p[U1*N1*N2+i*N2+j];
	Vcon[2] = p[U2*N1*N2+i*N2+j];
	Vcon[3] = p[U3*N1*N2+i*N2+j];

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

}





/* 
   Current metric: modified Kerr-Schild, squashed in theta
   to give higher resolution at the equator 
*/
void gcon_func(double *X, double gcon[][NDIM])
{

	int k, l;
	double sth, cth, irho2;
	double r, th;
	double hfac;
	/* required by broken math.h */
	void sincos(double in, double *sth, double *cth);

	DLOOP gcon[k][l] = 0.;

	bl_coord(X, &r, &th);

	sincos(th, &sth, &cth);
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


void gcov_func(double *X, double gcov[][NDIM])
{
	int k, l;
	double sth, cth, s2, rho2;
	double r, th;
	double tfac, rfac, hfac, pfac;
	/* required by broken math.h */
	void sincos(double th, double *sth, double *cth);

	DLOOP gcov[k][l] = 0.;

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



//void get_connection(double X[4], double lconn[4][4][4])


/* stopping criterion for geodesic integrator */
/* K not referenced intentionally */

//#define RMAX	100.
//#define ROULETTE	1.e4

//int stop_criterion(struct of_photon *ph)

//int record_criterion(struct of_photon *ph)

//void record_super_photon(struct of_photon *ph)


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
	fprintf(stderr, "\n");
	fprintf(stderr, "N_superph_made: %d\n", N_superph_made);
	fprintf(stderr, "N_superph_recorded: %d\n", N_superph_recorded);

	fclose(fp);

}
