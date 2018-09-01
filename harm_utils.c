#include "decs.h"
#include "harm_model.h"

/* Harm globals */

static double wgt[N_ESAMP + 1];

/** HARM utilities **/

/********************************************************************

				Interpolation routines

 ********************************************************************/

/*
 * Finds interpolated value of var at location given by i,j.
 * What is coeff?
 */
double interp_scalar(double **var, int i, int j, double coeff[4])
{

	double interp;

	interp =
	    var[i][j] * coeff[0] +
	    var[i][j + 1] * coeff[1] +
	    var[i + 1][j] * coeff[2] + var[i + 1][j + 1] * coeff[3];

	return interp;
}

double lnu_min, lnu_max, dlnu;

static void init_linear_interp_weight()
{

	lnu_min = log(NUMIN);
	lnu_max = log(NUMAX);
	dlnu = (lnu_max - lnu_min) / (N_ESAMP);
}

static double linear_interp_weight(double nu)
{

	int i;
	double di, lnu;

	lnu = log(nu);

	di = (lnu - lnu_min) / dlnu;
	i = (int) di;
	di = di - i;

	return exp((1. - di) * wgt[i] + di * wgt[i + 1]);

}


/***********************************************************************************

					End interpolation routines

 ***********************************************************************************/


#define JCST	(M_SQRT2*EE*EE*EE/(27*ME*CL*CL))
void init_weight_table(unsigned long long Ns)
{

	int i, j, l, lstart, lend, myid, nthreads;
	double Ne, Thetae, B, K2;
	double sum[N_ESAMP + 1], nu[N_ESAMP + 1];
	double fac, sfac;
	double Ucon[NDIM], Bcon[NDIM];

	fprintf(stderr, "Building table for superphoton weights\n");
	fflush(stderr);

	/*      Set up interpolation */
	init_linear_interp_weight();

#pragma acc parallel loop
	for (i = 0; i <= N_ESAMP; i++) {
		sum[i] = 0.;
		nu[i] = exp(i * dlnu + lnu_min);
	}

	sfac = dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit;

	// This part use to be parallel. It was made sequential for simplicity. Can be parallelized again
	// later.
	for (i = 0; i < N1; i++)
		for (j = 0; j < N2; j++) {
			get_fluid_zone(i, j, &Ne, &Thetae, &B,
				       Ucon, Bcon);
			if (Ne == 0. || Thetae < THETAE_MIN)
				continue;
			K2 = K2_eval(Thetae);
			fac =
			    (JCST * Ne * B * Thetae * Thetae /
			     K2) * sfac * geom[i][j].g;
			for (l = 0; l < N_ESAMP+1; l++)
				sum[l] += fac * F_eval(Thetae, B, nu[l]);
		}
    #pragma acc parallel loop
	for (i = 0; i <= N_ESAMP; i++)
		wgt[i] = log(sum[i] / (HPL * Ns) + WEIGHT_MIN);

	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	return;
}

#undef JCST

#define BTHSQMIN	(1.e-4)
#define BTHSQMAX	(1.e9)
#define	NINT		(40000)

double lb_min, dlb;
double nint[NINT + 1];
double dndlnu_max[NINT + 1];
void init_nint_table(void)
{

	int i, j;
	double Bmag, dn;
	static int firstc = 1;

	if (firstc) {
		lb_min = log(BTHSQMIN);
		dlb = log(BTHSQMAX / BTHSQMIN) / NINT;
		firstc = 0;
	}

	for (i = 0; i <= NINT; i++) {
		nint[i] = 0.;
		Bmag = exp(i * dlb + lb_min);
		dndlnu_max[i] = 0.;
		for (j = 0; j < N_ESAMP; j++) {
			dn = F_eval(1., Bmag,
				    exp(j * dlnu +
					lnu_min)) / (exp(wgt[j]) + 1.e-100);
			if (dn > dndlnu_max[i])
				dndlnu_max[i] = dn;
			nint[i] += dlnu * dn;
		}
		nint[i] *= dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit
		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		    * 1. / HPL;
		nint[i] = log(nint[i]);
		dndlnu_max[i] = log(dndlnu_max[i]);
	}

	return;
}

void init_zone(int i, int j, unsigned long long*nz, double *dnmax, unsigned long long Ns)
{

	int l;
	double Ne, Thetae, Bmag, lbth;
	double dl, dn, ninterp, K2;
	double Ucon[NDIM], Bcon[NDIM];

	get_fluid_zone(i, j, &Ne, &Thetae, &Bmag, Ucon, Bcon);

	if (Ne == 0. || Thetae < THETAE_MIN) {
		*nz = 0;
		*dnmax = 0.;
		return;
	}

	lbth = log(Bmag * Thetae * Thetae);

	dl = (lbth - lb_min) / dlb;
	l = (int) dl;
	dl = dl - l;
	if (l < 0) {
		*dnmax = 0.;
		*nz = 0;
		return;
	} else if (l >= NINT) {

		// fprintf(stderr,
		// 	"warning: outside of nint table range %g...change in harm_utils.c\n",
		// 	Bmag * Thetae * Thetae);
		// fprintf(stderr,"%g %g %g %g\n",Bmag,Thetae,lbth,(lbth - lb_min)/dlb) ;
		ninterp = 0.;
		*dnmax = 0.;
		for (l = 0; l <= N_ESAMP; l++) {
			dn = F_eval(Thetae, Bmag,
				    exp(j * dlnu +
					lnu_min)) / exp(wgt[l]);
			if (dn > *dnmax)
				*dnmax = dn;
			ninterp += dlnu * dn;
		}
		ninterp *= dx[1] * dx[2] * dx[3] * L_unit * L_unit * L_unit
		    * M_SQRT2 * EE * EE * EE / (27. * ME * CL * CL)
		    * 1. / HPL;
	} else {
		if (isinf(nint[l]) || isinf(nint[l + 1])) {
			ninterp = 0.;
			*dnmax = 0.;
		} else {
			ninterp =
			    exp((1. - dl) * nint[l] + dl * nint[l + 1]);
			*dnmax =
			    exp((1. - dl) * dndlnu_max[l] +
				dl * dndlnu_max[l + 1]);
		}
	}

	K2 = K2_eval(Thetae);
	if (K2 == 0.) {
		*nz = 0;
		*dnmax = 0.;
		return;
	}

	double dnz = geom[i][j].g * Ne * Bmag * Thetae * Thetae * ninterp / K2;

	if (dnz > Ns * log(NUMAX / NUMIN)) {
		// fprintf(stderr,
		// 	"Something very wrong in zone %d %d: \nB=%g  Thetae=%g  K2=%g  ninterp=%g\n\n",
		// 	i, j, Bmag, Thetae, K2, ninterp);
		*nz = 0;
		*dnmax = 0.;
	} else {
		// Randomly round decimal up or down
		if (fmod(dnz, 1.) > cpu_rng_uniforme()) *nz = (int) dnz + 1;
		else *nz = (int) dnz;
	}

	return;
}

void sample_zone_photon(int i, int j, double dnmax, struct of_photon *ph, int first_zone_photon)
{
/* Set all initial superphoton attributes */

	int l;
	double K_tetrad[NDIM], tmpK[NDIM], E, Nln;
	double nu, th, cth, sth, phi, sphi, cphi, jmax, weight;
	double Ne, Thetae, Bmag, Ucon[NDIM], Bcon[NDIM], bhat[NDIM];
	static double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];

	coord(i, j, ph->X);

	Nln = lnu_max - lnu_min;

	get_fluid_zone(i, j, &Ne, &Thetae, &Bmag, Ucon, Bcon);

	/* Sample from superphoton distribution in current simulation zone */
	do {
		nu = exp(cpu_rng_uniforme() * Nln + lnu_min);
		weight = linear_interp_weight(nu);
	} while (cpu_rng_uniforme() >
		 (F_eval(Thetae, Bmag, nu) / weight) / dnmax);

	ph->w = weight;
	jmax = jnu_synch(nu, Ne, Thetae, Bmag, M_PI / 2.);
	do {
		cth = 2. * cpu_rng_uniforme() - 1.;
		th = acos(cth);

	} while (cpu_rng_uniforme() >
		 jnu_synch(nu, Ne, Thetae, Bmag, th) / jmax);

	sth = sqrt(1. - cth * cth);
	phi = 2. * M_PI * cpu_rng_uniforme();
	cphi = cos(phi);
	sphi = sin(phi);

	E = nu * HPL / (ME * CL * CL);
	K_tetrad[0] = E;
	K_tetrad[1] = E * cth;
	K_tetrad[2] = E * cphi * sth;
	K_tetrad[3] = E * sphi * sth;

	/*
	if(E > 1.e-4) fprintf(stdout,"HOT: %d %d %g %g %g %g %g\n",
		i,j,E/(0.22*(EE*Bmag/(2.*M_PI*ME*CL))*(HPL/(ME*CL*CL))*Thetae*Thetae),
		ph->X[1],ph->X[2], Thetae,Bmag) ;
	*/

	if (first_zone_photon) {	/* first photon created in this zone, so make the tetrad */
		if (Bmag > 0.) {
			for (l = 0; l < NDIM; l++)
				bhat[l] = Bcon[l] * B_unit / Bmag;
		} else {
			for (l = 1; l < NDIM; l++)
				bhat[l] = 0.;
			bhat[1] = 1.;
		}
		make_tetrad(Ucon, bhat, geom[i][j].gcov, Econ, Ecov);
	}

	tetrad_to_coordinate(Econ, K_tetrad, ph->K);

	K_tetrad[0] *= -1.;
	tetrad_to_coordinate(Ecov, K_tetrad, tmpK);

	ph->E = ph->E0 = ph->E0s = -tmpK[0];
	ph->L = tmpK[3];
	ph->tau_scatt = 0.;
	ph->tau_abs = 0.;
	ph->X1i = ph->X[1];
	ph->X2i = ph->X[2];
	ph->nscatt = 0;
	ph->ne0 = Ne;
	ph->b0 = Bmag;
	ph->thetae0 = Thetae;

	return;
}

void Xtoij(double X[NDIM], int *i, int *j, double del[NDIM])
{

	*i = (int) ((X[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
	*j = (int) ((X[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;

	if (*i < 0) {
		*i = 0;
		del[1] = 0.;
	} else if (*i > N1 - 2) {
		*i = N1 - 2;
		del[1] = 1.;
	} else {
		del[1] = (X[1] - ((*i + 0.5) * dx[1] + startx[1])) / dx[1];
	}

	if (*j < 0) {
		*j = 0;
		del[2] = 0.;
	} else if (*j > N2 - 2) {
		*j = N2 - 2;
		del[2] = 1.;
	} else {
		del[2] = (X[2] - ((*j + 0.5) * dx[2] + startx[2])) / dx[2];
	}

	return;
}

/* return boyer-lindquist coordinate of point */
void bl_coord(double *X, double *r, double *th)
{

	*r = exp(X[1]) + R0;
	*th = M_PI * X[2] + ((1. - hslope) / 2.) * sin(2. * M_PI * X[2]);

	return;
}

void coord(int i, int j, double *X)
{

	/* returns zone-centered values for coordinates */
	X[0] = startx[0];
	X[1] = startx[1] + (i + 0.5) * dx[1];
	X[2] = startx[2] + (j + 0.5) * dx[2];
	X[3] = startx[3];

	return;
}


void set_units(char *munitstr)
{
	double MBH;

	sscanf(munitstr, "%lf", &M_unit);

	/** from this, calculate units of length, time, mass,
	    and derivative units **/
	MBH = 4.6e6 * MSUN ;
	L_unit = GNEWT * MBH / (CL * CL);
	T_unit = L_unit / CL;

	fprintf(stderr, "\nUNITS\n");
	fprintf(stderr, "L,T,M: %g %g %g\n", L_unit, T_unit, M_unit);

	RHO_unit = M_unit / pow(L_unit, 3);
	U_unit = RHO_unit * CL * CL;
	B_unit = CL * sqrt(4. * M_PI * RHO_unit);

	fprintf(stderr, "rho,u,B: %g %g %g\n", RHO_unit, U_unit, B_unit);

	Ne_unit = RHO_unit / (MP + ME);

	// max_tau_scatt = (6. * L_unit) * RHO_unit * 0.4;
    // max_tau_scatt = 0.001295;
	max_tau_scatt = 0.0024;

	fprintf(stderr, "max_tau_scatt: %g\n", max_tau_scatt);

}

/* set up all grid functions */
void init_geometry()
{
	int i, j;
	double X[NDIM];

	for (i = 0; i < N1; i++) {
		for (j = 0; j < N2; j++) {

			/* zone-centered */
			coord(i, j, X);

			gcov_func(X, geom[i][j].gcov);

			geom[i][j].g = gdet_func(geom[i][j].gcov);

			gcon_func(X, geom[i][j].gcon);

		}
	}

	/* done! */
}

/*

	return solid angle between points x2i, x2f
	and over all x3.

*/

double dOmega_func(double x2i, double x2f)
{
	double dO;

	dO = 2. * M_PI *
	    (-cos(M_PI * x2f + 0.5 * (1. - hslope) * sin(2 * M_PI * x2f))
	     + cos(M_PI * x2i + 0.5 * (1. - hslope) * sin(2 * M_PI * x2i))
	    );

	return (dO);
}

static void *malloc_rank1(int n1, int size)
{
	void *A;

	if ((A = (void *) malloc(n1 * size)) == NULL) {
		// fprintf(stderr, "malloc failure in malloc_rank1\n");
		exit(123);
	}

	return A;
}


static void **malloc_rank2(int n1, int n2, int size)
{

	void **A;
	int i;

	if ((A = (void **) malloc(n1 * sizeof(void *))) == NULL) {
		// fprintf(stderr, "malloc failure in malloc_rank2\n");
		exit(124);
	}

	for (i = 0; i < n1; i++) {
		A[i] = malloc_rank1(n2, size);
	}

	return A;
}


static double **malloc_rank2_cont(int n1, int n2)
{

	double **A;
	double *space;
	int i;

	space = malloc_rank1(n1 * n2, sizeof(double));

	A = malloc_rank1(n1, sizeof(double *));

	for (i = 0; i < n1; i++)
		A[i] = &(space[i * n2]);

	return A;
}

void init_storage(void)
{
	int i;

	p = malloc_rank1(NPRIM, sizeof(double *));
	for (i = 0; i < NPRIM; i++)
		p[i] = (double **) malloc_rank2_cont(N1, N2);
	geom =
	    (struct of_geom **) malloc_rank2(N1, N2,
					     sizeof(struct of_geom));
	#pragma acc enter data create (p[:NPRIM][:N1][:N2])
	return;
}
