//#include "decs.h"

/* 

   given energy of photon in fluid rest frame w, in units of electron rest mass
   energy, and temperature of plasma, again in electron rest-mass units, return hot
   cross section in cgs.
   
   This has been checked against Wienke's Table 1, with some disagreement at
   the one part in 10^{-3} level, see wienke_table_1 in the subdirectory hotcross.
   It is not clear what this is due to, but Table 1 does appear to have been evaluated
   using Monte Carlo integration (!).

   A better way to do this would be to make a table in w*thetae and w/thetae; most
   	of the variation is accounted for by w*thetae.
   
*/



/* normalized (per unit proper electron number density)
   electron distribution */
__device__
double d_dNdgammae(double thetae, double gammae)
{
	double K2f;

	if (thetae > 1.e-2) {
		K2f = cu_sf_bessel_Kn(2, 1. / thetae) * exp(1. / thetae);
	} else {
		K2f = sqrt(M_PI * thetae / 2.);
	}

	return ((gammae * sqrt(gammae * gammae - 1.) / (thetae * K2f)) *
		exp(-(gammae - 1.) / thetae));
}




__device__
double d_hc_klein_nishina(double we)
{
	double sigma;

	if (we < 1.e-3)
		return (1. - 2. * we);

	sigma = (3. / 4.) * (2. / (we * we) +
			     (1. / (2. * we) -
			      (1. + we) / (we * we * we)) * log(1. +
								2. * we) +
			     (1. + we) / ((1. + 2. * we) * (1. + 2. * we))
	    );

	return (sigma);
}




__device__
double d_boostcross(double w, double mue, double gammae)
{
	double we, boostcross, v;
	// double d_hc_klein_nishina(double we);

	/* energy in electron rest frame */
	v = sqrt(gammae * gammae - 1.) / gammae;
	we = w * gammae * (1. - mue * v);

	boostcross = d_hc_klein_nishina(we) * (1. - mue * v);

	if (boostcross > 2) {
		printf("w,mue,gammae: %g %g %g\n", w, mue, gammae);
		printf("v,we, boostcross: %g %g %g\n", v, we, boostcross);
		printf("kn: %g %g %g\n", v, we, boostcross);
	}

	if (isnan(boostcross)) {
		printf("isnan: %g %g %g\n", w, mue, gammae);
		return (0);
	}

	return (boostcross);
}





__device__
double d_total_compton_cross_num(double w, double thetae)
{
	double dmue, dgammae, mue, gammae, f, cross;
	// __device__ double d_dNdgammae(double thetae, double gammae);
	// __device__ double d_boostcross(double w, double mue, double gammae);
	// __device__ double d_hc_klein_nishina(double we);

	if (isnan(w)) {
		printf("compton cross isnan: %g %g\n", w, thetae);
		return (0.);
	}

	/* check for easy-to-do limits */
	if (thetae < MINT && w < MINW)
		return (SIGMA_THOMSON);
	if (thetae < MINT)
		return (d_hc_klein_nishina(w) * SIGMA_THOMSON);

	dmue = DMUE;
	dgammae = thetae * DGAMMAE;

	/* integrate over mu_e, gamma_e, where mu_e is the cosine of the
	   angle between k and u_e, and the angle k is assumed to lie,
	   wlog, along the z axis */
	cross = 0.;
	for (mue = -1. + 0.5 * dmue; mue < 1.; mue += dmue)
		for (gammae = 1. + 0.5 * dgammae;
		     gammae < 1. + MAXGAMMA * thetae; gammae += dgammae) {

			f = 0.5 * d_dNdgammae(thetae, gammae);

			cross +=
			    dmue * dgammae * d_boostcross(w, mue,
							gammae) * f;

			if (isnan(cross)) {
				printf("%g %g %g %g %g %g\n", w,
					thetae, mue, gammae,
					d_dNdgammae(thetae, gammae),
					d_boostcross(w, mue, gammae));
			}
		}


	return (cross * SIGMA_THOMSON);
}

__device__
double total_compton_cross_lkup(double w, double thetae)
{
	int i, j;
	double lw, lT, di, dj, lcross;
	// __device__ double total_compton_cross_num(double w, double thetae);
	// __device__ double d_hc_klein_nishina(double we);

	/* cold/low-energy: just use thomson cross section */
	if (w * thetae < 1.e-6)
		return (SIGMA_THOMSON);

	/* cold, but possible high energy photon: use klein-nishina */
	if (thetae < MINT)
		return (d_hc_klein_nishina(w) * SIGMA_THOMSON);

	/* in-bounds for table */
	if ((w > MINW && w < MAXW) && (thetae > MINT && thetae < MAXT)) {

		lw = log10(w);
		lT = log10(thetae);
		i = (int) ((lw - lminw) / dlw);
		j = (int) ((lT - lmint) / dlTT);
		di = (lw - lminw) / dlw - i;
		dj = (lT - lmint) / dlTT - j;

		lcross =
		    (1. - di) * (1. - dj) * table[i][j] + di * (1. -
								dj) *
		    table[i + 1][j] + (1. - di) * dj * table[i][j + 1] +
		    di * dj * table[i + 1][j + 1];

		if (isnan(lcross)) {
			printf("%g %g %d %d %g %g\n", lw, lT, i, j, di, dj);
		}

		return (pow(10., lcross));
	}

	printf("out of bounds: %g %g\n", w, thetae);
	return (d_total_compton_cross_num(w, thetae));

}

