
/***********************************************************************************
    Copyright 2013 Joshua C. Dolence, Charles F. Gammie, Monika Mo\'scibrodzka,
                   and Po Kin Leung

                        GRMONTY  version 1.0   (released February 1, 2013)

    This file is part of GRMONTY.  GRMONTY v1.0 is a program that calculates the
    emergent spectrum from a model using a Monte Carlo technique.

    This version of GRMONTY is configured to use input files from the HARM code
    available on the same site.   It assumes that the source is a plasma near a
    black hole described by Kerr-Schild coordinates that radiates via thermal 
    synchrotron and inverse compton scattering.
    
    You are morally obligated to cite the following paper in any
    scientific literature that results from use of any part of GRMONTY:

    Dolence, J.C., Gammie, C.F., Mo\'scibrodzka, M., \& Leung, P.-K. 2009,
        Astrophysical Journal Supplement, 184, 387

    Further, we strongly encourage you to obtain the latest version of 
    GRMONTY directly from our distribution website:
    http://rainman.astro.illinois.edu/codelib/

    GRMONTY is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    GRMONTY is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GRMONTY; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/

#include "decs.h"
#pragma omp threadprivate(r)

/*

Routines for treating Compton scattering via Monte Carlo.

Uses a Gnu Scientific Library (GSL) random number generator.
The choice of generator can be changed in init_monty_rand;
now set to Mersenne twister.

Sampling procedures for electron distribution is based on
Canfield, Howard, and Liang, 1987, ApJ 323, 565.

*/

void init_monty_rand(int seed)
{
	r = gsl_rng_alloc(gsl_rng_mt19937);	/* use Mersenne twister */
	gsl_rng_set(r, seed);
}

/* return pseudo-random value between 0 and 1 */
double monty_rand()
{
	return (gsl_rng_uniform(r));
}



//void sample_scattered_photon(double k[4], double p[4], double kp[4])

//void boost(double v[4], double u[4], double vp[4])

//double sample_thomson()

//double sample_klein_nishina(double k0)

//double klein_nishina(double a, double ap)

//void sample_electron_distr_p(double k[4], double p[4], double Thetae)

//void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e)

//double sample_y_distr(double Thetae)

//double sample_mu_distr(double beta_e)
