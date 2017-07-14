
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



/* Global variables and function prototypes for harm(2d) models */

#ifndef global
#define global extern
#endif

global double ****econ;
global double ****ecov;
global double ***bcon;
global double ***bcov;
global double ***ucon;
global double ***ucov;
global double ***p;
global double **ne;
global double **thetae;
global double **b;

/* HARM model internal utilities */
void init_weight_table(void);                       // defined in harm_utils.c
//void bl_coord(double *X, double *r, double *th);    // defined in harm_utils.c
void bl_coord(double *X, double *r, double *th, double *phi);    // defined in harm_utils.c
void make_zone_centered_tetrads(void);              // 
void set_units(char *munitstr);                     // defined in harm_utils.c
void init_geometry(void);                           // defined in harm_utils.c
void init_harm_data(char *fname);                   // defined in init_harm_data.c
void init_nint_table(void);                         // defined in harm_utils.c
void init_storage(void);                            // defined in harm_utils.c
double dOmega_func(double x2i, double x2f);         // defined in harm_utils.c

//void sample_zone_photon(int i, int j, double dnmax, struct of_photon *ph);    // defined in harm_utils.c
void sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph);    // defined in harm_utils.c
//double interp_scalar(double **var, int i, int j, double coeff[4]);            // defined in harm_utils.c
double interp_scalar(double **var, int i, int j, int k, double coeff[8]);            // defined in harm_utils.c
//int get_zone(int *i, int *j, double *dnamx);                                  // defined in harm_utils.c
int get_zone(int *i, int *j, int *k, double *dnamx);                                  // defined in harm_utils.c
//void Xtoij(double X[NDIM], int *i, int *j, double del[NDIM]);                 // defined in harm_utils.c
void Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);                 // defined in harm_utils.c
//void coord(int i, int j, double *X);                                          // defined in harm_utils.c
void coord(int i, int j, int k, double *X);                                          // defined in harm_utils.c
//void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM]);                    // defined in harm_model.c
void get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM]);                    // defined in harm_model.c

