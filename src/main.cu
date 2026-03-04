/*
 * GPUmonty - main.cu
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

#include "defs.h"
#include "kernels.h"
#include "model.h"
#include "weights.h"
#include "metrics.h"
#include "jnu_mixed.h"
#include "hotcross.h"
#include "utils.h"
#include "par.h"

__host__ void init_model(char *args[]);
__host__ void init_geometry();
int main(int argc, char *argv[])
{
	fprintf(stderr, "GPUmonty. githash: %s\n", xstr(VERSION));
	time_t starttime = time(NULL);
	time_t time_seeding;
	
	// load parameters from command line
  	load_par_from_argv(argc, argv, &params);

	/* initialize model data, auxiliary variables */
	init_model(argv);

	/** main loop **/
	N_superph_recorded = 0;
	N_scatt = 0;

	fprintf(stderr, "Entering main loop...\n");
	fflush(stderr);

	if(params.seed == -1){
		time_seeding =  time(NULL);
	}else{
		time_seeding = 0;
	}
    mainFlowControl(time_seeding, p);
    printf("Time spent running the full code: %.4f seconds\n", ((double)(time(NULL) - starttime)));
	printf("Ntot = %d/Number of Blocks = %d /Block Size = %d\n", (int) params.Ns, N_BLOCKS, N_THREADS);

	return (0);

}

__host__ void set_units(Params params)
{
	/* Sets the global units based on the parameters passed in */
	L_unit = GNEWT * params.MBH_par * MSUN / (CL * CL);
	M_unit = params.M_unit;
	Rho_unit = M_unit / (L_unit * L_unit * L_unit);
	/*Derived units*/
	T_unit = L_unit/CL; /*UNIT of time*/
	U_unit = Rho_unit * CL * CL; /*UNITy of energy density*/
	B_unit = CL * sqrt(4. * M_PI * Rho_unit); /*Unit of magnetig field*/
	Ne_unit = Rho_unit/(MP + ME); /*Unit of electron density*/

	/*Set global variables for device*/
	cudaMemcpyToSymbol(d_L_unit, &L_unit, sizeof(double));
	cudaMemcpyToSymbol(d_MBH, &params.MBH_par, sizeof(double));
	cudaMemcpyToSymbol(d_B_unit, &B_unit, sizeof(double));
	cudaMemcpyToSymbol(d_Ne_unit, &Ne_unit, sizeof(double));
}

__host__ void init_model(char *args[])
{
	/* This will tell the units defined in decs.h. 
	There used to be a function here for this, but it's extremely 
	unecessary as well as taking M_UNIT as an argument*/
	set_units(params);
	fprintf(stderr, "\nUNITS\n");
	fprintf(stderr, "L,T,M: %g %g %g\n", L_unit, T_unit, M_unit);
	fprintf(stderr, "rho,u,B: %g %g %g\n", Rho_unit, U_unit, B_unit);
	max_tau_scatt = (6. * L_unit) * Rho_unit * 0.4;

	fprintf(stderr, "Initial max_tau_scatt: %g\n", max_tau_scatt);


	fprintf(stderr, "getting simulation data...\n");
	init_data();
	/* initialize the metric */
	fprintf(stderr, "initializing geometry...\n");
	fflush(stderr);
	init_geometry();

	if(isinf(geom[SPATIAL_INDEX2D(0,0)].gcov[3][3])){
		fprintf(stderr, "Negative determinant of the metric at the origin. Exiting...\n");
		exit(0);
	}
	fprintf(stderr, "done.\n\n");
	fflush(stderr);

	/* make look-up table for hot cross sections */
	init_hotcross();

	/* make table for solid angle integrated emissivity and K2 */
	init_emiss_tables();

	/* make table for superphoton weights */
	init_weight_table();
	//init_weight_table_blackbody();
	/* make table for quick evaluation of ns_zone */
	init_nint_table();

}
/* set up all grid functions */
__host__ void init_geometry()
{
	int i, j, k;
	double X[NDIM];

	for (i = 0; i < N1; i++) {
		for (j = 0; j < N2; j++) {
			for (k = 0; k < N3; k++) {

				/* zone-centered */
				coord(i, j, k, X);
				gcov_func(X, geom[SPATIAL_INDEX2D(i,j)].gcov);
				geom[SPATIAL_INDEX2D(i,j)].g = gdet_func(geom[SPATIAL_INDEX2D(i,j)].gcov);
				gcon_func(X, geom[SPATIAL_INDEX2D(i,j)].gcov, geom[SPATIAL_INDEX2D(i,j)].gcon);

			}
		}
	}
}

__host__ void report_spectrum(unsigned long long N_superph_made, struct of_spectrum ***spect, const char * filename)
{
    int i, j, k;
    double dx2, dOmega, nuLnu, tau_scatt, L;
    FILE *fp;
    char filepath[256]; 

    // Construct the file path
    snprintf(filepath, sizeof(filepath), "./output/%s", filename);
    fp = fopen(filepath, "w");
    if (fp == NULL) {
        fprintf(stderr, "trouble opening spectrum file\n");
        exit(0);
    }

    /* --- NEW: Print dOmega values to file header --- */
    // We calculate dx2 here to use for the dOmega loop
    dx2 = (stopx[2] - startx[2]) / (2. * N_THBINS);
    
    fprintf(fp, "# dOmega:"); // Header tag
    for (j = 0; j < N_THBINS; j++) {
        // Calculate dOmega for this theta bin
        dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);
        fprintf(fp, " %.15e", dOmega);
    }
    fprintf(fp, "\n"); // End header line
    /* ----------------------------------------------- */

    /* output */
    max_tau_scatt = 0.;
    L = 0.;

	for (k=0; k < N_TYPEBINS; k++){
		// Running through all energy bins
		for (i = 0; i < N_EBINS; i++) {
	
			/* output log_10(photon energy/(me c^2))*/
			fprintf(fp, "%10.5g ", (i * dlE + lE0) / M_LN10);
	
			for (j = 0; j < N_THBINS; j++) {
	
				/* convert accumulated photon number in each bin 
				   to \nu L_\nu, in units of Lsun */
				// dx2 is already calculated above, but valid here too
				dOmega = 2. * dOmega_func(j * dx2, (j + 1) * dx2);
	
				nuLnu = (ME * CL * CL) * (4. * M_PI / dOmega) * (1. / dlE);
	
				nuLnu *= spect[k][j][i].dEdlE;
				nuLnu /= LSUN;
				
				tau_scatt = spect[k][j][i].tau_scatt / (spect[k][j][i].dNdlE + SMALL);
	
				fprintf(fp,
					"%10.5g %10.5g %10.5g %10.5g %10.5g %10.5g ",
					nuLnu,
					spect[k][j][i].tau_abs / (spect[k][j][i].dNdlE + SMALL), 
					tau_scatt,
					spect[k][j][i].X1iav / (spect[k][j][i].dNdlE + SMALL),
					sqrt(fabs(spect[k][j][i].X2isq / (spect[k][j][i].dNdlE + SMALL))),
					sqrt(fabs(spect[k][j][i].X3fsq / (spect[k][j][i].dNdlE + SMALL)))
					);
	
				if (tau_scatt > max_tau_scatt){
					max_tau_scatt = tau_scatt;
				}
				L += nuLnu * dOmega * dlE/(4.*M_PI);
			}
			fprintf(fp, "\n");
		}
	}

    // Standard logging to stderr
    printf("\n\033[1m==================== OUTPUT ====================\033[0m\n");
    fprintf(stderr,
        "luminosity %g erg/s\ndMact %g\nefficiency %g\nL/Ladv %g\nmax_tau_scatt %g\n",
        L * LSUN, dMact * M_unit / T_unit / (MSUN / YEAR),
        L * LSUN / (dMact * M_unit * CL * CL / T_unit),
        L * LSUN / (Ladv * M_unit * CL * CL / T_unit),
        max_tau_scatt);
	
	printf("\n");
    // Cleaning up the stderr loop since we now write to file
    fprintf(stderr, "Number of superphotons made: %llu\n", N_superph_made);
    fprintf(stderr, "Number of superphotons scattered: %llu\n", N_scatt);
    fprintf(stderr, "Number of superphotons recorded: %llu\n", N_superph_recorded);
    fprintf(stderr, "Data saved in %s\n", filepath);
    fclose(fp);
	printf("\n\033[1m================================================\033[0m\n");

}
