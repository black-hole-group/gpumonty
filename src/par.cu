/*
 * GPUmonty - par.cu
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
#include <math.h>
#include "decs.h"
#include "par.h"
#include <string.h>
#include <errno.h>
__host__ void load_par_from_argv(int argc, char *argv[], Params *params) {

  params->seed        = -1;

  params->bias_guess  = 1.e-5;
  params->fitBias     = 0.;
  params->fitBiasNs   = 10000.;
  params->targetRatio = M_SQRT2;

  params->tp_over_te = 3.;
  params->beta_crit = 1.;
  params->trat_small = 1.;
  params->trat_large = 10.;
  params->Thetae_max = 1.e100;

  params->bremsstrahlung = 0;
  params->thermal_synch = 1;
  params->scattering   = 1;
  params->kappa_synch = 0;
  params->powerlaw_synch = 0;

  strcpy(params->spectrum, "spectrum.spec");
  params->MBH_par = 4.1e6;   // MBH for Sgr A* is updated by Gravity Collaboration 2018,615,L15

  // Load parameters
  for (int i=0; i<argc-1; ++i) {
    if ( strcmp(argv[i], "-par") == 0 ) {
      load_par(argv[i+1], params);
    }
  }
}

// __host__ void try_set_radiation_parameter(const char *line)
// {
//   read_param(line, "variable_kappa_min", &variable_kappa_min, TYPE_DBL);
//   read_param(line, "variable_kappa_max", &variable_kappa_max, TYPE_DBL);

//   read_param(line, "powerlaw_gamma_cut", &powerlaw_gamma_cut, TYPE_DBL);
//   read_param(line, "powerlaw_gamma_min", &powerlaw_gamma_min, TYPE_DBL);
//   read_param(line, "powerlaw_gamma_max", &powerlaw_gamma_max, TYPE_DBL);
//   read_param(line, "powerlaw_p", &powerlaw_p, TYPE_DBL);
// }
// sets default values for elements of params (if desired) and loads
// from par file 'fname'


// Color Definitions
#define RESET   "\033[0m"
#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define GRAY    "\033[90m"

__host__ void CheckConfigErrors(Params *params) {
    // Don't use fitbias if you're not doing scattering
    if (params->fitBias && !params->scattering) {
        fprintf(stderr, RED "! ERROR: 'fit_bias' cannot be enabled without 'scattering'! Please update your .par file.\n" RESET);
        exit(EXIT_FAILURE);
    }

    //It's either thermal synchortron or kappa or powerlaw, but not more than one.
    if(params->thermal_synch + params->kappa_synch + params->powerlaw_synch > 1){
        fprintf(stderr, RED "! ERROR: More than one synchrotron emission mechanism enabled! Please update your .par file to enable only one of 'synchrotron', 'kappa_synch', or 'powerlaw_synch'.\n" RESET);
        exit(EXIT_FAILURE);
    }
    // You gotta have at least one emission active
    if(params->thermal_synch + params->kappa_synch + params->powerlaw_synch + params->bremsstrahlung == 0){
        fprintf(stderr, RED "! ERROR: No emission mechanism enabled! Please update your .par file to enable at least one of 'synchrotron', 'kappa_synch', 'powerlaw_synch', or 'bremsstrahlung'.\n" RESET);
        exit(EXIT_FAILURE);
    }

}
__host__ void load_par (const char *fname, Params *params) {
    char line[256];
    FILE *fp = fopen(fname, "r");

    // Tracker for all variables - prefixed with 'found_' to avoid macro issues
    struct {
        int seed, Ns, MBH_par, M_unit, dump, spectrum, bias_guess, fit_bias, fit_bias_ns, ratio, scattering, bremsstrahlung, thermal_synch;
        int tp_te, beta, trat_s, trat_l, theta_m;
        //Nonthermal emissions
        int kappa_synch, powerlaw_synch;
    } f = {0}; 

    if (fp == NULL) {
        printf("\033[1;31m! unable to load parameter file '%s'.\033[0m\n", fname);
        exit(-1);
    }

    while (fgets(line, 255, fp) != NULL) {
        if (line[0] == '#' || line[0] == '\n') continue; 

        // We check if the keyword is in the line, then call the void read_param
        if (strstr(line, "seed")) { read_param(line, "seed", &(params->seed), 2); f.seed = 1; }
        if (strstr(line, "Ns"))   { read_param(line, "Ns", &(params->Ns), 2); f.Ns = 1; }
        if (strstr(line, "MBH"))  { read_param(line, "MBH", &(params->MBH_par), 2); f.MBH_par = 1; }
        if (strstr(line, "M_unit")){ read_param(line, "M_unit", &(params->M_unit), 2); f.M_unit = 1; }
        
        if (strstr(line, "dump"))     { read_param(line, "dump", (void *)(params->dump), 3); f.dump = 1; }
        if (strstr(line, "spectrum")) { read_param(line, "spectrum", (void *)(params->spectrum), 3); f.spectrum = 1; }

        if (strstr(line, "scattering")) { read_param(line, "scattering", &(params->scattering), 1); f.scattering = 1; }
        if (strstr(line, "bremsstrahlung")) { read_param(line, "bremsstrahlung", &(params->bremsstrahlung), 1); f.bremsstrahlung = 1; }
        if (strstr(line, "thermal_synch")) { read_param(line, "thermal_synch", &(params->thermal_synch), 1); f.thermal_synch = 1; }

        if (strstr(line, "bias_guess"))        { read_param(line, "bias_guess", &(params->bias_guess), 2); f.bias_guess = 1; }
        if (strstr(line, "fit_bias"))    { read_param(line, "fit_bias", &(params->fitBias), 1); f.fit_bias = 1; }
        if (strstr(line, "fit_bias_ns")) { read_param(line, "fit_bias_ns", &(params->fitBiasNs), 2); f.fit_bias_ns = 1; }
        if (strstr(line, "ratio"))       { read_param(line, "ratio", &(params->targetRatio), 2); f.ratio = 1; }
        
        if (strstr(line, "tp_over_te")) { read_param(line, "tp_over_te", &(params->tp_over_te), 2); f.tp_te = 1; }
        if (strstr(line, "beta_crit"))  { read_param(line, "beta_crit", &(params->beta_crit), 2); f.beta = 1; }
        if (strstr(line, "trat_small")) { read_param(line, "trat_small", &(params->trat_small), 2); f.trat_s = 1; }
        if (strstr(line, "trat_large")) { read_param(line, "trat_large", &(params->trat_large), 2); f.trat_l = 1; }
        if (strstr(line, "Thetae_max")) { read_param(line, "Thetae_max", &(params->Thetae_max), 2); f.theta_m = 1; }
        if (strstr(line, "kappa_synch")) { read_param(line, "kappa_synch", &(params->kappa_synch), 1); f.kappa_synch = 1; }
        if (strstr(line, "powerlaw_synch")) { read_param(line, "powerlaw_synch", &(params->powerlaw_synch), 1); f.powerlaw_synch = 1; }
    }
    fclose(fp);
    CheckConfigErrors(params);


    // --- FINAL REPORT ---
    printf("\n\033[1m=== PARAMETER LOADING REPORT ===\033[0m\n");
    
    // Lambdas updated to handle 'const char*' for string compatibility
    auto print_status = [](const char* name, int found, double val) {
        if (found) printf("\033[1;32m[SET]     \033[0m %-15s : %-10g\n", name, val);
        else       printf("\033[1;31m[MISSING] \033[0m %-15s : %-10g (default)\n", name, val);
    };

    auto print_status_inv = [](const char* name, int found, double val) {
        if (!found) printf("\033[1;32m[Random]\033[0m %-15s : %-10g (default)\n", name, val);
        else       printf("\033[1;31m[Constant] \033[0m %-15s : %-10g\n", name, val);
    };


    auto print_status_i = [](const char* name, int found, int val) {
        if (found) printf("\033[1;32m[SET]     \033[0m %-15s : %-10d\n", name, val);
        else       printf("\033[1;31m[MISSING] \033[0m %-15s : %-10d (default)\n", name, val);
    };

    if(0){
      /*Just to shut the warning from not using it*/
      print_status_i("seed", f.seed, params->seed);
    }
    

    auto print_status_s = [](const char* name, int found, const char* val) {
        if (found) printf("\033[1;32m[SET]     \033[0m %-15s : %-10s\n", name, val);
        else       printf("\033[1;31m[MISSING] \033[0m %-15s : %-10s (default)\n", name, val);
    };

    print_status_inv("seed", f.seed, params->seed);
    print_status("Ns", f.Ns, params->Ns);
    print_status("MBH", f.MBH_par, params->MBH_par);
    print_status("M_unit", f.M_unit, params->M_unit);
    print_status_s("dump", f.dump, params->dump);
    print_status_s("spectrum", f.spectrum, params->spectrum);

    print_status_i("scattering", f.scattering, params->scattering);
    print_status_i("thermal_synch", f.thermal_synch, params->thermal_synch);
    print_status_i("bremsstrahlung", f.bremsstrahlung, params->bremsstrahlung);
    print_status_i("kappa_synch", f.kappa_synch, params->kappa_synch);
    print_status_i("powerlaw_synch", f.powerlaw_synch, params->powerlaw_synch);

    print_status("bias_guess", f.bias_guess, params->bias_guess);
    print_status_i("fit_bias", f.fit_bias, params->fitBias);
    print_status("fit_bias_ns", f.fit_bias_ns, params->fitBiasNs);
    print_status("ratio", f.ratio, params->targetRatio);

    print_status("tp_over_te", f.tp_te, params->tp_over_te);
    #ifdef IHARM
      print_status("beta_crit", f.beta, params->beta_crit);
      print_status("trat_small", f.trat_s, params->trat_small);
      print_status("trat_large", f.trat_l, params->trat_large);
      print_status("Thetae_max", f.theta_m, params->Thetae_max);
    #else
      printf("\033[1;33m[IGNORED] \033[0m %-15s : N/A (not IHARM model)\n", "beta_crit");
      printf("\033[1;33m[IGNORED] \033[0m %-15s : N/A (not IHARM model)\n", "trat_small");
      printf("\033[1;33m[IGNORED] \033[0m %-15s : N/A (not IHARM model)\n", "trat_large");
      printf("\033[1;33m[IGNORED] \033[0m %-15s : N/A (not IHARM model)\n", "Thetae_max");
    #endif
    printf("\033[1m================================\033[0m\n");
    params->loaded = 1;
}
#undef RESET
#undef RED
#undef GREEN


// loads value -> (type *)*val if line corresponds to (key,value)
__host__ void read_param (const char *line, const char *key, void *val, int type) {

  // silence valgrind warnings
  char word[256] = {0};
  char value[256] = {0};

  sscanf(line, "%s %s", word, value);

  if (strcmp(word, key) == 0) {
    switch (type) {
    case TYPE_INT:
      sscanf(value, "%d", (int *)val);
      break;
    case TYPE_DBL:
      sscanf(value, "%lf", (double *)val);
      break;
    case TYPE_STR:
      sscanf(value, "%s", (char *)val);
      break;
    default:
      fprintf(stderr, "! attempt to load unknown type '%d'.\n", type);
    }
  }

}

