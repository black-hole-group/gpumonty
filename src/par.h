/*
Declaration of the par.cu functions
*/
#define TYPE_INT (1)
#define TYPE_DBL (2)
#define TYPE_STR (3)

#ifndef PAR_H
#define PAR_H

__host__ void load_par_from_argv(int argc, char *argv[], Params *params);
__host__ void load_par(const char *, Params *);
// __host__ void try_set_radiation_parameter(const char *line);

__host__ void read_param(const char *, const char *, void *, int);
#endif