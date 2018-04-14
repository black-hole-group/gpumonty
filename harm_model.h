/* Global variables and function prototypes for harm(2d) models */
#ifndef HARM_MODEL_H 
#define HARM_MODEL_H

#ifndef global
#define global extern
#endif

global float ****econ;
global float ****ecov;
global float ***bcon;
global float ***bcov;
global float ***ucon;
global float ***ucov;
global float ***p;
global float **ne;
global float **thetae;
global float **b;

// pointers for device arrays
global float *d_p; 


/* HARM model internal utilities */
void init_weight_table(void);
void bl_coord(double *X, double *r, double *th);
void make_zone_centered_tetrads(void);
void set_units(char *munitstr);
void init_geometry(void);
void init_harm_data(char *fname);
void init_nint_table(void);
void init_storage(void);
double dOmega_func(double x2i, double x2f);

void sample_zone_photon(int i, int j, double dnmax, struct of_photon *ph);
double interp_scalar(double **var, int i, int j, double coeff[4]);
int get_zone(int *i, int *j, double *dnamx);
void Xtoij(double X[NDIM], int *i, int *j, double del[NDIM]);
void coord(int i, int j, double *X);
void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM]);

#endif