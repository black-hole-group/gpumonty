/* Global variables and function prototypes for harm(2d) models */

#ifndef _HARM_MODEL_H
#define _HARM_MODEL_H

// Unused variables (never set/read)
// double ****econ;
// double ****ecov;
// double ***bcon;
// double ***bcov;
// double ***ucon;
// double ***ucov;
// double **ne;
// double **thetae;
// double **b;

// Variables transformed to defines
// double R0;
// double a;
// double hslope;

extern double *harm_p;
extern __device__ double *d_harm_p;

#ifdef __CUDA_ARCH__
#define HARM_P(x,y,z) (d_harm_p[(x)*d_N1*d_N2 + (y)*d_N2 + (z)])
#else
#define HARM_P(x,y,z) (harm_p[(x)*N1*N2 + (y)*N2 + (z)])
#endif


#define R0 0.0
#define a  0.9375
#define hslope 0.3


// Inexistent function
// void make_zone_centered_tetrads(void);

// init_iharm2dv3_data's functions
void init_harm_data(char *fname);

// harm_model's functions
void get_fluid_zone(int i, int j, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM]);

// harm_utils's functions
void init_storage(void);
double dOmega_func(double x2i, double x2f);
void init_weight_table(unsigned long long Ns);
void init_nint_table(void);
void coord(int i, int j, double *X);
void set_units(char *munitstr);
void sample_zone_photon(int i, int j, double dnmax, struct of_photon *ph,
						int first_zone_photon);
void init_geometry(void);
__host__ __device__
double interp_p_scalar(int x, int y, int z, double coeff[4]);
__host__ __device__
void Xtoij(double X[NDIM], int *i, int *j, double del[NDIM]);
__host__ __device__
void bl_coord(double *X, double *r, double *th);

#endif
