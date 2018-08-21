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

