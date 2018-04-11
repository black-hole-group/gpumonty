/* Global variables and function prototypes for HARMPI models */

#ifndef global
#define global extern
#endif

global double ****econ;
global double ****ecov;
global double ***bcon;
global double ***bcov;
global double ***ucon;
global double ***ucov;
global double ****p;
global double ***ne;
global double ***thetae;
global double ***b;

/* HARM model internal utilities */
void init_weight_table(void);                       // defined in harm_utils.c
void bl_coord(double *X, double *r, double *th);    // defined in harm_utils.c
void make_zone_centered_tetrads(void);              // 
void set_units(char *munitstr);                     // defined in harm_utils.c
void init_geometry(void);                           // defined in harm_utils.c
void init_harm_data(char *fname);                   // defined in init_harm_data.c
void init_nint_table(void);                         // defined in harm_utils.c
void init_storage(void);                            // defined in harm_utils.c
double dOmega_func(double x2i, double x2f);         // defined in harm_utils.c

void sample_zone_photon(int i, int j, int k, double dnmax, struct of_photon *ph);    // defined in harm_utils.c
double interp_scalar(double ***var, int i, int j, int k, double coeff[8]);            // defined in harm_utils.c
int get_zone(int *i, int *j, int *k, double *dnamx);                                  // defined in harm_utils.c
void Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]);                 // defined in harm_utils.c
void coord(int i, int j, int k, double *X);                                          // defined in harm_utils.c
void get_fluid_zone(int i, int j, int k, double *Ne, double *Thetae, double *B,
		    double Ucon[NDIM], double Bcon[NDIM]);                    // defined in harm_model.c
