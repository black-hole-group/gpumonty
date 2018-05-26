/*
  Functions that were previously in harm_utils.c and need to
  be executed on the device. 
*/

/* 
 * New version for the row-major 1D array for GPU code.
 * - n=variable-selector index
 * - i=x1 index
 * - j=x2 index
 */
__device__
double interp_scalar(double *var, int n, int i, int j, double coeff[4])
{

	double interp;

	interp =
	    var[n*N1*N2+i*N2+j] * coeff[0] +
	    var[n*N1*N2+i*N2+j+1] * coeff[1] +
	    var[n*N1*N2+(i+1)*N2+j] * coeff[2] + 
	    var[n*N1*N2+(i+1)*N2+j+1] * coeff[3];

	return interp;
}

__device__
void Xtoij(double X[NDIM], int *i, int *j, double del[NDIM])
{

	*i = (int) ((X[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
	*j = (int) ((X[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;

	if (*i < 0) {
		*i = 0;
		del[1] = 0.;
	} else if (*i > N1 - 2) {
		*i = N1 - 2;
		del[1] = 1.;
	} else {
		del[1] = (X[1] - ((*i + 0.5) * dx[1] + startx[1])) / dx[1];
	}

	if (*j < 0) {
		*j = 0;
		del[2] = 0.;
	} else if (*j > N2 - 2) {
		*j = N2 - 2;
		del[2] = 1.;
	} else {
		del[2] = (X[2] - ((*j + 0.5) * dx[2] + startx[2])) / dx[2];
	}

	return;
}

/* 
  return boyer-lindquist coordinate of point 
  also defined on host
*/
__device__
void d_bl_coord(double *X, double *r, double *th)
{

	*r = exp(X[1]) + R0;
	*th = M_PI * X[2] + ((1. - hslope) / 2.) * sin(2. * M_PI * X[2]);

	return;
}