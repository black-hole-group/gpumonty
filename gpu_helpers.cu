// #include "decs.h"
// #include <math.h>
//
// unsigned short int malloc_counter = 0;
// __global__ void get_connection_kernel(double X[4], double lconn[64], double a, double hslope);
//
// __host__ void has_error_happend(cudaError_t error);
//
// __host__
// void cuda_get_connection(double X[4], double lconn[4][4][4]){
// 	/*malloc and set cuda space*/
// 	size_t Xsize = 4 * sizeof(double);
// 	double *d_X = NULL;
// 	has_error_happend(cudaMalloc((void **)&d_X, Xsize));
// 	has_error_happend(cudaMemcpy(d_X, X, Xsize, cudaMemcpyHostToDevice));
//
// 	size_t Lsize = 4 * 4 * 4 * sizeof(double);
// 	double *d_lconn = NULL;
// 	has_error_happend(cudaMalloc((void **)&d_lconn, Lsize));
//
// 	/*3d to 1d array*/
// 	// for (size_t i = 0; i < 4; i++)
// 	// 	for (size_t j = 0; j < 4; j++)
// 	// 		for (size_t k = 0; k < 4; k++)
// 	// 			tmp[i*16 + j*4 + k] = lconn[i][j][k];
// 	/*3d to 1d array*/
//
// 	has_error_happend(cudaMemcpy(d_lconn, tmp, Lsize, cudaMemcpyHostToDevice));
//
// 	double d_a = a;
// 	double d_hslope = hslope;
// 	/*malloc and set cuda space*/
//
// 	get_connection_kernel<<<2, 2>>>(d_X, d_lconn, d_a, d_hslope);
//   has_error_happend(cudaGetLastError());
//
//   has_error_happend(cudaMemcpy(X, d_X, Xsize, cudaMemcpyDeviceToHost));
// 	has_error_happend(cudaMemcpy(tmp, d_lconn, Lsize, cudaMemcpyDeviceToHost));
//
//   /* 1d to 3d*/
// 	double *tmp = (double *)malloc(Lsize);
// 	for (size_t i = 0; i < 4; i++)
// 		for (size_t j = 0; j < 4; j++)
// 			for (size_t k = 0; k < 4; k++)
// 				lconn[i][j][k] = tmp[i*16 + j*4 + k];
// 	/* 1d to 3d*/
//
// 	has_error_happend(cudaFree(d_X));
//   has_error_happend(cudaFree(d_lconn));
// 	free(tmp);
// }
//
// __global__
// void get_connection_kernel(double X[4], double *lconn, double d_a, double d_hslope) {
// 	double r1, r2, r3, r4, sx, cx;
// 	double th, dthdx2, dthdx22, d2thdx22, sth, cth, sth2, cth2, sth4,
// 	cth4, s2th, c2th;
// 	double a2, a3, a4, rho2, irho2, rho22, irho22, rho23, irho23,
// 	irho23_dthdx2;
// 	double fac1, fac1_rho23, fac2, fac3, a2cth2, a2sth2, r1sth2,
// 	a4cth4;
//
// 	r1 = exp(X[1]);
// 	r2 = r1 * r1;
// 	r3 = r2 * r1;
// 	r4 = r3 * r1;
//
// 	sincos(2. * M_PI * X[2], &sx, &cx);
// 	th       = M_PI   * X[2]   + 0.5  * (1      - d_hslope) * sx;
// 	d2thdx22 = -2.    * M_PI   * M_PI * (1      - d_hslope) * sx;
// 	dthdx2   = M_PI   * (1.    + (1   - d_hslope) * cx);
// 	dthdx22  = dthdx2 * dthdx2;
//
// 	sincos(th, &sth, &cth);
// 	sth2   = sth  * sth;
// 	r1sth2 = r1   * sth2;
// 	sth4   = sth2 * sth2;
//
// 	cth2   = cth  * cth;
// 	cth4   = cth2 * cth2;
// 	c2th   = 2    * cth2 - 1.;
// 	s2th   = 2.   * sth  * cth;
//
// 	a2     = d_a  * d_a;
// 	a2sth2 = a2 * sth2;
// 	a2cth2 = a2 * cth2;
// 	a3     = a2 * d_a;
// 	a4     = a3 * d_a;
// 	a4cth4 = a4 * cth4;
//
// 	rho2          = r2     + a2cth2;
// 	rho22         = rho2   * rho2;
// 	rho23         = rho22  * rho2;
// 	irho2         = 1.     / rho2;
// 	irho22        = irho2  * irho2;
// 	irho23        = irho22 * irho2;
// 	irho23_dthdx2 = irho23 / dthdx2;
//
// 	fac1       = r2   - a2cth2;
// 	fac1_rho23 = fac1 * irho23;
// 	fac2       = a2   + 2     * r2   + a2 * c2th;
// 	fac3       = a2   + r1    * (-2. + r1);
//
// 	lconn[0*16 + 0*4 + 0] = 2.  * r1  * fac1_rho23;
// 	lconn[0*16 + 0*4 + 1] = r1  * (2. * r1        + rho2)     * fac1_rho23;
// 	lconn[0*16 + 0*4 + 2] = -a2 * r1  * s2th      * dthdx2    * irho22;
// 	lconn[0*16 + 0*4 + 3] = -2. * d_a   * r1sth2    * fac1_rho23;
//
// 	lconn[0*16 + 1*4 + 1] = 2.  * r2 * (r4  + r1     * fac1  - a4cth4) * irho23;
// 	lconn[0*16 + 1*4 + 2] = -a2 * r2 * s2th * dthdx2 * irho22;
// 	lconn[0*16 + 1*4 + 3] = d_a   * r1 * (-r1 * (r3    + 2     * fac1)   + a4cth4) * sth2 * irho23;
//
// 	lconn[0*16 + 2*4 + 2] = -2. * r2     * dthdx22 * irho2;
// 	lconn[0*16 + 2*4 + 3] = a3  * r1sth2 * s2th    * dthdx2 * irho22;
//
// 	lconn[0*16 + 3*4 + 3] = 2.  * r1sth2 * (-r1    * rho22  + a2sth2 * fac1) * irho23;
//
// 	lconn[1*16 + 0*4 + 0] = fac3 * fac1 / (r1 * rho23);
// 	lconn[1*16 + 0*4 + 1] = fac1 * (-2. * r1  + a2sth2) * irho23;
// 	lconn[1*16 + 0*4 + 2] = 0.;
// 	lconn[1*16 + 0*4 + 3] = -d_a   * sth2 * fac3 * fac1   / (r1   * rho23);
//
// 	lconn[1*16 + 1*4 + 1] = (r4 * (-2.   + r1)  * (1. + r1)  + a2 * (a2 * r1 * (1. + 3. * r1) * cth4 + a4cth4 * cth2 + r3 * sth2 + r1 * cth2 * (2. * r1 + 3. * r3 - a2sth2))) * irho23;
// 	lconn[1*16 + 1*4 + 2] = -a2 * dthdx2 * s2th / fac2;
// 	lconn[1*16 + 1*4 + 3] = d_a   * sth2   * (a4  * r1  * cth4 + r2 * (2  * r1 + r3  - a2sth2) +	a2cth2 * (2. * r1 * (-1. + r2) + a2sth2)) * irho23;
//
// 	lconn[1*16 + 2*4 + 2] = -fac3 * dthdx22 * irho2;
// 	lconn[1*16 + 2*4 + 3] = 0.;
//
// 	lconn[1*16 + 3*4 + 3] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);
//
// 	lconn[2*16 + 0*4 + 0] = -a2 * r1 * s2th * irho23_dthdx2;
// 	lconn[2*16 + 0*4 + 1] = r1  * lconn[2*16 + 0*4 + 0];
// 	lconn[2*16 + 0*4 + 2] = 0.;
// 	lconn[2*16 + 0*4 + 3] = d_a   * r1 * (a2  + r2) * s2th * irho23_dthdx2;
//
// 	lconn[2*16 + 1*4 + 1] = r2 * lconn[2*16 + 0*4 + 0];
// 	lconn[2*16 + 1*4 + 2] = r2 * irho2;
// 	lconn[2*16 + 1*4 + 3] = (d_a * r1 * cth * sth * (r3 * (2. + r1) + a2 * (2. * r1 * (1. + r1) * cth2 + a2 * cth4 + 2 * r1sth2))) * irho23_dthdx2;
//
// 	lconn[2*16 + 2*4 + 2] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;
// 	lconn[2*16 + 2*4 + 3] = 0.;
//
// 	lconn[2*16 + 3*4 + 3] = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4. + r1) + a2cth2) + 2. * r1 * a4 * sth4) * irho23_dthdx2;
//
// 	lconn[3*16 + 0*4 + 0] = d_a       * fac1_rho23;
// 	lconn[3*16 + 0*4 + 1] = r1      * lconn[3*16 + 0*4 + 0];
// 	lconn[3*16 + 0*4 + 2] = -2.     * d_a             * r1 * cth * dthdx2 / (sth * rho22);
// 	lconn[3*16 + 0*4 + 3] = -a2sth2 * fac1_rho23;
//
// 	lconn[3*16 + 1*4 + 1] = d_a  * r2  * fac1_rho23;
// 	lconn[3*16 + 1*4 + 2] = -2 * d_a   * r1        * (a2    + 2     * r1    * (2. + r1) + a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
// 	lconn[3*16 + 1*4 + 3] = r1 * (r1 * rho22     - a2sth2 * fac1) * irho23;
//
// 	lconn[3*16 + 2*4 + 2] = -d_a * r1 * dthdx22 * irho2;
// 	lconn[3*16 + 2*4 + 3] = dthdx2 * (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22;
//
// 	lconn[3*16 + 3*4 + 3] = (-d_a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
// }
//
// __host__
// void has_error_happend(cudaError_t error) {
//   malloc_counter++;
//   if (error != cudaSuccess)
//   {
//     fprintf(
//       stderr,
//       "something bad happend -%d-(error code %s)!\n",
//       malloc_counter,
//       cudaGetErrorString(error)
//     );
//     exit(EXIT_FAILURE);
//   }
// }
