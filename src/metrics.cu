#include "decs.h"
/* 
	In this file, given gcov_func in the model, we can calculate the gcon, gdet and also the connection terms.
*/

gsl_matrix *gsl_gcov, *gsl_gcon;
gsl_permutation *perm;
#pragma omp threadprivate (gsl_gcov, gsl_gcon, perm)

/* assumes gcov has been set first; returns determinant */
double gdet_func(double gcov[][NDIM])
{
  #if(SPHERE_TEST)
    return sqrt(-gcov[0][0] * gcov[1][1] * gcov[2][2] * gcov[3][3]);
  #else
  double d;
	int k, l, signum;
	if (gsl_gcov == NULL) {
		gsl_gcov = gsl_matrix_alloc(NDIM, NDIM);
		gsl_gcon = gsl_matrix_alloc(NDIM, NDIM);
		perm = gsl_permutation_alloc(NDIM);
	}

	DLOOP gsl_matrix_set(gsl_gcov, k, l, gcov[k][l]);

	gsl_linalg_LU_decomp(gsl_gcov, perm, &signum);

	d = gsl_linalg_LU_det(gsl_gcov, signum);

	return (sqrt(fabs(d)));
  #endif
}

__host__  __device__ int LU_decompose( double A[][NDIM], int permute[] )
{
  double row_norm[NDIM];

  double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */

  double  absmax, maxtemp;

  int i, j, k, max_row;
  int n = NDIM;


  max_row = 0;

  /* Find the maximum elements per row so that we can pretend later
     we have unit-normalized each equation: */

  for( i = 0; i < n; i++ ) { 
    absmax = 0.;
    
    for( j = 0; j < n ; j++ ) { 
      
      maxtemp = fabs( A[i][j] ); 

      if( maxtemp > absmax ) { 
	absmax = maxtemp; 
      }
    }

    /* Make sure that there is at least one non-zero element in this row: */
    if( absmax == 0. ) { 
     //printf( "LU_decompose(): row-wise singular matrix!\n");
      return(1);
    }

    row_norm[i] = 1. / absmax ;   /* Set the row's normalization factor. */
  }


  /* The following the calculates the matrix composed of the sum 
     of the lower (L) tridagonal matrix and the upper (U) tridagonal
     matrix that, when multiplied, form the original maxtrix.  
     This is what we call the LU decomposition of the maxtrix. 
     It does this by a recursive procedure, starting from the 
     upper-left, proceding down the column, and then to the next
     column to the right.  The decomposition can be done in place 
     since element {i,j} require only those elements with {<=i,<=j} 
     which have already been computed.
     See pg. 43-46 of "Num. Rec." for a more thorough description. 
  */

  /* For each of the columns, starting from the left ... */
  for( j = 0; j < n; j++ ) {

    /* For each of the rows starting from the top.... */

    /* Calculate the Upper part of the matrix:  i < j :   */
    for( i = 0; i < j; i++ ) {
      for( k = 0; k < i; k++ ) { 
	A[i][j] -= A[i][k] * A[k][j];
      }
    }

    absmax = 0.0;

    /* Calculate the Lower part of the matrix:  i <= j :   */

    for( i = j; i < n; i++ ) {

      for (k = 0; k < j; k++) { 
	A[i][j] -= A[i][k] * A[k][j];
      }

      /* Find the maximum element in the column given the implicit 
	 unit-normalization (represented by row_norm[i]) of each row: 
      */
      maxtemp = fabs(A[i][j]) * row_norm[i] ;

      if( maxtemp >= absmax ) {
	absmax = maxtemp;
	max_row = i;
      }

    }

    /* Swap the row with the largest element (of column j) with row_j.  absmax
       This is the partial pivoting procedure that ensures we don't divide
       by 0 (or a small number) when we solve the linear system.  
       Also, since the procedure starts from left-right/top-bottom, 
       the pivot values are chosen from a pool involving all the elements 
       of column_j  in rows beneath row_j.  This ensures that 
       a row  is not permuted twice, which would mess things up. 
    */
    if( max_row != j ) {

      /* Don't swap if it will send a 0 to the last diagonal position. 
	 Note that the last column cannot pivot with any other row, 
	 so this is the last chance to ensure that the last two 
	 columns have non-zero diagonal elements.
       */

      if( (j == (n-2)) && (A[j][j+1] == 0.) ) {
	max_row = j;
      }
      else { 
	for( k = 0; k < n; k++ ) { 

	  maxtemp       = A[   j   ][k] ; 
	  A[   j   ][k] = A[max_row][k] ;
	  A[max_row][k] = maxtemp; 

	}

	/* Don't forget to swap the normalization factors, too... 
	   but we don't need the jth element any longer since we 
	   only look at rows beneath j from here on out. 
	*/
	row_norm[max_row] = row_norm[j] ; 
      }
    }

    /* Set the permutation record s.t. the j^th element equals the 
       index of the row swapped with the j^th row.  Note that since 
       this is being done in successive columns, the permutation
       vector records the successive permutations and therefore
       index of permute[] also indexes the chronology of the 
       permutations.  E.g. permute[2] = {2,1} is an identity 
       permutation, which cannot happen here though. 
    */

    permute[j] = max_row;

    if( A[j][j] == 0. ) { 
      A[j][j] = absmin;
    }


  /* Normalize the columns of the Lower tridiagonal part by their respective 
     diagonal element.  This is not done in the Upper part because the 
     Lower part's diagonal elements were set to 1, which can be done w/o 
     any loss of generality.
  */
    if( j != (n-1) ) { 
      maxtemp = 1. / A[j][j]  ;
      
      for( i = (j+1) ; i < n; i++ ) {
	A[i][j] *= maxtemp;
      }
    }

  }

  return(0);

  /* End of LU_decompose() */

}

__host__ __device__ void LU_substitution( double A[][NDIM], double B[], int permute[] )
{
  int i, j ;
  int n = NDIM;
  double tmpvar;

  
  /* Perform the forward substitution using the LU matrix. 
   */
  for(i = 0; i < n; i++) {

    /* Before doing the substitution, we must first permute the 
       B vector to match the permutation of the LU matrix. 
       Since only the rows above the currrent one matter for 
       this row, we can permute one at a time. 
    */
    tmpvar        = B[permute[i]];
    B[permute[i]] = B[    i     ];
    for( j = (i-1); j >= 0 ; j-- ) { 
      tmpvar -=  A[i][j] * B[j];
    }
    B[i] = tmpvar; 
  }
	   

  /* Perform the backward substitution using the LU matrix. 
   */
  for( i = (n-1); i >= 0; i-- ) { 
    for( j = (i+1); j < n ; j++ ) { 
      B[i] -=  A[i][j] * B[j];
    }
    B[i] /= A[i][i] ; 
  }

  /* End of LU_substitution() */

}

__host__ __device__ int invert_matrix( double Am[][NDIM], double Aminv[][NDIM] )  
{ 

  int i,j;
  int n = NDIM;
  int permute[NDIM]; 
  double dxm[NDIM], Amtmp[NDIM][NDIM];

  for (int j = 0; j < NDIM; j++) {
      for (i = 0; i < NDIM; i++) {
          Amtmp[j][i] = Am[j][i];
      }
  }

  // Get the LU matrix:
  if( LU_decompose( Amtmp,  permute ) != 0  ) { 
    printf("invert_matrix(): singular matrix encountered! \n");
    printf("This is probably due to a nan value somewhere rather than determinant = 0. Investigate!\n");
	return(1);
  }

  for( i = 0; i < n; i++ ) { 
    for( j = 0 ; j < n ; j++ ) { dxm[j] = 0. ; }
    dxm[i] = 1.; 
    
    /* Solve the linear system for the i^th column of the inverse matrix: :  */
    LU_substitution( Amtmp,  dxm, permute );

    for( j = 0 ; j < n ; j++ ) {  Aminv[j][i] = dxm[j]; }

  }

  return(0);
}


// __host__ __device__ void gcon_func(double X[4], double gcov[][NDIM], double gcon[][NDIM])
// {
//   invert_matrix( gcov, gcon );
// }

__host__ __device__ void gcon_func(double X[4], double gcov[][NDIM], double gcon[][NDIM])
{
	int k, l;
	#if(SPHERE_TEST)
	DLOOP gcon[k][l] = 0.;
	/*Flat space in spherical coordinates for the test*/							
    gcon[0][0] = -1.;
    gcon[1][1] = 1.;
    gcon[2][2] = 1./gcov[2][2];
    gcov[3][3] = 1/gcov[3][3];
	#else
		double irho2;
		double r, th;
		double hfac;


		DLOOP gcon[k][l] = 0.;
		bl_coord(X, &r, &th);

	#ifdef __CUDA_ARCH__
	double bhspin = d_a;
	double thetaslope = d_hslope;
	#else
	double bhspin = a;
	double thetaslope = hslope;
	#endif

		sth = sin(th);
		cth = cos(th);

		sth = fabs(sth) + SMALL;

		irho2 = 1. / (r * r + bhspin *bhspin * cth * cth);

		//transformation for Kerr-Schild -> modified Kerr-Schild 
		hfac = M_PI + (1. - thetaslope) * M_PI * cos(2. * M_PI * X[2]);

		#if(HAMR)
		hfac = M_PI;
		#endif

		gcon[0][0] = -1. - 2. * r * irho2;
		gcon[0][1] = 2. * irho2;

		gcon[1][0] = gcon[0][1];
		gcon[1][1] = irho2 * (r * (r - 2.) + bhspin * bhspin) / (r * r);
		gcon[1][3] = bhspin * irho2 / r;

		gcon[2][2] = irho2 / (hfac * hfac);

		gcon[3][1] = gcon[1][3];
		gcon[3][3] = irho2 / (sth * sth);
	#endif
}


// #define DEL (1.e-7)
// __device__ void GPU_get_connection(double X[NDIM], double conn[NDIM][NDIM][NDIM])
// {
// 	/* required by broken math.h */
// 	//void sincos(double th, double *sth, double *cth);
// 	#if(SPHERE_TEST)
//   	double r1, th;
// 		r1 = X[1];
// 		th = X[2];

// 		for (int i = 0; i < NDIM; i++)
// 				for (int j = 0; j < NDIM; j++)
// 						for (int k = 0; k < NDIM; k++)
// 							conn[i][j][k] = 0.;
// 		/*Taken from https://arxiv.org/pdf/0904.4184*/
// 		conn[1][2][2] = -r1;
// 		conn[2][3][3] = - sin(th) * cos(th);
// 		conn[1][3][3] = - r1 * pow(sin(th), 2.);
// 		conn[3][1][3] = 1./r1;
// 		conn[3][3][1] = 1./r1;
// 		conn[2][2][1] = 1./r1; 
// 		conn[2][1][2] = 1./r1; 
// 		conn[3][2][3] = 1/tan(th);
// 		conn[3][3][2] = 1/tan(th);
// 	#else
//     double tmp[NDIM][NDIM][NDIM];
//     double Xh[NDIM], Xl[NDIM];
//     double gcon[NDIM][NDIM];
//     double gcov[NDIM][NDIM];
//     double gh[NDIM][NDIM];
//     double gl[NDIM][NDIM];

//     gcov_func(X, gcov);
//     gcon_func(gcov, gcon);

//     // take partial derivatives of metric
//     for (int k = 0; k < NDIM; k++) {
//       for (int l = 0; l < NDIM; l++)   Xh[l] = X[l];
//       for (int l = 0; l < NDIM; l++)   Xl[l] = X[l];
//       Xh[k] += DEL;
//       Xl[k] -= DEL;
//       gcov_func(Xh, gh);
//       gcov_func(Xl, gl);

//       for (int i = 0; i < NDIM; i++){
//         for (int j = 0; j < NDIM; j++){
//           conn[i][j][k] =  (gh[i][j] - gl[i][j])/(Xh[k] - Xl[k]);
//         }
//       }
//     }

//     // Rearrange to find \Gamma_{ijk}
//     for (int i = 0; i < NDIM; i++)
//       for (int j = 0; j < NDIM; j++)
//         for (int k = 0; k < NDIM; k++)
//           tmp[i][j][k] =  0.5 * (conn[j][i][k] + conn[k][i][j] - conn[k][j][i]);

//     // G_{ijk} -> G^i_{jk}
//     for (int i = 0; i < NDIM; i++) {
//       for (int j = 0; j < NDIM; j++) {
//         for (int k = 0; k < NDIM; k++) {
//           conn[i][j][k] = 0.;
//           for (int l = 0; l < NDIM; l++) 
//             conn[i][j][k] += gcon[i][l]*tmp[l][j][k];
//         }
//       }
//     }
//   #endif
// }
// #undef DEL


__device__ void GPU_get_connection(double X[4], double lconn[4][4][4])
{
	#if(SPHERE_TEST)
		double r1, th;
		r1 = X[1];
		th = X[2];

		for (int i = 0; i < NDIM; i++)
				for (int j = 0; j < NDIM; j++)
						for (int k = 0; k < NDIM; k++)
							lconn[i][j][k] = 0.;
		/*Taken from https://arxiv.org/pdf/0904.4184*/
		lconn[1][2][2] = -r1;
		lconn[2][3][3] = - sin(th) * cos(th);
		lconn[1][3][3] = - r1 * pow(sin(th), 2.);
		lconn[3][1][3] = 1./r1;
		lconn[3][3][1] = 1./r1;
		lconn[2][2][1] = 1./r1; 
		lconn[2][1][2] = 1./r1; 
		lconn[3][2][3] = 1/tan(th);
		lconn[3][3][2] = 1/tan(th);

	#else

	double r1, r2, r3, r4, sx, cx;
	double th, dthdx2, dthdx22, d2thdx22, sth, cth, sth2, cth2, sth4,
	    cth4, s2th, c2th;
	double a2, a3, a4, rho2, irho2, rho22, irho22, rho23, irho23,
	    irho23_dthdx2;
	double fac1, fac1_rho23, fac2, fac3, a2cth2, a2sth2, r1sth2,
	    a4cth4;
	/* required by broken math.h */
	//void sincos(double th, double *sth, double *cth);

	r1 = exp(X[1]);
	r2 = r1 * r1;
	r3 = r2 * r1;
	r4 = r3 * r1;

	//sincos(2. * M_PI * X[2], &sx, &cx);
	sx = sin(2 * M_PI * X[2]);
	cx = cos(2 * M_PI * X[2]);
	/* HARM-2D MKS */
	#if(HAMR)
		double x2_mod;
		x2_mod = (X[2] + 1.)/2.;
		th = M_PI * x2_mod;
		dthdx2 = M_PI * (1./2.);
		d2thdx22 = 0;
	#else
	th = M_PI * X[2] + 0.5 * (1 - d_hslope) * sx;
	dthdx2 = M_PI * (1. + (1 - d_hslope) * cx);
	d2thdx22 = -2. * M_PI * M_PI * (1 - d_hslope) * sx;
	#endif
	dthdx22 = dthdx2 * dthdx2;

	//sincos(th, &sth, &cth);
	sth = sin(th);
	cth = cos(th);
	sth2 = sth * sth;
	r1sth2 = r1 * sth2;
	sth4 = sth2 * sth2;
	cth2 = cth * cth;
	cth4 = cth2 * cth2;
	s2th = 2. * sth * cth;
	c2th = 2 * cth2 - 1.;

	a2 = d_a * d_a;
	a2sth2 = a2 * sth2;
	a2cth2 = a2 * cth2;
	a3 = a2 * d_a;
	a4 = a3 * d_a;
	a4cth4 = a4 * cth4;

	rho2 = r2 + a2cth2;                
	rho22 = rho2 * rho2;
	rho23 = rho22 * rho2;
	irho2 = 1. / rho2;
	irho22 = irho2 * irho2;
	irho23 = irho22 * irho2;
	irho23_dthdx2 = irho23 / dthdx2;

	fac1 = r2 - a2cth2;
	fac1_rho23 = fac1 * irho23;
	fac2 = a2 + 2 * r2 + a2 * c2th;
	fac3 = a2 + r1 * (-2. + r1);

	lconn[0][0][0] = 2. * r1 * fac1_rho23;
	lconn[0][0][1] = r1 * (2. * r1 + rho2) * fac1_rho23;
	lconn[0][0][2] = -a2 * r1 * s2th * dthdx2 * irho22;
	// printf("a2 = %le\n", a2);
	// printf("r1 = %le\n", r1);
	// printf("irho22 = %le\n", irho22);
	// printf("s2th = %le\n", s2th);
	// printf("dthdx2 = %le\n", dthdx2);

	lconn[0][0][3] = -2. * d_a * r1sth2 * fac1_rho23;

	//lconn[0][1][0] = lconn[0][0][1];
	lconn[0][1][1] = 2. * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
	lconn[0][1][2] = -a2 * r2 * s2th * dthdx2 * irho22;
	lconn[0][1][3] =
	    d_a * r1 * (-r1 * (r3 + 2 * fac1) + a4cth4) * sth2 * irho23;

	//lconn[0][2][0] = lconn[0][0][2];
	//lconn[0][2][1] = lconn[0][1][2];
	lconn[0][2][2] = -2. * r2 * dthdx22 * irho2;
	lconn[0][2][3] = a3 * r1sth2 * s2th * dthdx2 * irho22;
	//lconn[0][3][0] = lconn[0][0][3];
	//lconn[0][3][1] = lconn[0][1][3];
	//lconn[0][3][2] = lconn[0][2][3];
	lconn[0][3][3] =
	    2. * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;

	lconn[1][0][0] = fac3 * fac1 / (r1 * rho23);
	lconn[1][0][1] = fac1 * (-2. * r1 + a2sth2) * irho23;
	lconn[1][0][2] = 0.;
	lconn[1][0][3] = -d_a * sth2 * fac3 * fac1 / (r1 * rho23);

	//lconn[1][1][0] = lconn[1][0][1];
	lconn[1][1][1] =
	    (r4 * (-2. + r1) * (1. + r1) +
	     a2 * (a2 * r1 * (1. + 3. * r1) * cth4 + a4cth4 * cth2 +
		   r3 * sth2 + r1 * cth2 * (2. * r1 + 3. * r3 -
					    a2sth2))) * irho23;
	lconn[1][1][2] = -a2 * dthdx2 * s2th / fac2;
	lconn[1][1][3] =
	    d_a * sth2 * (a4 * r1 * cth4 + r2 * (2 * r1 + r3 - a2sth2) +
			a2cth2 * (2. * r1 * (-1. + r2) + a2sth2)) * irho23;

	//lconn[1][2][0] = lconn[1][0][2];
	//lconn[1][2][1] = lconn[1][1][2];
	lconn[1][2][2] = -fac3 * dthdx22 * irho2;
	lconn[1][2][3] = 0.;

	//lconn[1][3][0] = lconn[1][0][3];
	//lconn[1][3][1] = lconn[1][1][3];
	//lconn[1][3][2] = lconn[1][2][3];
	lconn[1][3][3] =
	    -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);

	lconn[2][0][0] = -a2 * r1 * s2th * irho23_dthdx2;
	lconn[2][0][1] = r1 * lconn[2][0][0];
	lconn[2][0][2] = 0.;
	lconn[2][0][3] = d_a * r1 * (a2 + r2) * s2th * irho23_dthdx2;

	//lconn[2][1][0] = lconn[2][0][1];
	lconn[2][1][1] = r2 * lconn[2][0][0];
	lconn[2][1][2] = r2 * irho2;
	lconn[2][1][3] =
	    (d_a * r1 * cth * sth *
	     (r3 * (2. + r1) +
	      a2 * (2. * r1 * (1. + r1) * cth2 + a2 * cth4 +
		    2 * r1sth2))) * irho23_dthdx2;

	//lconn[2][2][0] = lconn[2][0][2];
	//lconn[2][2][1] = lconn[2][1][2];
	lconn[2][2][2] =
	    -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;

	lconn[2][2][3] = 0.;

	//lconn[2][3][0] = lconn[2][0][3];
	//lconn[2][3][1] = lconn[2][1][3];
	//lconn[2][3][2] = lconn[2][2][3];
	lconn[2][3][3] =
	    -cth * sth * (rho23 +
			  a2sth2 * rho2 * (r1 * (4. + r1) + a2cth2) +
			  2. * r1 * a4 * sth4) * irho23_dthdx2;

	lconn[3][0][0] = d_a * fac1_rho23;
	lconn[3][0][1] = r1 * lconn[3][0][0];
	lconn[3][0][2] = -2. * d_a * r1 * cth * dthdx2 / (sth * rho22);
	lconn[3][0][3] = -a2sth2 * fac1_rho23;

	//lconn[3][1][0] = lconn[3][0][1];
	lconn[3][1][1] = d_a * r2 * fac1_rho23;
	lconn[3][1][2] =
	    -2 * d_a * r1 * (a2 + 2 * r1 * (2. + r1) +
			   a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
	lconn[3][1][3] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

	//lconn[3][2][0] = lconn[3][0][2];
	//lconn[3][2][1] = lconn[3][1][2];
	lconn[3][2][2] = -d_a * r1 * dthdx22 * irho2;
	lconn[3][2][3] =
	    dthdx2 * (0.25 * fac2 * fac2 * cth / sth +
		      a2 * r1 * s2th) * irho22;

	//lconn[3][3][0] = lconn[3][0][3];
	//lconn[3][3][1] = lconn[3][1][3];
	//lconn[3][3][2] = lconn[3][2][3];
	lconn[3][3][3] = (-d_a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
	#endif

}

__host__ __device__ void lower(double *ucon, double Gcov[NDIM][NDIM], double *ucov)
{

	ucov[0] = Gcov[0][0] * ucon[0]
	    + Gcov[0][1] * ucon[1]
	    + Gcov[0][2] * ucon[2]
	    + Gcov[0][3] * ucon[3];
	ucov[1] = Gcov[1][0] * ucon[0]
	    + Gcov[1][1] * ucon[1]
	    + Gcov[1][2] * ucon[2]
	    + Gcov[1][3] * ucon[3];
	ucov[2] = Gcov[2][0] * ucon[0]
	    + Gcov[2][1] * ucon[1]
	    + Gcov[2][2] * ucon[2]
	    + Gcov[2][3] * ucon[3];
	ucov[3] = Gcov[3][0] * ucon[0]
	    + Gcov[3][1] * ucon[1]
	    + Gcov[3][2] * ucon[2]
	    + Gcov[3][3] * ucon[3];

	return;
}