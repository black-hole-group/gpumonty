/*
 * GPUmonty - metrics.cu
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
#include "decs.h"
#include "metrics.h"
#include "model.h"
/* 
	In this file, given gcov_func in the model, we can calculate the gcon, gdet and also the connection terms.
*/

gsl_matrix *gsl_gcov, *gsl_gcon;
gsl_permutation *perm;
#pragma omp threadprivate (gsl_gcov, gsl_gcon, perm)

/* assumes gcov has been set first; returns determinant */
double gdet_func(double gcov[][NDIM])
{
	#ifdef SPHERE_TEST
		return sqrt(-gcov[0][0] * gcov[1][1] * gcov[2][2] * gcov[3][3]);
	#endif
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



#ifdef FIND_GCON_MATRIX_INV
	__host__ __device__ void gcon_func(const double X[4], double gcov[][NDIM], double gcon[][NDIM])
	{
	invert_matrix( gcov, gcon );
	}
#else
	__host__ __device__ void gcon_func(const double X[4], double gcov[][NDIM], double gcon[][NDIM])
	{
		int k, l;
		#ifdef SPHERE_TEST
		DLOOP gcon[k][l] = 0.;
		/*Flat space in spherical coordinates for the test*/

		gcon[0][0] = -1.;
		gcon[1][1] = 1./gcov[1][1];
		gcon[2][2] = 1./gcov[2][2];
		gcon[3][3] = 1./gcov[3][3];
		#else
		
			double irho2;
			double r, th;
			double hfac;
			double sth, cth;

			DLOOP gcon[k][l] = 0.;
			bl_coord(X, &r, &th);

			#ifdef __CUDA_ARCH__
			double thetaslope = d_hslope;
			double local_bhspin = d_bhspin;
			#else
			double thetaslope = hslope;
			double local_bhspin = bhspin;
			#endif

			sincos(th, &sth, &cth);
			sth = fabs(sth) + SMALL;

			irho2 = 1. / (r * r + local_bhspin * local_bhspin * cth * cth);

			//transformation for Kerr-Schild -> modified Kerr-Schild 
			hfac = M_PI + (1. - thetaslope) * M_PI * cos(2. * M_PI * X[2]);

			gcon[0][0] = -1. - 2. * r * irho2;
			gcon[0][1] = 2. * irho2;

			gcon[1][0] = gcon[0][1];
			gcon[1][1] = irho2 * (r * (r - 2.) + local_bhspin * local_bhspin) / (r * r);
			gcon[1][3] = local_bhspin * irho2 / r;

			gcon[2][2] = irho2 / (hfac * hfac);

			gcon[3][1] = gcon[1][3];
			gcon[3][3] = irho2 / (sth * sth);
		#endif
	}
#endif

#ifndef SPHERE_TEST

	__device__ void ConnectionAnalyticalWrapper(const double X[4], double lconn[4][4][4])
	{
		if(d_METRIC == METRIC_MKS){
			ConnectionAnalyticalMKS(X, lconn);
		}else if(d_METRIC == METRIC_MKS3 || d_METRIC == METRIC_FMKS){
			ConnectionAnalytical_MKS3_FMKS(X, lconn);
		}else{
			printf("Error: ConnectionAnalyticalWrapper called with unknown metric type %d, the code will crash!\n", d_METRIC);
		}
	}
	__device__ void ConnectionAnalytical_MKS3_FMKS(const double X[4], double lconn[4][4][4]){
		double r1, drdx1, th, dthdx1, dthdx2, d2thdx12, d2thdx1dx2, d2thdx22;
		{
			if(d_METRIC == METRIC_MKS3){
				r1 = exp(X[1]) + d_mks3R0;
				drdx1 = exp(X[1]);
				
				double r_p = pow(r1, -d_mks3MP0);
				double radial_term = d_mks3MY1 + pow(2.0, d_mks3MP0) * (-d_mks3MY1 + d_mks3MY2) * r_p;
				double dradial_dx1 = -d_mks3MP0 * drdx1 * pow(2.0, d_mks3MP0) * (-d_mks3MY1 + d_mks3MY2) * (r_p / r1);
				
				double deriv_factor = 1.0 - 2.0 * radial_term;
				double half_pi = M_PI * 0.5;
				double cot_term = 1.0 / tan(d_mks3H0 * half_pi);
				
				double x2_term = 1.0 - 2.0 * X[2];
				double inner_arg = d_mks3H0 * M_PI * (-0.5 + radial_term * x2_term + X[2]);
				
				double c_inner = cos(inner_arg);
				double sec2_term = 1.0 / (c_inner * c_inner);
				double t_inner = tan(inner_arg);
				
				double common_chain = 0.5 * d_mks3H0 * M_PI * M_PI * cot_term * sec2_term;
				
				th = half_pi * (1.0 + cot_term * t_inner);
				dthdx1 = common_chain * x2_term * dradial_dx1;
				dthdx2 = common_chain * deriv_factor;
				
				d2thdx22 = d_mks3H0 * d_mks3H0 * deriv_factor * deriv_factor * M_PI * M_PI * M_PI * cot_term * sec2_term * t_inner;
				
				d2thdx1dx2 = 2.0 * common_chain * dradial_dx1 * (d_mks3H0 * M_PI * x2_term * deriv_factor * t_inner - 1.0);
				
				double d2radial_dx11 = dradial_dx1 * (1.0 - (1.0 + d_mks3MP0) * drdx1 / r1);
				d2thdx12 = common_chain * x2_term * (
					2.0 * d_mks3H0 * M_PI * t_inner * x2_term * dradial_dx1 * dradial_dx1 
					+ d2radial_dx11
				);
			}else if(d_METRIC == METRIC_FMKS){
				r1 = exp(X[1]);
				drdx1 = exp(X[1]);
				double y = 2.0 * X[2] - 1.0;
				double y_over_xt = y / d_poly_xt;
				double pow_alpha = pow(y_over_xt, d_poly_alpha);

				double thG = M_PI * X[2] + ((1.0 - d_hslope) / 2.0) * sin(2.0 * M_PI * X[2]);
				double thJ = d_poly_norm * y * (1.0 + pow_alpha / (d_poly_alpha + 1.0)) + 0.5 * M_PI;

				double W = exp(d_mks_smooth * (d_startx[1] - X[1]));
				double dWdx1 = -d_mks_smooth * W;
				double d2Wdx12 = d_mks_smooth * d_mks_smooth * W;

				// 3. Derivatives of thG with respect to X[2]
				double dthGdx2 = M_PI + M_PI * (1.0 - d_hslope) * cos(2.0 * M_PI * X[2]);
				double d2thGdx22 = -2.0 * M_PI * M_PI * (1.0 - d_hslope) * sin(2.0 * M_PI * X[2]);

				// 4. Derivatives of thJ with respect to X[2] (Analytically simplified)
				double dthJdx2 = 2.0 * d_poly_norm * (1.0 + pow_alpha);
				double pow_alpha_minus_1 = pow(y_over_xt, d_poly_alpha - 1.0);
				double d2thJdx22 = 4.0 * d_poly_norm * d_poly_alpha / d_poly_xt * pow_alpha_minus_1;

				// 5. Final Combined Derivatives
				th = thG + W * (thJ - thG);

				dthdx1 = dWdx1 * (thJ - thG);
				dthdx2 = dthGdx2 + W * (dthJdx2 - dthGdx2);

				d2thdx12 = d2Wdx12 * (thJ - thG);
				d2thdx1dx2 = dWdx1 * (dthJdx2 - dthGdx2);
				d2thdx22 = d2thGdx22 + W * (d2thJdx22 - d2thGdx22);
			}
		} 


	
		double r2, r3, r4;
		double sth, cth, sth2, cth2, sth4, cth4, s2th, c2th;
		double a2, a3, a4, a2sth2, a2cth2, a4cth4, r1sth2;
		double rho2, rho22, rho23, irho2, irho22, irho23;
		double fac1, fac2, fac3, fac1_rho23, gamma002;
		{
			r2 = r1 * r1;
			r3 = r2 * r1;
			r4 = r3 * r1;

			sincos(th, &sth, &cth);

			sth2 = sth * sth;
			r1sth2 = r1 * sth2;
			sth4 = sth2 * sth2;
			cth2 = cth * cth;
			cth4 = cth2 * cth2;
			s2th = 2.0 * sth * cth;
			c2th = 2.0 * cth2 - 1.0;

			a2 = d_bhspin * d_bhspin;
			a2sth2 = a2 * sth2;
			a2cth2 = a2 * cth2;
			a3 = a2 * d_bhspin;
			a4 = a3 * d_bhspin;
			a4cth4 = a4 * cth4;

			rho2 = r2 + a2cth2;                
			rho22 = rho2 * rho2;
			rho23 = rho22 * rho2;
			
			irho2 = 1.0 / rho2;
			irho22 = irho2 * irho2;
			irho23 = irho22 * irho2;

			fac1 = r2 - a2cth2;
			fac1_rho23 = fac1 * irho23;
			fac2 = a2 + 2.0 * r2 + a2 * c2th; 
			fac3 = a2 + r1 * (-2.0 + r1);
			
			gamma002 = -a2 * r1 * s2th * irho22;
		} 


	
		{
			lconn[0][0][0] = 2.0 * r1 * fac1_rho23;
			lconn[0][0][1] = drdx1 * (2.0 * r1 + rho2) * fac1_rho23 + gamma002 * dthdx1;
			lconn[0][0][2] = gamma002 * dthdx2;
			lconn[0][0][3] = -2.0 * d_bhspin * r1sth2 * fac1_rho23;

			double base_011 = 2.0 * (r4 + r1 * fac1 - a4cth4) * irho23;
			double drdx1_2 = drdx1 * drdx1;
			double dthdx1_2 = dthdx1 * dthdx1;

			lconn[0][1][1] = base_011 * drdx1_2 + 2.0 * gamma002 * drdx1 * dthdx1 - 2.0 * r2 * irho2 * dthdx1_2;
			lconn[0][1][2] = dthdx2 * (gamma002 * drdx1 - 2.0 * r2 * irho2 * dthdx1);
			lconn[0][1][3] = d_bhspin * drdx1 * (-r1 * (r3 + 2.0 * fac1) + a4cth4) * sth2 * irho23 
						+ a3 * r1sth2 * s2th * irho22 * dthdx1;

			double dthdx2_2 = dthdx2 * dthdx2;
			lconn[0][2][2] = -2.0 * r2 * irho2 * dthdx2_2;
			lconn[0][2][3] = a3 * r1sth2 * s2th * irho22 * dthdx2;
			
			lconn[0][3][3] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;
		} 


	
		{
			double idrdx1 = 1.0 / drdx1;
			double idthdx2 = 1.0 / dthdx2;
			double idrdx1_idthdx2 = idthdx2 * idrdx1;
			
			lconn[1][0][0] = fac3 * fac1 * irho23 * idrdx1;
			lconn[1][0][1] = fac1 * (-2.0 * r1 + a2sth2) * irho23;
			lconn[1][0][2] = 0.0;
			lconn[1][0][3] = -d_bhspin * sth2 * lconn[1][0][0];

			double term_111_1 = -(r2 - a2cth2) * (4.0 * r1 + fac3 - 2.0 * a2sth2) * irho23;
			double term_111_2 = -2.0 * a2 * s2th / fac2;
			double term_111_3 = -fac3 * irho2;
			
			double dthdx1_2 = dthdx1 * dthdx1;
			
			lconn[1][1][1] = term_111_1 * drdx1 + 1.0 + term_111_2 * dthdx1 + term_111_3  * dthdx1_2;

			lconn[1][1][2] = dthdx2 * (0.5 * term_111_2 + term_111_3 * r1 * idrdx1 * dthdx1);
			
			lconn[1][1][3] = d_bhspin * sth2 * (a4 * r1 * cth4 + r2 * (2.0 * r1 + r3 - a2sth2) 
							+ a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) * irho23;
			
			double dthdx2_2 = dthdx2 * dthdx2;
			lconn[1][2][2] = term_111_3 * r1 * idrdx1 * dthdx2_2;
			lconn[1][2][3] = 0.0;
			lconn[1][3][3] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) * irho23 * idrdx1;

			double gamma002_irho2 = gamma002 * irho2;
			
			lconn[2][0][0] = gamma002_irho2 * idthdx2 - fac3 * fac1_rho23 * dthdx1 * idrdx1_idthdx2;
			lconn[2][0][1] = (gamma002_irho2 * drdx1 - lconn[1][0][1] * dthdx1) * idthdx2; // reused lconn[1][0][1]
			lconn[2][0][2] = 0.0;
			lconn[2][0][3] = d_bhspin * r1 * (a2 + r2) * s2th * irho23 * idthdx2 
						+ d_bhspin * sth2 * fac3 * fac1 * irho23 * dthdx1 * idrdx1_idthdx2;
						
			double drdx1_2 = drdx1 * drdx1;
			double dthdx1_3 = dthdx1_2 * dthdx1;

			lconn[2][1][1] = (
				gamma002_irho2 * drdx1_2
				- term_111_1 * drdx1 * dthdx1
				+ 2.0 * r1 * irho2 * drdx1 * dthdx1
				- term_111_2 * dthdx1_2
				- a2 * cth * sth * irho2 * dthdx1_2
				- term_111_3 * dthdx1_3
				- dthdx1 + d2thdx12
			) * idthdx2;

			lconn[2][1][2] = r1 * irho2 * drdx1 
						- 0.5 * term_111_2 * dthdx1 
						- a2 * cth * sth * irho2 * dthdx1 
						- term_111_3 * r1 * idrdx1 * dthdx1_2 
						+ d2thdx1dx2 * idthdx2;

			double term_213_a = d_bhspin * cth * sth * (r3 * (2.0 + r1) 
							+ a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2.0 * r1sth2)) * irho23;
			lconn[2][1][3] = (term_213_a * drdx1 - lconn[1][1][3] * dthdx1) * idthdx2; // reused lconn[1][1][3]

			lconn[2][2][2] = -a2 * cth * sth * irho2 * dthdx2 
						+ d2thdx22 * idthdx2 
						- term_111_3 * r1 * idrdx1 * dthdx2 * dthdx1; 
						
			lconn[2][2][3] = 0.0;

			double term_233_a = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) 
								+ 2.0 * r1 * a4 * sth4) * irho23;
			lconn[2][3][3] = (term_233_a - lconn[1][3][3] *dthdx1) * idthdx2; // reused lconn[1][3][3]
	


		} 


		
		{
			lconn[3][0][0] = d_bhspin * fac1_rho23;
			
			double term_301_b = -2.0 * d_bhspin * r1 * cth / (sth * rho22); 
			lconn[3][0][1] = lconn[3][0][0] * drdx1 + term_301_b * dthdx1;
			lconn[3][0][2] = term_301_b * dthdx2;
			lconn[3][0][3] = -a2sth2 * fac1_rho23;

			double term_311_2 = -2.0 * d_bhspin * (a2 + 2.0 * r1 * (2.0 + r1) + a2 * c2th) * cth / (sth * fac2 * fac2);
			double term_311_3 = -d_bhspin * r1 * irho2;
			
			double drdx1_2 = drdx1 * drdx1;
			double dthdx1_2 = dthdx1 * dthdx1;

			lconn[3][1][1] = lconn[3][0][0] * drdx1_2 + 2.0 * term_311_2 * drdx1 * dthdx1 + term_311_3 * dthdx1_2;
			

			lconn[3][1][2] = term_311_2 * drdx1 * dthdx2 + term_311_3 * dthdx2 * dthdx1;

			double term_313_1 = (r1 * rho22 - a2sth2 * fac1) * irho23;
			double term_313_2 = (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22;
			
			lconn[3][1][3] = term_313_1 * drdx1 + term_313_2 * dthdx1;

			double dthdx2_2 = dthdx2 * dthdx2;
			lconn[3][2][2] = term_311_3 * dthdx2_2;
			lconn[3][2][3] = term_313_2 * dthdx2;
			
			lconn[3][3][3] = (-d_bhspin * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
		} 
	}

	
	__device__ void ConnectionAnalyticalMKS(const double X[NDIM], double lconn[NDIM][NDIM][NDIM])
{
    double r1, r2, r3, r4;
    double sth, cth, sth2, cth2, sth4, cth4, s2th, c2th;
    double a2, a3, a4, a2cth2, a2sth2, r1sth2, a4cth4;
    double rho2, irho2, rho22, irho22, rho23, irho23;
    double fac1, fac1_rho23, fac2, fac3;
    double dthdx2, dthdx22, d2thdx22, irho23_dthdx2;

    r1 = exp(X[1]);
    r2 = r1 * r1;
    r3 = r2 * r1;
    r4 = r3 * r1;

    {
        double sx = sin(2. * M_PI * X[2]);
        double cx = cos(2. * M_PI * X[2]);
        double th = M_PI * X[2] + 0.5 * (1. - d_hslope) * sx;
        dthdx2    = M_PI * (1. + (1. - d_hslope) * cx);
        d2thdx22  = -2. * M_PI * M_PI * (1. - d_hslope) * sx;
        dthdx22   = dthdx2 * dthdx2;
        sincos(th, &sth, &cth);
    } 

    sth2   = sth * sth;
    r1sth2 = r1 * sth2;
    sth4   = sth2 * sth2;
    cth2   = cth * cth;
    cth4   = cth2 * cth2;
    s2th   = 2. * sth * cth;
    c2th   = 2. * cth2 - 1.;

    a2     = d_bhspin * d_bhspin;
    a2sth2 = a2 * sth2;
    a2cth2 = a2 * cth2;
    a3     = a2 * d_bhspin;
    a4     = a3 * d_bhspin;
    a4cth4 = a4 * cth4;

    rho2   = r2 + a2cth2;
    rho22  = rho2 * rho2;
    rho23  = rho22 * rho2;
    irho2  = 1. / rho2;
    irho22 = irho2 * irho2;
    irho23 = irho22 * irho2;
    irho23_dthdx2 = irho23 / dthdx2;

    fac1       = r2 - a2cth2;
    fac1_rho23 = fac1 * irho23;
    fac2       = a2 + 2. * r2 + a2 * c2th;
    fac3       = a2 + r1 * (-2. + r1);

    //
    {
        double c00 = 2. * r1 * fac1_rho23;
        double c02 = -a2 * r1 * s2th * dthdx2 * irho22;
        lconn[0][0][0] = c00;
        lconn[0][0][1] = r1 * (2. * r1 + rho2) * fac1_rho23;
        lconn[0][0][2] = c02;
        lconn[0][0][3] = -2. * d_bhspin * r1sth2 * fac1_rho23;

        lconn[0][1][1] = 2. * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
        lconn[0][1][2] = r1 * c02;
        lconn[0][1][3] = d_bhspin * r1 * (-r1 * (r3 + 2. * fac1) + a4cth4) * sth2 * irho23;

        lconn[0][2][2] = -2. * r2 * dthdx22 * irho2;
        lconn[0][2][3] = a3 * r1sth2 * s2th * dthdx2 * irho22;
        lconn[0][3][3] = 2. * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;
    }

    //lconn[1]
    {
        double ir1rho23 = irho23 / r1;
        double f3f1     = fac3 * fac1;
        lconn[1][0][0] = f3f1 * ir1rho23;
        lconn[1][0][1] = fac1 * (-2. * r1 + a2sth2) * irho23;
        lconn[1][0][2] = 0.;
        lconn[1][0][3] = -d_bhspin * sth2 * f3f1 * ir1rho23;

        lconn[1][1][1] = (r4 * (-2. + r1) * (1. + r1) +
                          a2 * (a2 * r1 * (1. + 3. * r1) * cth4 + a4cth4 * cth2 +
                          r3 * sth2 + r1 * cth2 * (2. * r1 + 3. * r3 - a2sth2))) * irho23;
        lconn[1][1][2] = -a2 * dthdx2 * s2th / fac2;
        lconn[1][1][3] = d_bhspin * sth2 * (a4 * r1 * cth4 + r2 * (2. * r1 + r3 - a2sth2) +
                          a2cth2 * (2. * r1 * (-1. + r2) + a2sth2)) * irho23;

        lconn[1][2][2] = -fac3 * dthdx22 * irho2;
        lconn[1][2][3] = 0.;
        lconn[1][3][3] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) * ir1rho23;
    } 

    //lconn[2]
    {
        double base = -a2 * r1 * s2th * irho23_dthdx2;
        lconn[2][0][0] = base;
        lconn[2][0][1] = r1  * base;
        lconn[2][0][2] = 0.;
        lconn[2][0][3] = d_bhspin * r1 * (a2 + r2) * s2th * irho23_dthdx2;

        lconn[2][1][1] = r2 * base;
        lconn[2][1][2] = r2 * irho2;
        lconn[2][1][3] = (d_bhspin * r1 * cth * sth *
                          (r3 * (2. + r1) +
                           a2 * (2. * r1 * (1. + r1) * cth2 + a2 * cth4 + 2. * r1sth2)))
                         * irho23_dthdx2;

        lconn[2][2][2] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;
        lconn[2][2][3] = 0.;
        lconn[2][3][3] = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4. + r1) + a2cth2) +
                          2. * r1 * a4 * sth4) * irho23_dthdx2;
    }

    //lconn[3]
    {
        double c30    = d_bhspin * fac1_rho23;
        double cth_sth = cth / sth;
        lconn[3][0][0] = c30;
        lconn[3][0][1] = r1 * c30;
        lconn[3][0][2] = -2. * d_bhspin * r1 * cth_sth * dthdx2 * irho22;
        lconn[3][0][3] = -a2sth2 * fac1_rho23;

        lconn[3][1][1] = d_bhspin * r2 * fac1_rho23;
        lconn[3][1][2] = -2. * d_bhspin * r1 * (a2 + 2. * r1 * (2. + r1) + a2 * c2th)
                         * cth_sth * dthdx2 / (fac2 * fac2);
        lconn[3][1][3] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

        lconn[3][2][2] = -d_bhspin * r1 * dthdx22 * irho2;
        lconn[3][2][3] = dthdx2 * (0.25 * fac2 * fac2 * cth_sth + a2 * r1 * s2th) * irho22;
        lconn[3][3][3] = (-d_bhspin * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
    }
}
#else
	#define DEL (1.e-7)
	__device__ void get_connection(const double X[NDIM], double lconn[NDIM][NDIM][NDIM])
	{
		double gcon[NDIM][NDIM];
		{
			double gcov[NDIM][NDIM];
			gcov_func(X, gcov);
			gcon_func(X, gcov, gcon);
		}
		// take partial derivatives of metric
		for (int k = 0; k < NDIM; k++) {
			double Xh[NDIM], Xl[NDIM];
			for (int l = 0; l < NDIM; l++){
				Xh[l] = X[l];
				Xl[l] = X[l];
			} 
			Xh[k] += DEL;
			Xl[k] -= DEL;
			double gh[NDIM][NDIM];
			double gl[NDIM][NDIM];
			gcov_func(Xh, gh);
			gcov_func(Xl, gl);

			for (int i = 0; i < NDIM; i++){
			for (int j = 0; j < NDIM; j++){
				lconn[i][j][k] =  (gh[i][j] - gl[i][j])/(2 * DEL);
			}
			}
		}

		// Rearrange to find \Gamma_{ijk}
		double tmp[NDIM][NDIM][NDIM];
		for (int i = 0; i < NDIM; i++){
			for (int j = 0; j < NDIM; j++){
				for (int k = 0; k < NDIM; k++){
					tmp[i][j][k] =  0.5 * (lconn[j][i][k] + lconn[k][i][j] - lconn[k][j][i]);
				}
			}
		}

		// G_{ijk} -> G^i_{jk}
		for (int i = 0; i < NDIM; i++) {
			for (int j = 0; j < NDIM; j++) {
				for (int k = 0; k < NDIM; k++) {
					lconn[i][j][k] = 0.;
					for (int l = 0; l < NDIM; l++) 
					lconn[i][j][k] += gcon[i][l]*tmp[l][j][k];
				}
		    }
		}
	}
	#undef DEL

	__device__ void ConnectionAnalyticalWrapper(const double X[4], double lconn[4][4][4])
	{
		return get_connection(X, lconn);
	}
#endif


__host__ __device__ void lower(double *ucon, const double Gcov[NDIM][NDIM], double *ucov)
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