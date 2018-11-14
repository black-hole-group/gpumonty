/* bessel.c - extracted from ftp://ftp.astro.rug.nl/gipsy/src/gipsy_src_23May2011.tar.gz and
    modified by Matheus Bernardino (matheus.bernardino@usp.br) at 25/07/2018

                      Copyright (c) 1998
                  Kapteyn Institute Groningen
                     All Rights Reserved.


COPYRIGHT Release 3.8

           Groningen Image Processing SYstem (GIPSY)

             COPYRIGHT (c) 1978, 1984, 1992, 1993, 1994, 1995, 2000, 2001

             Kapteyn Astronomical Institute,
             University of Groningen
             P.O. Box 800
             9700 AV Groningen
             The Netherlands


     The information in this document is subject to change without
     notice and should not be construed as a commitment by the Kapteyn
     Astronomical Institute.

     The Kapteyn Astronomical Institute assumes no responsibility  for
     the use or reliability of its software.

     Permission to use, copy, and distribute GIPSY software and its
     documentation for any purpose is hereby granted, provided that
     this copyright notice appears in all copies.

     Permission to modify the software is granted, but not the right
     to distribute the modified code. Modifications are to be distributed
     via the GIPSY source server, which is currently gipsy.astro.rug.nl.
     You can send your modifications to the GIPSY manager, who will take
     care of the distribution.

     Permission to install modified or new code directly can be obtained
     from the GIPSY Manager. The E-Mail address of the GIPSY Manager
     is listed in $gip_sys/manager.mgr.

     Reports of software failures will only be considered when you have
     an automatic update of GIPSY sources installed at your site. See
     the Download and Installation Guide:
     http://www.astro.rug.nl/~gipsy/installation/installation.html


     References: "The Groningen Image Processing SYstem, GIPSY",
                 J.M. van der Hulst, J.P. Terlouw, K.G. Begeman,
                 W. Zwitser and P.R. Roelfsema, in
                 Astronomical Data Analysis Software and Systems I,
                 ed. D.M. Worall, C. Biemesderfer and J. Barnes,
                 ASP Conf. series no. 25, p. 131.

                 "The Evolution of GIPSY, or the Survival of
                 an Image Processing System",  M.G.R. Vogelaar
                 and J.P. Terlouw, to appear in
                 Astronomical Data Analysis Software and Systems X,
                 eds. F.A. Primini, F.R. Harnden, Jr., H.E. Payne,
                 ASP Conf. series.


*/

/*
#>            bessel.dc2

Function:     BESSEL

Purpose:      Evaluate Bessel function J, Y, I, K of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          See bessj.dc2, bessy.dc2, bessi.dc2 or bessk.dc2

Description:  The differential equation

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- + (x - v ).w = 0
                        2       dx
                      dx

              has two solutions called Bessel functions of the first kind
              Jv(x) and Bessel functions of the second kind Yv(x).
              The routines bessj and bessy return the J and Y for
              integer v and therefore are called Bessel functions
              of integer order.

              The differential equation

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- - (x + v ).w = 0
                        2       dx
                      dx

              has two solutions called modified Bessel functions
              Iv(x) and Kv(x).
              The routines bessi and bessk return the I and K for
              integer v and therefore are called Modified Bessel
              functions of integer order.
              (Abramowitz & Stegun, Handbook of mathematical
              functions, ch. 9, pages 358,- and 374,- )

              The implementation is based on the ideas from
              Numerical Recipes, Press et. al.
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/

/*
#> bessel.h
#if !defined(_bessel_h_)
#define _bessel_h_
extern double bessj( int, double );
extern double bessy( int, double );
extern double bessi( int, double );
extern double bessk( int, double );
#endif
#<
*/



#include "bessel.h"
#include <math.h>


__host__ __device__
static double bessi0( double x );
__host__ __device__
static double bessi1( double x);
__host__ __device__
static double bessk0( double x );
__host__ __device__
static double bessk1( double x );


__host__ __device__
static double bessi0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function In(x) and n=0.  */
/*------------------------------------------------------------*/
{
   double ax,ans;
   double y;


   if ((ax=fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
         +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
   } else {
      y=3.75/ax;
      ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
         +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
         +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
         +y*0.392377e-2))))))));
   }
   return ans;
}



__host__ __device__
static double bessi1( double x)
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function In(x) and n=1.  */
/*------------------------------------------------------------*/
{
   double ax,ans;
   double y;


   if ((ax=fabs(x)) < 3.75) {
      y=x/3.75,y=y*y;
      ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
         +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
   } else {
      y=3.75/ax;
      ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
         -y*0.420059e-2));
      ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
         +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
      ans *= (exp(ax)/sqrt(ax));
   }
   return x < 0.0 ? -ans : ans;
}


__host__ __device__
static double bessk0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=0.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;
}



__host__ __device__
static double bessk1( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=1.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144
         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
         +y*(0.325614e-2+y*(-0.68245e-3)))))));
   }
   return ans;
}


/*
#>            bessk.dc2

Function:     bessk

Purpose:      Evaluate Modified Bessel function Kv(x) of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          #include "bessel.h"
              double   result;
              result = bessk( int n,
                              double x )


              bessk    Return the Modified Bessel function Kv(x) of
                       integer order for input value x.
              n        Integer order of Bessel function.
              x        Double at which the function is evaluated.


Description:  bessk evaluates at x the Modified Bessel function Kv(x) of
              integer order n.
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/


__host__ __device__
double bessk( int n, double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n >= 0*/
/* Note that for x == 0 the functions bessy and bessk are not */
/* defined and a -1E40 is returned.                           */
/*------------------------------------------------------------*/
{
   int j;
   double bk,bkm,bkp,tox;


   if (n < 0 || x == 0.0)
   {
       return -1E40;
   }
   if (n == 0)
      return( bessk0(x) );
   if (n == 1)
      return( bessk1(x) );

   tox=2.0/x;
   bkm=bessk0(x);
   bk=bessk1(x);
   for (j=1;j<n;j++) {
      bkp=bkm+j*tox*bk;
      bkm=bk;
      bk=bkp;
   }
   return bk;
}
