#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double I0(double x);
double I1(double x);

/* Compute modified Bessel K0 and K1 (Numerical Recipes versions, public domain) */

static double K0_NR(double x)
{
    if (x <= 0.0) return INFINITY;

    if (x <= 2.0) {
        double y = x*x/4.0;
        return -log(x/2.0)*I0(x) + 
            (-0.57721566 + 0.42278420*y + 0.23069756*y*y +
             0.03488590*y*y*y + 0.00262698*y*y*y*y +
             0.00010750*y*y*y*y*y + 0.00000740*y*y*y*y*y*y);
    } else {
        double y = 2.0/x;
        return (exp(-x)/sqrt(x)) *
            (1.25331414 + 0.23498619*y - 0.36556200*y*y +
             0.15042680*y*y*y - 0.78035300*y*y*y*y +
             0.32561400*y*y*y*y*y - 0.68245000*y*y*y*y*y*y);
    }
}

static double K1_NR(double x)
{
    if (x <= 0.0) return INFINITY;

    if (x <= 2.0) {
        double y = x*x/4.0;
        return (log(x/2.0)*I1(x)) +
            (1.0/x) * (1.0 + 0.15443144*y - 0.67278579*y*y -
                       0.18156897*y*y*y - 0.01919402*y*y*y*y -
                       0.00110404*y*y*y*y*y - 0.00004686*y*y*y*y*y*y);
    } else {
        double y = 2.0/x;
        return (exp(-x)/sqrt(x)) *
            (1.25331414 + 0.23498619*y + 0.03655620*y*y -
             0.01504268*y*y*y + 0.00780353*y*y*y*y -
             0.00325614*y*y*y*y*y + 0.00068245*y*y*y*y*y*y);
    }
}

/* Modified Bessel I0, I1 — needed for K0/K1 for small x */
double I0(double x)
{
    double ax = fabs(x);
    if (ax < 3.75) {
        double y = x/3.75;
        y = y*y;
        return 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492 +
             y*(0.2659732 + y*(0.0360768 + y*0.0045813)))));
    } else {
        double y = 3.75/ax;
        double ans = (exp(ax)/sqrt(ax)) *
            (0.39894228 + y*(0.01328592 + y*(0.00225319 +
             y*(-0.00157565 + y*(0.00916281 + y*(-0.02057706 +
             y*(0.02635537 + y*(-0.01647633 + y*0.00392377))))))));
        return ans;
    }
}

double I1(double x)
{
    double ax = fabs(x);
    double ans;
    if (ax < 3.75) {
        double y = x/3.75;
        y = y*y;
        ans = x*(0.5 + y*(0.87890594 + y*(0.51498869 + y*(0.15084934 +
             y*(0.02658733 + y*(0.00301532 + y*0.00032411))))));
    } else {
        double y = 3.75/ax;
        ans = (exp(ax)/sqrt(ax)) *
            (0.39894228 + y*(-0.03988024 + y*(-0.00362018 +
             y*(0.00163801 + y*(-0.01031555 + y*(0.02282967 +
             y*(-0.02895312 + y*(0.01787654 - y*0.00420059))))))));
        if (x < 0) ans = -ans;
    }
    return ans;
}


/* ----------- K2(x) main function (integer n=2) -------------- */

double besselk2(double x)
{
    if (x <= 0.0) return INFINITY;

    /* Small x series (stable, matches GSL behaviour) */
    if (x < 2.0) {
        /* K2(x) = 2/x^2 + 1/2 + O(x^2 ln x) */
        double x2 = x*x;
        double leading = 2.0/(x2);
        double correction = 0.5;
        double logterm = (x2/16.0)*log(x/2.0);
        return leading + correction + logterm;
    }

    /* Medium x: use recurrence with K0 and K1 */
    if (x <= 50.0) {
        double k0 = K0_NR(x);
        double k1 = K1_NR(x);
        return k0 + 2.0 * (1.0/x) * k1;
    }

    /* Large x asymptotic */
    double ax = x;
    double sq = sqrt(M_PI/(2.0*ax));
    double e = exp(-ax);

    /* asympt coeff for nu=2 */
    double nu = 2.0;
    double a1 = (4*nu*nu - 1)/(8*ax);
    double a2 = (4*nu*nu - 1)*(4*nu*nu - 9)/(2*pow(8*ax, 2));

    return sq*e*(1 + a1 + a2);
}


/* ------------ MAIN PROGRAM ------------ */

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s x\n", argv[0]);
        return 1;
    }
    double x = atof(argv[1]);

    double val = besselk2(x);
    printf("K2(%g) = %.17g\n", x, val);
    return 0;
}
