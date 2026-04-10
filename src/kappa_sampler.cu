/*
 * GPUmonty - kappa_sampler.cu
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
#include "utils.h"
#include "jnu_mixed.h"
// rewrite this

struct f_params {
    double u;
};


__device__ double brent_find_root(double (*f)(double, double, void*), void* params,
                                  double x0, double x1, double tol, int max_steps, double kappa) {
    
    double a = x0, b = x1, c = x0;
    double fa = f(a, kappa, params);
    double fb = f(b, kappa, params);
    double fc = fa;
    double d = b - a, e = d;

    if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
        printf("GPU Warning: Root not bracketed in [%e, %e]\n", a, b);
        return b;
    }

    
    for (int step = 0; step < max_steps; step++) {
        if (fabs(fc) < fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }

        // Tightly scope iteration thresholds
        double xm = 0.5 * (c - b);
        double tol1 = 2.0 * 2.2204460492503131e-16 * fabs(b) + 0.5 * tol;

        if (fabs(xm) <= tol1 || fb == 0.0) break;

        if (fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
            // Only instantiate s, p, q if we are doing interpolation
            double s = fb / fa;
            double p, q;

            if (a == c) {
                // Secant method
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic interpolation
                double q_temp = fa / fc; 
                double r = fb / fc;
                p = s * (2.0 * xm * q_temp * (q_temp - r) - (b - a) * (r - 1.0));
                q = (q_temp - 1.0) * (r - 1.0) * (s - 1.0);
            }
            
            if (p > 0.0) q = -q;
            p = fabs(p);

            
            double min_bound = 3.0 * xm * q - fabs(tol1 * q);
            double e_q = fabs(e * q); 
            if (e_q < min_bound) min_bound = e_q; 

            if (2.0 * p < min_bound) {
                e = d; 
                d = p / q;
            } else {
                d = xm; 
                e = d;
            }
        } else {
            // Bisection
            d = xm; 
            e = d;
        }

        a = b; 
        fa = fb;
        
        if (fabs(d) > tol1) b += d;
        else b += (xm > 0.0 ? tol1 : -tol1);

        fb = f(b, kappa, params);

        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a; 
            fc = fa;
            d = b - a; 
            e = d; 
        }
    }

    return b;
}
__device__ double find_y(double u, double (*df)(double, double), double (*f)(double, double, void *), double w, double kappa) {
    double low = 1e-6; 
    double high = 1e6; 
    double tol = 1e-10;
    int max_steps = 1000;
    struct f_params params = {u};
    return brent_find_root(f, &params, low, high, tol, max_steps, kappa);
}

__device__ double dF3(double y, double kappa) {
    double value, denom, num;
    double y2 = y * y;

    num = 4. * y2 * pow((kappa + y2) / (kappa), -kappa - 1.) *
          tgamma(kappa);
    denom = sqrt(M_PI) * sqrt(kappa) * tgamma(kappa - 1. / 2.);
    value = num / denom;

    return value;
}

__device__ double dF4(double y, double kappa) {
    double y2 = y * y;
    double num = 2. * (kappa - 1) * y2 * y * pow((kappa + y2) / (kappa), -kappa - 1.);

    return num / kappa;
}

__device__ double dF5(double y, double kappa) {
    double value, denom, num;
    double y2 = y * y;

    num = 8 * y2 * y2 * pow((kappa + y2) / (kappa), -kappa - 1) *
          tgamma(kappa);
    denom =
        3 * sqrt(M_PI) * pow(kappa, 3. / 2.) * tgamma(kappa - 3. / 2.);
    value = num / denom;

    return value;
}

__device__ double dF6(double y, double kappa) {
    double value, denom, num;
    double y2 = y * y;

    num = (kappa * kappa - 3. * kappa + 2) * pow(y, 5.) *
          pow((kappa + y2) / (kappa), -kappa - 1);
    denom = kappa * kappa;
    value = num / denom;
    return value;
}

__device__ double F3(double y, double kappa, void *params) {
    struct f_params *p = (struct f_params *)params;
    double u = p->u;
    double value, denom, num, hyp2F1;

    double y2 = y * y;
    double z = -y2 / kappa;

    hyp2F1 = hypergeom_eval(-z, kappa);

    num = -sqrt(kappa) * pow(((y2 + kappa) / kappa), -kappa) *
          tgamma(kappa) *
          (-kappa * hyp2F1 + y2 * (2 * kappa + 1) + kappa);
    denom = y * sqrt(M_PI) * tgamma(3. / 2. + kappa);

    value = num / denom - u;

    return value;
}

__device__ double F4(double y, double kappa, void *params) {
    struct f_params *p = (struct f_params *)params;
    double u = p->u;
    double value, denom, num;
    double y2 = y * y;

    num = 1 - (1 + y2) * pow((kappa + y2) / (kappa), -kappa);
    denom = 1;

    value = num / denom - u;

    return value;
}

__device__ double F5(double y, double kappa, void *params) {
    struct f_params *p = (struct f_params *)params;
    double u = p->u;

    double value, denom, num, hyp2F1;

    double y2 = y * y;
    double z = -y2 / kappa;

    hyp2F1 = hypergeom_eval(-z, kappa);

    num = pow((y2 + kappa) / kappa, -kappa) * tgamma(kappa) *
          (3 * kappa * kappa * (hyp2F1 - 1) +
           (1. - 4. * kappa * kappa) * y2 * y2 -
           3. * kappa * (2. * kappa + 1.) * y2);
    denom = 3. * pow(kappa, 1. / 2.) * y * sqrt(M_PI) *
            tgamma(3. / 2. + kappa);
    value = num / denom - u;

    return value;
}

__device__ double F6(double y, double kappa, void *params) {
    struct f_params *p = (struct f_params *)params;
    double u = p->u;

    double value, denom, num;
    double y2 = y * y;
    double y4 = y2 * y2;

    num = (y4 - (y4 + 2. * y2 + 2.) * kappa) *
          pow((kappa + y2) / (kappa), -kappa);
    denom = 2 * kappa;
    value = num / denom + 1 - u;

    return value;
}

__device__ double sample_y_distr_nth(double Thetae, double kappa, curandState * localState) {
    double w = (kappa - 3.) / kappa * Thetae;
    double S_3, pi_3, pi_4, pi_5, pi_6, y = -1, x1, x2, prob;
    double num, den;

    pi_3 = sqrt(kappa) * sqrt(M_PI) * tgamma(-1. / 2. + kappa) /
           (4. * tgamma(kappa));
    pi_4 = kappa / (2. * kappa - 2.) * sqrt(0.5 * w);
    pi_5 = 3. * pow(kappa, 3. / 2.) * sqrt(M_PI) *
           tgamma(-3. / 2. + kappa) / (8. * tgamma(kappa)) * w;
    pi_6 =
        kappa * kappa / (2. - 3. * kappa + kappa * kappa) * w * sqrt(0.5 * w);

    S_3 = pi_3 + pi_4 + pi_5 + pi_6;

    pi_3 /= S_3;
    pi_4 /= S_3;
    pi_5 /= S_3;
    pi_6 /= S_3;

    do {
        x1 = curand_uniform_double(localState);
        double u = curand_uniform_double(localState);
        if (x1 < pi_3) {
            y = find_y(u, dF3, F3, w, kappa);
        } else if (x1 < pi_3 + pi_4) {
            y = find_y(u, dF4, F4, w, kappa);
        } else if (x1 < pi_3 + pi_4 + pi_5) {
            y = find_y(u, dF5, F5, w, kappa);
        } else {
            y = find_y(u, dF6, F6, w, kappa);
        }

        x2 = curand_uniform_double(localState);
        num = sqrt(1. + 0.5 * w * y * y);
        den = (1. + y * sqrt(0.5 * w));
        prob = (num / den) * exp(-(w * y * y) / GAMMA_MAX);
        if (y != y)
            prob = 0;

    } while (x2 >= prob);

    return (y);
}
