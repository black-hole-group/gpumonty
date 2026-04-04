/*
 * GPUmonty - kappa_sampler.h
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


 #ifndef KAPPA_SAMPLER_H
 #define KAPPA_SAMPLER_H
    /**
     * @brief Finds the root of a 1D continuous function using Brent's method.
     *
     * This device-callable implementation relies on Brent's algorithm, which combines root bracketing, bisection, the secant method, and 
     * inverse quadratic interpolation. The initial interval [x0, x1] must strictly 
     * bracket a root (i.e., f(x0) and f(x1) must have opposite signs).
     *
     * @param f         Pointer to the target __device__ function whose root is being sought.
     * @param params    Void pointer to any additional state or parameters required by f.
     * @param x0        One endpoint of the initial search interval.
     * @param x1        The other endpoint of the initial search interval.
     * @param tol       The desired absolute tolerance/accuracy for the converged root.
     * @param max_steps The maximum number of algorithm iterations allowed.
     *
     * @return The estimated x-coordinate of the root. If the initial interval 
     * does not bracket a root, it issues a device printf warning and returns x1.
     */
    __device__ double brent_find_root(double (*f)(double, void*), void* params, double x0, double x1, double tol, int max_steps);


    /**
     * @brief Configures and executes the root-finding algorithm for a specific target function.
     *
     * This device function acts as a wrapper that sets up the initial search interval 
     * [low, high], the desired absolute tolerance, and the maximum number of iterations. 
     * It packs the parameter `u` into an `f_params` struct before delegating the 
     * actual computation to `brent_find_root`.
     *
     * @param u  The primary parameter (e.g., target value) packed into a struct and passed to `f`.
     * @param df Pointer to the derivative of the target function (unused by Brent's method, maintained for API compatibility).
     * @param f  Pointer to the target __device__ function whose root is being sought.
     * @param w  A weight or window parameter used to compute the dynamic lower and upper bounds of the search interval.
     *
     * @return The estimated x-coordinate (solution) where the target function evaluates to zero.
     */
    __device__ double find_y(double u, double (*df)(double), double (*f)(double, void *), double w);

    /**
     * @brief Samples a value 'y' from a nonthermal (kappa) distribution using rejection sampling.
     *
     * This device function constructs a mixture model based on a global kappa parameter 
     * (KAPPA_SYNCH) and the electron temperature (Thetae). It computes four statistical 
     * weights (pi_3 through pi_6) involving gamma functions, normalizes them, and selects 
     * a branch using a uniformly distributed random number. It then draws a candidate `y` 
     * using inverse transform sampling (via `find_y`), followed by an accept/reject step 
     * based on a calculated probability threshold.
     *
     * @param Thetae     Dimensionless electron temperature, used to calculate the scaling factor (w).
     * @param localState Pointer to the thread-local cuRAND state for generating uniform random numbers.
     *
     * @return A randomly sampled double-precision value 'y' conforming to the target non-thermal distribution.
     */
    __device__ double sample_y_distr_nth(double Thetae, curandState * localState);
 #endif