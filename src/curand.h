/*
Declarations of the functions in curand.cu file
*/

#ifndef CURAND_H
#define CURAND_H
/**  @brief Initializes the cuRAND random number generator state for each GPU thread. 
 * 
 * @param seed The value used to randomize the generator's starting state.
 * 
 * @return void
 */
__device__ void init_monty_rand(int seed);

/**
 * @brief Generates a random 3D unit vector uniformly distributed on the sphere.
 *
 * Uses Marsaglia’s method (also attributed to Knop; CACM 13 (1970), 326) to
 * generate an isotropic direction by sampling points uniformly inside the
 * unit disk and mapping them to the surface of the unit sphere.
 *
 *
 * @see [Algorithm 381: random vectors uniform in solid angle]
 *      (https://dl.acm.org/doi/10.1145/362349.362377)
 * 
 * @param x Pointer to the x-component of the resulting unit vector.
 * @param y Pointer to the y-component of the resulting unit vector.
 * @param z Pointer to the z-component of the resulting unit vector.
 * @param localState Pointer to the thread-local cuRAND state.
 */
__device__ void generate_random_direction(double *x, double *y, double *z, curandState * localState);

/**
 * @brief Samples a Chi-square distributed random variable for a given degree of freedom.
 * Uses the transformation property where a Chi-square distribution with \f$k\f$ degrees of freedom 
 * is equivalent to a Gamma distribution scaled by 2: \f$\chi^2(k) \sim 2\Gamma(k/2, 1)\f$.
 * χ
 * @see [Wikipedia article about χ-Γ relation](https://en.wikipedia.org/wiki/Chi-squared_distribution#Relation_to_other_distributions)
 * 
 * @param df Degrees of freedom (\f$k\f$).
 * @param localState Pointer to the thread-local cuRAND state.
 * 
 * @return A random value sampled from the \f$\chi^2\f$ distribution.
 */
__device__ double chi_square(int df, curandState * localState);

/**
 * @brief Samples a random variable from the standard exponential distribution (\f$\lambda = 1\f$).
 * Implements the Inverse Transform Sampling method by transforming a uniform \f$(0,1]\f$ 
 * random variable into an exponential distribution.
 * @see [Check wikipedia article example](https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples)
 * @param localState Pointer to the thread-local cuRAND state.
 * @return A random value sampled from the exponential distribution.
 */
__device__ double legacy_standard_exponential(curandState * localState);


/**
 * @brief Generates two independent standard normal variables (\f$\mu=0, \sigma=1\f$) using the Marsaglia polar method.
 * This is a rejection-based variant of the Box-Muller transform that avoids trigonometric functions 
 * by sampling coordinates within a unit circle.
 * 
 * @see [Marsaglia_polar_method](https://en.wikipedia.org/wiki/Marsaglia_polar_method)
 * 
 * @param out1,out2 Pointers to store the two independent Gaussian results.
 * @param localState Pointer to the thread-local cuRAND state.
 */
__device__ void legacy_gauss(double* out1, double* out2, curandState * localState);

/**
 * @brief Samples a standard Gamma distribution (\f$\Gamma(\alpha, 1)\f$) using Marsaglia-Tsang (for \f$ \alpha \geq 1 \f$) and an acceptance–rejection
 * transformation to reduce the problem to the shape ≥ 1 case (for \f$ \alpha < 1 \f$).
 * 
 * @param shape Shape parameter (\f$\alpha\f$) of the distribution.
 * @param localState Pointer to the thread-local cuRAND state.
 * @return A random value sampled from the Gamma distribution.
 */
__device__ double legacy_standard_gamma(double shape, curandState * localState);
#endif