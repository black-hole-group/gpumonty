#include "decs.h"
#include "curand.h"
__device__ void GPU_init_monty_rand(int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &my_curand_state[tid]);
}




// __device__ double chi_square(int df)
// {
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	double sum = 0.0f;
// 	for (int i = 0; i < df; ++i)
// 	{
// 		double normal_variate = curand_normal(&my_curand_state[tid]);
// 		sum += normal_variate * normal_variate;
// 	}
// 	return sum;
// }

__device__ void generate_random_direction(double *x, double *y, double *z, curandState * localState)
{
	double s, a;

	/* This is a variant of the algorithm for computing a random point
	* on the unit sphere; the algorithm is suggested in Knuth, v2,
	* 3rd ed, p136; and attributed to Robert E Knop, CACM, 13 (1970),
	* 326.
	*/

	/* Begin with the polar method for getting x,y inside a unit circle
	*/
	do
	{
		*x = -1 + 2 * curand_uniform_double(localState);
		*y = -1 + 2 * curand_uniform_double(localState);
		s = (*x) * (*x) + (*y) * (*y);
	}
	while (s > 1.0);

	*z = -1 + 2 * s;              /* z uniformly distributed from -1 to 1 */
	a = 2 * sqrt (1 - s);         /* factor to adjust x,y so that x^2+y^2
									* is equal to 1-z^2 */
	*x *= a;
	*y *= a;
}

__device__ double legacy_standard_exponential(curandState *  localState)
{
	return -log(1. - curand_uniform_double(localState));
}

__device__ void legacy_gauss(double* out1, double* out2, curandState * localState) {
    double x1, x2, r2;

    do {
        x1 = 2.0 * curand_uniform_double(localState) - 1.0;
        x2 = 2.0 * curand_uniform_double(localState) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    // Reuse r2 variable to store the scaling factor
    r2 = sqrt(-2.0 * log(r2) / r2);  // r2 now contains 'f'
    *out1 = r2 * x1;
    *out2 = r2 * x2;
}

__device__ double legacy_standard_gamma(double shape, curandState * localState) {
	if (shape == 1.0) {
		return legacy_standard_exponential(localState);
	}
	else if (shape == 0.0) {
		return 0.0;
	} 
	else if (shape < 1.0) {
		// Small shape parameter case - scope variables to loop
		double inv_shape = 1.0 / shape;
		double one_minus_shape = 1.0 - shape;
		
		for (;;) {
			double U = curand_uniform_double(localState);
			double V = legacy_standard_exponential(localState);
			
			if (U <= one_minus_shape) {
				double X = pow(U, inv_shape);
				if (X <= V) {
					return X;
				}
			} else {
				double Y = -log((1 - U) / shape);
				double X = pow(one_minus_shape + shape * Y, inv_shape);
				if (X <= (V + Y)) {
					return X;
				}
			}
		}
	} 
	else {
		// Large shape parameter case
		double b = shape - 1. / 3.;
		double c = 1. / sqrt(9 * b);
		
		double out1, out2;
		legacy_gauss(&out1, &out2, localState);
		double X = out1;
		bool use_out2_next = true;
		
		for (;;) {
			double V;
			do {
				if (use_out2_next && X != out2) {
					X = out2;
					use_out2_next = false;
				} else {
					legacy_gauss(&out1, &out2, localState);
					X = out1;
					use_out2_next = true;
				}
				V = 1.0 + c * X;
			} while (V <= 0.0);

			V = V * V * V;
			double U = curand_uniform_double(localState);
			double X_sq = X * X;
			
			if (U < 1.0 - 0.0331 * X_sq * X_sq) {
				return (b * V);
			}
			if (log(U) < 0.5 * X_sq + b * (1. - V + log(V))) {
				return (b * V);
			}
		}
	}
}


__device__ double chi_square(int df, curandState * localState)
{
	return 2.0 * legacy_standard_gamma(0.5 * df, localState);
}