#include "decs.h"

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

__device__ void generate_random_direction(double *x, double *y, double *z, curandState localState)
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
		*x = -1 + 2 * curand_uniform_double(&localState);
		*y = -1 + 2 * curand_uniform_double(&localState);
		s = (*x) * (*x) + (*y) * (*y);
	}
	while (s > 1.0);

	*z = -1 + 2 * s;              /* z uniformly distributed from -1 to 1 */
	a = 2 * sqrt (1 - s);         /* factor to adjust x,y so that x^2+y^2
									* is equal to 1-z^2 */
	*x *= a;
	*y *= a;
}

__device__ double legacy_standard_exponential(curandState localState)
{
	return -log(1 - curand_uniform_double(&localState));
}

__device__ void legacy_gauss(double* out1, double* out2, curandState localState) {
    double f, x1, x2, r2;

    do {
        x1 = 2.0 * curand_uniform_double(&localState) - 1.0;
        x2 = 2.0 * curand_uniform_double(&localState) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    f = sqrt(-2.0 * log(r2) / r2);
    *out1 = f * x1;
    *out2 = f * x2;
}
__device__ double legacy_standard_gamma(double shape, curandState localState) {
	double b, c;
	double U, V, X, Y;

	if (shape == 1.0) {
		return legacy_standard_exponential(localState);
	}
	else if (shape == 0.0) {
		return 0.0;
	} else if (shape < 1.0) {
		for (;;) {
		U = curand_uniform_double(&localState);
		V = legacy_standard_exponential(localState);
		if (U <= 1.0 - shape) {
			X = pow(U, 1. / shape);
			if (X <= V) {
			return X;
			}
		} else {
			Y = -log((1 - U) / shape);
			X = pow(1.0 - shape + shape * Y, 1. / shape);
			if (X <= (V + Y)) {
			return X;
			}
		}
		}
	} else {
		b = shape - 1. / 3.;
		c = 1. / sqrt(9 * b);
		double out1, out2;
		for (;;) {
            legacy_gauss(&out1, &out2, localState);
            X = out1;
		do {
            if(X == out2){
                legacy_gauss(&out1, &out2, localState);
                X = out1;
            }else{
                X = out2;
            }
			V = 1.0 + c * X;
		} while (V <= 0.0);

		V = V * V * V;
		U = curand_uniform_double(&localState);
		if (U < 1.0 - 0.0331 * (X * X) * (X * X))
			return (b * V);
		if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
			return (b * V);
		}
	}
}


__device__ double chi_square(int df, curandState localState)
{
	return 2.0 * legacy_standard_gamma(0.5 * df, localState);
}