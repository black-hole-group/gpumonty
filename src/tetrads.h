/*
 Declaration of the functions used in tetrads.cu file
*/

#ifndef TETRADS_H
#define TETRADS_H
/**
 * @brief Constructs an orthonormal tetrad (local laboratory frame) for a moving observer.
 * 
 * A tetrad represents a local Minkowski frame $(e_{(0)}, e_{(1)}, e_{(2)}, e_{(3)})$ 
 * at a point in curved spacetime. 
 * 
 * @note This function uses the **Gram-Schmidt process** to orthogonalize a set of coordinate basis vectors against the fluid's 4-velocity.
 * 
 * @param Ucon  The contravariant 4-velocity of the fluid/observer ($u^\mu$).
 * @param trial A trial vector used to define the first spatial direction (usually radial).
 * @param Gcov  The covariant metric tensor ($g_{\mu\nu}$).
 * @param Econ  [Output] Contravariant tetrad $e^\mu_{(a)}$. First index (k) is the tetrad label, 
 * second (l) is the coordinate index.
 * @param Ecov  [Output] Covariant tetrad $e_{(a)\mu}$.
 * 
 * @return void
 */
__device__ void make_tetrad(double Ucon[NDIM], double trial[NDIM], const double Gcov[NDIM][NDIM], double Econ[NDIM][NDIM], double Ecov[NDIM][NDIM]);

/**
 * @brief Transforms a vector from the local tetrad (fluid) frame to the global coordinate frame.
 * 
 * This function performs a change of basis. Physical 
 * processes are calculated in the local orthonormal.
 * 
 * The transformation is defined by the sum:
 * \f$ K^\mu = \sum_{a=0}^{3} e^\mu_{(a)} K^{(a)} \f$
 * 
 * @param Econ     The contravariant tetrad basis vectors \f$ e^\mu_{(a)} \f$.
 * @param K_tetrad The components of the vector in the local frame \f$ K^{(a)} \f$.
 * @param K        [Output] The components of the vector in the coordinate frame \f$ K^\mu \f$.
 * 
 * @return void
 */
__device__ void tetrad_to_coordinate(const double Econ[NDIM][NDIM], const double K_tetrad[NDIM], double K[NDIM]);

/**
 * @brief Projects a global coordinate vector into the local orthonormal tetrad (fluid) frame.
 * This is the inverse operation of `tetrad_to_coordinate`. It is used to "measure" 
 * global quantities from the perspective of a local observer moving with the fluid. 
 *
 * The projection is defined by the inner product of the coordinate vector with the 
 * covariant tetrad basis:
 * \f$ K^{(a)} = e_{(a)\mu} K^\mu \f$
 * 
 * @param Ecov     The covariant tetrad basis vectors \f$ e_{(a)\mu} \f$.
 * @param K        The components of the vector in the coordinate frame \f$ K^\mu \f$.
 * @param K_tetrad [Output] The components of the vector in the local frame \f$ K^{(a)} \f$.
 * 
 * @return void
 */
__device__ void coordinate_to_tetrad(const double Ecov[NDIM][NDIM], const double K[NDIM], double K_tetrad[NDIM]);

/**
 * @brief Implements the Kronecker delta function \f$ \delta_{ij} \f$.
 *
 * @param i Row or vector index.
 * @param j Column or reference index.
 * @return 1.0 if i == j, otherwise 0.0.
 */
__device__ double delta(int i, int j);

/**
 * @brief Normalizes a contravariant 4-vector to unit length in curved spacetime.
 *
 * 
 * It follows the expression in General Relativity:
 *
 *  \f$ V^\mu_{\rm norm} = \frac{V^\mu}{\sqrt{\sum^4_{\mu = 0} \sum^4_{\nu = 0} g_{\mu\nu} V^\mu V^\nu}} \f$
 * 
 * @param vcon The contravariant vector \f$v^\mu\f$ to be normalized (modified in-place).
 * @param Gcov The covariant metric tensor \f$g_{\mu\nu}\f$ at the vector's location.
 * 
 * @return void
 */
__device__ void normalize(double *vcon, const double Gcov[NDIM][NDIM]);

/**
 * @brief Projects one vector out of another to ensure orthogonality in curved spacetime.
 * This is used during the Gram-Schmidt process. It modifies 
 * vector 'a' such that it becomes perpendicular to vector 'b'. This orthogonality is defined via the metric tensor: 
 * \f$g_{\mu\nu} A^\mu B^\nu = 0\f$.
 *
 * @param vcona The vector to be modified (\f$A^\mu\f$). After the call, it will be orthogonal to vconb.
 * @param vconb The reference vector (\f$B^\mu\f$) to project against.
 * @param Gcov  The covariant metric tensor (\f$g_{\mu\nu}\f$).
 * 
 * @return void
 */
__device__ void project_out(double *vcona, double *vconb, const double Gcov[NDIM][NDIM]);
#endif