/*
Declaration of the functions used in weights.cu file
*/

#ifndef WEIGHTS_H
#define WEIGHTS_H
void init_weight_table(void);
void init_weight_table_blackbody(void);
__host__ void init_nint_table(void);
__device__ double GPU_linear_interp_weight(const double nu);
#endif
