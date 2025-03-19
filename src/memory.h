/*
Declaration of the memory.cu functions
*/

#ifndef MEMORY_H
#define MEMORY_H
__host__ void transferGlobalVariables();
__host__ int setMaxBlocks();
__host__ void cummulativePhotonsPerZone(unsigned long long * generated_photons_arr, unsigned long long * d_index_to_ijk);
__host__ unsigned long long photonsPerBatch(unsigned long long tot_nph, int * batch_divisions);
#endif