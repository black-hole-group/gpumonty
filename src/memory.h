/*
Declaration of the memory.cu functions
*/

#ifndef MEMORY_H
#define MEMORY_H
__host__ void transferGlobalVariables();
__host__ int setMaxBlocks();
__host__ void cummulativePhotonsPerZone(unsigned long long * generated_photons_arr, unsigned long long * d_index_to_ijk);
__host__ unsigned long long photonsPerBatch(unsigned long long tot_nph, int * batch_divisions);
__host__ void allocatePhotonData(struct of_photonSOA *ph, unsigned long long size);
__host__ void freePhotonData(struct of_photonSOA * ph);
__host__ void createTableTextureObj(cudaTextureObject_t * texObj, double table[][NT + 1], const int width, const int height, cudaArray_t * cuArray);
__host__ void createdPTextureObj(cudaTextureObject_t * texObj, double * dP, cudaArray_t * cuArray);
__host__ void create1DTextureObj(cudaTextureObject_t * texObj, double * ptr, cudaArray_t * cudaArray);
#endif