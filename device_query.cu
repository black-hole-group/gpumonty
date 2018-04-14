#include "decs.h"    
#include "harm_model.h"

/*
Queries GPU for the amount of free memory.
Returns max number of photons that GPU can hold at once.
If number of photons desired is smaller than nmaxgpu, then
returns total number of photons desired instead.

in: size of HARM arrays along each dimension
out: max number of superphotons GPU can hold at once
 
The GPU must hold the following n1xn2xn3 arrays in its 
global memory during processing: 3*B, rho, T, 4*v
 */
int get_max_photons(int n1, int n2, int n3) {
    cudaDeviceProp prop;
    size_t free, total;
    
    // assumes one GPU
    cudaGetDeviceProperties(&prop, 0); 
    cudaMemGetInfo(&free, &total);

    float memtotal=(float)prop.totalGlobalMem; // bytes
    float memfree=(float)free; // bytes

    // total size of HARM arrays in bytes, 9+1 arrays
    // adds one extra array to leave room and avoid filling up the memory
    float sizeHARM=10.0*n1*n2*n3;

    // estimates max number of photons GPU can process at once
    // based on size of of_photon struct
    int nmax=(int)((memfree-sizeHARM)/(25.*4.));

    printf("GPU model: %s\n", prop.name);
    printf("Total memory = %f (GB)\n", memtotal/1e9);
    printf("Free memory = %f (GB)\n", memfree/1e9);
    printf("Max superphotons processed each pass = %d\n", nmax);    

    return nmax;
}





/*
Transfer HARM arrays to GPU.

TODO:
- [ ] modify p to include extra spatial dimension
*/
void mallocDevice(int nprim, int n1, int n2, int n3) {
    cudaMalloc(&d_p, nprim*n1*n2*n3*sizeof(float));
    cudaMemcpy(d_p, p, nprim*n1*n2*n3*sizeof(float), cudaMemcpyHostToDevice);
}




/*

Deallocates device variables

*/
void freeDevice() {
    cudaFree(d_p);
}