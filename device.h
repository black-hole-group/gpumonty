#ifndef DEVICE_H 
#define DEVICE_H

/*
  Definitions needed for CUDA functions
  ======================================
*/




void launchKernel(double *p, simvars sim, allunits units, misc setup, double *pharr, int nph);

#endif