#ifndef KERNEL_H
#define KERNEL_H

/*
  Definitions needed for CUDA functions
  ======================================
*/




void launchKernel(double *p, simvars sim, allunits units, settings setup, double *pharr, int nph);

#endif