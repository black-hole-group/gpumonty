#ifndef _ACC_PRINT_H
#define _ACC_PRINT_H

// Those functions come from the post at:
// https://parallel-computing.pro/index.php/11-openacc/53-using-cuda-device-functions-from-openacc
// (Accessed in 27/07/18)

#pragma acc routine nohost
void acc_printi (int i);
#pragma acc routine nohost
void acc_printd (double d);
#pragma acc routine nohost
void acc_printf (float f);

#endif
