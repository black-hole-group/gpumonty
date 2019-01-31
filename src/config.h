#ifndef _CONFIG_H
#define _CONFIG_H

// Default config values
// * default kernel config is the one found to be optimal in a GTX1080 8GB
extern int BLOCK_SIZE;
extern int NUM_BLOCKS;
extern int N_CPU_THS;
#define N_GPU_THS (BLOCK_SIZE * NUM_BLOCKS)

#endif
