# Some considerations

This GPU version is still incomplete. As of this version, GPU is not better than CPU at least for a low photon count. Some improvements should and must be made before the last version go out.

Some comparisons for this version:
GPU/CPU TIMES
CPU: 8 threads
GPU: 40 blocks, 256 threads (ran in a P6000).

5.000 photon_parameter:
    - GPU: 0m41.197s
    - CPU: 0m3,973s

50.000 photon_parameter:
    - GPU: 1m9,896s
    - CPU: 0m24,837s

500.000 photon_parameter:
    - GPU: 3m41,309s
    - CPU: 3m54,560s

1.000.000 photon_parameter:
    -GPU: 6m34,634s
    -CPU: 7m34,012s

2.000.000 photon_parameter:
    -GPU: 11m49,412s
    -CPU: 


Known need improvements:
- GPU_track function as a whole. This function is very slow, maybe atomic adds for scattering photons could be taken out? As well as refactoring

- Sampling right now is very slow. This is mainly because of rejection sampling. In this version of the code, I tried to implement rejection sampling in multiple threads to speed up the process and it actually did, but I think it can be done in a more proper way (I took this idea from this [paper](https://dspace.mit.edu/bitstream/handle/1721.1/132084/11222_2021_10003_ReferencePDF.pdf?sequence=1&isAllowed=y)). Right now, the main things that come to mind are: 
    - Excessive __syncthreads() calls;
    - Function  GPU_get_fluid_zone is both being called multiple times (256 times per block, which is unnecessary) and also calculating Ucov/Ucon, which is not needed for this part of the code. The unecessary calling may be causing excess function overhead. 
    - The outer for loop is number of blocks dependent, which is not a large number and may slow down the process a lot. As of right now, this is not very good, since we are looping for $10^4 - 10^5$ photons. However, doing the rejection sample using multiple threads is actually pretty good. I'm not exactly sure if all the threads in a warps are exitting the loop as soon as the first one catches something, but since it got way faster, I am assuming it does. 

- The bias method (using max_tau_scatt) may be getting in the way of luminosity calculation and also be responsible for some inconsistencies in the plot. This issue should be addressable, but right now, nothing comes to mind.

- Maybe for a huge number of photons ($ > 10^7$) we could encounter memory error for allocating arrays? This is just a supposition and to be honest, shouldn't be a problem in modern GPUs.

TODO (beside everything said above):
- Separate the functions in different files in a more concise way, like: 'Reading_Data.cu', 'Create_photons.cu', 'Track_photons.cu', 'Record_report_photons.cu'.