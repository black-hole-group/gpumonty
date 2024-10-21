# Some considerations

This GPU version is still incomplete. As of this version, GPU is not better than CPU at least for a low photon count. Some improvements should and must be made before the last version go out.

Known need improvements:
- GPU_track: I need to refactor this. This function is taking way to much time. I guess it had a great improvement in regards to separating the scatterings photons and resolving them later, but I can't comprove it because every version previous is biased, which means that it wasn't working fine. Mainly because the initial photon_count was wrong (I was setting it equal to photon_count, but photon count was getting decreased inside the for loop). I thought it wouldn't matter, but it did! Some ideas to refactor: I analyzed with ncu and it seems that a lot of threads are getting stalled waiting for it to complete. I guess this difference is introduced because the range of photons that the computation is heavy is actually small (Those that enter the while loop in GPU_track_super_photon), so a lot of threads just pass through the for loop and keep waiting for the other ones to arrive. My idea here is to solve all the photons that doesn't go into the loop first, then finally use all the threads to compute those photons that actually meet the criterium;

- The bias method (using max_tau_scatt) may be getting in the way of luminosity calculation and also be responsible for some inconsistencies in the plot. This issue should be addressable, but right now, nothing comes to mind. Furthermore, some of the problems I was having in previous plots was due to the problem described in the paragraph above, where not all of the photons were being analyzed.

- Memory problem with large number of photons. Normal grmonty computer one photon at a time. I'm trying to compute all the photons altogether, which gives me a problem of storing these photons. With a large number of photons, gpu just does not have enough memory. For a V100/A100, this should not be that big of a deal, unless I'm trying to compute like exorbitants amount of photons. Nonetheless, we could implement a way of knowing if the photon count will be too much: Measure the size of the photon quantity, see if it matches in Gb with the size of the GPU, if not, divide the section into parts and evaluate them. 


-There was a mistake in the GPU_linear_interp_F where dlK is 1/dlK, which i wasnt doing. Now it seems fixed! This mistake was responsible for generating a second bump very close to the first one
- There is a mistake in the calculate frequency function. It is calculating one frequency for every zone. Instead, it should calculate one frequency for each photon, but got to use the data for the zone.

- 

TODO (beside everything said above):
- Separate the functions in different files in a more concise way, like: 'Reading_Data.cu', 'Create_photons.cu', 'Track_photons.cu', 'Record_report_photons.cu'.


BUG FIXES:
-> Luminosity is a bit off. If I run the original code from the github repository and change the parameters in order to fit my code (mainly RMAX = 10.), luminosities are off. I gotta check if there is a parameter that I'm not changing.
-> Fix the size of the photon array





To profile it with ncu:
ncu -f -o report_ncu_3 time ./grmonty 500 ./data/dump019 4.e19



Latest tests with optimization:
using (176 x 256) threads:

50.000 photon parameter:
GPU:
N_superph_made: 1036195
N_superph_recorded: 324931
real    0m1.829s

CPU:
N_superph_made: 805973
N_superph_recorded: 338323
real    0m2.932s

150.000 photon_parameter:
GPU:
N_superph_made: 3108567
N_superph_recorded: 976862
real    0m3.659s

CPU:
N_superph_made: 2417743
N_superph_recorded: 1017090
real    0m7.283s

350.000 photon_parameter:
GPU:
N_superph_made: 7253185
N_superph_recorded: 2267559
real    0m7.174s

CPU:
N_superph_made: 5641245
N_superph_recorded: 2367496
real    0m16.140s

750.000 photon_parameter:
GPU:
N_superph_made: 15542463
N_superph_recorded: 4871364
real    0m14.372s

CPU:
N_superph_made: 12088621
N_superph_recorded: 5052572
real    0m33.416s

1.500.000 photon_parameter:
GPU:
N_superph_made: 31084992
N_superph_recorded: 9679602
real    0m27.790s

CPU:
N_superph_made: 24177075
N_superph_recorded: 10093335
real    1m8.503s

2.000.000 photon_parameter:
GPU:
N_superph_made: 41446542
N_superph_recorded: 12969870
real    0m36.645s

CPU:
N_superph_made: 32236260
N_superph_recorded: 13456522
real    1m32.065s

Time spent running the full code: 10.799421 seconds. Ntot = 150000
Time spent running the full code: 44.711290 seconds. Ntot = 750000
Time spent running the full code: 58.753125 seconds. Ntot = 1000000
Time spent running the full code: 1143.118472 seconds. Ntot = 100000000 //in the same file
Time spent running the full code: 1210.156004 seconds. Ntot = 100000000 // with dlto. it took an extra 67 seconds 1 min and 7 seconds longer
Time spent running the full code: 6116.992129 seconds. Ntot = 100000000
We got to change EPS when we are dealing with different grids. This seems to work with the low end part of the optically thick sphere

REMEMBER TO ENABLE SCATTERING ALLOCATIONS ONCE TEST IS DONE
The error in the high frequencies is related to the size of the bin