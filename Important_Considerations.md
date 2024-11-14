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


10^4
GRMONTY:
real	2.35
GPUMonty:
real	2.46
10^5
GRMONTY:
real	6.41
GPUMonty:
real	4.40


10^6
GRMONTY:
real 43.20
GPUMonty:
real 9.74


1e7
GRMONTY:
real 424.24
GPUmonty:
    76.24


1e8
GRMONTY:
real 4164.30
GPUmonty:
754.35
#fez 4 grmonty

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pedro/gsl/lib
==558094== Metric result:
Invocations                               Metric Name                            Metric Description         Min         Max         Avg
Device "Quadro GP100 (0)"
    Kernel: GPU_track(of_photon*, double*, double*, of_spectrum*, of_photon*, int, int)
          4                             flop_count_dp   Floating Point Operations(Double Precision)  2.1092e+10  1.1384e+12  3.2358e+11
          4                             flop_count_sp   Floating Point Operations(Single Precision)   455917968  2.4855e+10  7066167016
    Kernel: GPU_track_scat(of_photon*, double*, double*, of_spectrum*, of_photon*, int, int)
         18                             flop_count_dp   Floating Point Operations(Double Precision)           0  1.6444e+11  1.2828e+10
         18                             flop_count_sp   Floating Point Operations(Single Precision)           0  3611195045   281153543
    Kernel: GPU_generate_photons(of_geom*, double*, long, __int64*, double*)
          1                             flop_count_dp   Floating Point Operations(Double Precision)   568966672   568966672   568966672
          1                             flop_count_sp   Floating Point Operations(Single Precision)    18239522    18239522    18239522
    Kernel: GPU_sample_photons_batch(of_photon*, of_geom*, double*, __int64*, double*, int, __int64, __int64*)
          4                             flop_count_dp   Floating Point Operations(Double Precision)  1.9669e+10  2.9225e+10  2.2621e+10
          4                             flop_count_sp   Floating Point Operations(Single Precision)   747165432  1002162858   825891245
root@terminator:/home/pedro/gpumonty# /usr/local/cuda-12.4/bin/nvprof --metrics flop_dp_efficiency ./gpumonty 1e7 ./data/MAD_0.9_new.bin MADnvprofmake

metrics used:
flop_dp_efficiency
sm_efficiency


data:
96.22% generate
99.78%

use make BUILD_TYPE=debug for debugging and make for release

