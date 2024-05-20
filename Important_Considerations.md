# Some considerations

This GPU version is still incomplete. As of this version, GPU is not better than CPU at least for a low photon count. Some improvements should and must be made before the last version go out.

Known need improvements:
- GPU_track: I need to refactor this. This function is taking way to much time. I guess it had a great improvement in regards to separating the scatterings photons and resolving them later, but I can't comprove it because every version previous is biased, which means that it wasn't working fine. Mainly because the initial photon_count was wrong (I was setting it equal to photon_count, but photon count was getting decreased inside the for loop). I thought it wouldn't matter, but it did! Some ideas to refactor: I analyzed with ncu and it seems that a lot of threads are getting stalled waiting for it to complete. I guess this difference is introduced because the range of photons that the computation is heavy is actually small (Those that enter the while loop in GPU_track_super_photon), so a lot of threads just pass through the for loop and keep waiting for the other ones to arrive. My idea here is to solve all the photons that doesn't go into the loop first, then finally use all the threads to compute those photons that actually meet the criterium;

- The bias method (using max_tau_scatt) may be getting in the way of luminosity calculation and also be responsible for some inconsistencies in the plot. This issue should be addressable, but right now, nothing comes to mind. Furthermore, some of the problems I was having in previous plots was due to the problem described in the paragraph above, where not all of the photons were being analyzed.

- Memory problem with large number of photons. Normal grmonty computer one photon at a time. I'm trying to compute all the photons altogether, which gives me a problem of storing these photons. With a large number of photons, gpu just does not have enough memory. For a V100/A100, this should not be that big of a deal, unless I'm trying to compute like exorbitants amount of photons. Nonetheless, we could implement a way of knowing if the photon count will be too much: Measure the size of the photon quantity, see if it matches in Gb with the size of the GPU, if not, divide the section into parts and evaluate them. 

TODO (beside everything said above):
- Separate the functions in different files in a more concise way, like: 'Reading_Data.cu', 'Create_photons.cu', 'Track_photons.cu', 'Record_report_photons.cu'.






To profile it with ncu:
ncu -f -o report_ncu_3 time ./grmonty 500 ./data/dump019 4.e19



Latest tests with optimization:
50.000 photon_parameter:
GPU: 230 x 256 threads
real    0m23.128s
CPU:
real    0m22.710s

100.000 photon_parameter:
GPU: 230 x 256 threads
real    0m24.429s
CPU:
real    0m28.689s

500.000 photon_parameter:
GPU: 230 x 256 threads
real    0m51.311s
CPU:
real    1m22.931s

1.000.000 photon_parameter:
GPU: 230 x 256 threads
real    1m27.657s
CPU:
real    2m39.139s

1.500.000 photon_parameter:
GPU:230 x 256 threads
real    2m39.836s
CPU: 
real    3m31.231s