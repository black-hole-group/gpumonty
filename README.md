# GPUMONTY: A GPU-Accelerated Relativistic Monte Carlo Code

GPUMONTY is a Monte Carlo code designed for calculating the emergent spectrum from a model using a Monte Carlo technique. It is particularly suited for studying plasmas near black holes described by Kerr-Schild coordinates, radiating through thermal synchrotron and inverse Compton scattering. The code is based on the work presented in Dolence et al. 2009 Astrophysical Journal Supplement.

## Running the code
The code is written in C and parallelized using Nvidia's Graphical Processing Unit (GPU) language [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and is configured to use input files from the HARM ([Gammie et al. 2003](https://arxiv.org/abs/astro-ph/0301509)) and H-AMR ([Liska et al. 2019](https://arxiv.org/abs/1912.10192)) codes. 

- Before compiling: 

1) Code utilizes the [GNU scientific library(GSL)](https://www.gnu.org/software/gsl/) in the C portion of the code. Locate the gsl and cuda paths in your computer (if necessary) and modify the variables ```CUDA_PATH``` and ```GSL_PATH```  in the makefile.
2) You need to identify the compute capability of the GPU you are using. This can be found in the [NVIDIA website](https://developer.nvidia.com/cuda-gpus). Then, you got to properly change the number in the ```ARCH```, ```CODE``` and ```CODE_LTO```.
3) You also need to identify the model you want to use for your data. In the ```MODEL_DIR``` variable of the makefile, change its path to the location of the model you want to use. 

After you have changed all these things, compile with by typing ```make```. In case you want to compile it for debug, compile by typing ```make BUILD_TYPE=debug```.

- Code parameters
  
The file called ```config.h``` holds many of code parameters by means of pre-compiled variables, adjust the variables as you need. ```N_BLOCKS``` will set the number of blocks that will compose the GPU grid and ```N_THREADS``` is the number of threads per block. You may need to change parameters.
In case you are running H-AMR (```#define HAMR (1)```) or SPHERE_TEST (```#define SPHERE_TEST (1)```), enable the proper switches in the ```config.h``` file as well as modify the model in the makefile.

- Run the code

The command follows a simple structure of the arguments (```N_s```, ```path_to_data```, ```Name of the output file```). An example is given by
```
./gpumonty 50000 ./data/dump019 gpumonty_spec.spec
```

-Analyze the output

Inside the python files, we have a notebook to guide you on how to open, extract and plot the spectrum in python. Each index in the nuLnu array is related to one of the theta_bins.

## Possible errors

- Can't open file/Invalid: Either the data path is not valid or the function that reads the data is not reading the data in the right order.

- "Not all the photons created in scatterings have been evaluated": This problem usually happens if the number of scattered photons is outstandly large and corrupt memory location in scattering photon array. 

- Invalid memory location in GPU_track kernel. This happens when too many photons are scattered, therefore the gpu does not have enough memory to account for all of the scattered photons and ends up crashing. This is most likely caused by the bias function, which depending on how it is written, it can develop a cascade effect where the scattering grows exponentially. This is brieffly discussed in [Dolence et al. 2009](https://arxiv.org/pdf/0909.0708). Remember that you can modify the size of the memory allocation for scattering photons by changing ```#define SCATTERINGS_PER_PHOTON``` in config.h file. This defines the size of the scattering memory allocation by saying that total_number_of_scatterings = SCATTERINGS_PER_PHOTON * number_of_generated_photons.

# LICENSE 

`gpumonty` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `LICENSE` file or the [GNU General Public License](http://www.gnu.org/licenses/) for more details.


