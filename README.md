# GRMONTY: A Relativistic Monte Carlo Code

GRMONTY is a Monte Carlo code designed for calculating the emergent spectrum from a model using a Monte Carlo technique. It is particularly suited for studying plasmas near black holes described by Kerr-Schild coordinates, radiating through thermal synchrotron and inverse Compton scattering. The code is based on the work presented in Dolence et al. 2009 Astrophysical Journal Supplement.

## Getting Started
The code is written in C and parallelized using Nvidia's Graphical Processing Unit (GPU) language [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and is configured to use input files from the HARM ([Gammie et al. 2003](https://arxiv.org/abs/astro-ph/0301509)) /H-AMR ([Liska et al. 2019](https://arxiv.org/abs/1912.10192)) codes. 

## H-AMR Branch

This branch of GRMONTY has been modified to work with data from the H-AMR code ([Liska et al. 2019](https://arxiv.org/abs/1912.10192)). To use H-AMR data, a notebook is provided to convert H-AMR's dump files into a binary file with the components ordered appropriately for GRMONTY.

### H-AMR Data

We utilize a notebook to convert H-AMR's dump files into a binary file with the correct component order for GRMONTY. An example file, `HAMR_GRMONTY_DUMP323.bin`, is provided for a 2D simulation with dimensions $(256 \times 256)$ in $r - \theta$ dimensions. We do not provide the notebook for this conversion yet. To run the code to read H-AMR data, switch on the corresponding flags in `decs.h`: 

``` #define HAMR (0)```
   
If you want to run 3D data, please also activate the switch:

``` #define HAMR3D (1)```
   
Note that both switches must be activated for H-AMR 3D data.

### Changes in the Code
Several modifications have been made to handle H-AMR data. Changes include adjustments to the conversion functions for spatial coordinates, differences in the correlation formula, and modifications in handling $x_1, x_2,$ and $x_3$ coordinates. Therefore, we use different functions to calculate:

* Gcov $(g_{\mu\nu})$ and Gcon $(g^{\mu \nu})$ components are calculated in functions ```gcov_func_hamr``` and ```gcon_func_hamr```
* $x_1, x_2$ and $x_3$ coordinates based on the cell indexes in function ```coord_hamr```
All the declarations are in harm_model.h file.

The pointers ```p``` and ```geom```  used to be declared as nested arrays but we changed them by flattening the 3-dimensional spatial indexes into a 1-dimensional array with a 3D index, like this:
```p[NPRIM][i][j][k] -> p[NPRIM][SPATIAL_INDEX3D(i,j,k)]``` and also ```geom[i][j] -> geom[SPATIAL_INDEX2D(i,j)]```.

The M_unit is now set in code instead of an argument to main. You should change it in config.h precompiled variable ```#define M_UNIT```.

## Setting GPU resources
Number of blocks and number of threads per block can be set in ```gpu_header.h```, basically that's where all the functions used in GPU_grmonty.cu are are declared. Before compiling the code, check your gpu's compute capability in order to compile it right. You should modify -arch=compute_xx and -code=sm_xx in ```makefile```.

## Run the Code
In order to compile the code, you should have cudatoolkit, gcc compiler and GNU Scientific Library (GSL) installed. If necessary, change your ```CUDA_PATH``` and GSL ```include``` and ```lib``` folder path in makefile. Compile the code by using ```make```. I don't think we are using openmp right now in the code, so there isn't a need to set the number of CPU threads. After the compilation, there should be an executable called gpumonty. To run this, use command line below

```
./grmonty 50000 ./data/dump019 gpumonty_spec
```
The first argument is regarding the $N_s$ parameter as described by the original [paper](https://arxiv.org/abs/0909.0708). The second is the directory of the data file. The last argument is the name of the output file.

# LICENSE 

`gpumonty` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `LICENSE` file or the [GNU General Public License](http://www.gnu.org/licenses/) for more details.
