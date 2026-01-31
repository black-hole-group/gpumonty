# GPUmonty: A GPU-Accelerated Relativistic Monte Carlo Code

GPUmonty is a GPU-accelerated Monte Carlo radiative transfer code for simulating spectra from accreting black holes based on NVIDIA CUDA. It traces photon packets ("superphotons") through accretion flows around black holes and models: (i) synchrotron emission from hot electrons, (ii) photon propagation along geodesics in curved spacetime (Kerr metric), (iii) Compton scattering (Thomson and Klein-Nishina regimes), (iv) spectral synthesis at different viewing angles.

**Key features:**

  - Over 10x faster than igrmonty
  - Auto-tuning for optimal GPU performance
  - Interfaces with multiple GRMHD codes (iHARM, HARM, H-AMR)
  - HDF5 input/output for simulation data

  **Use cases:**

  - Simulating spectra from AGN and X-ray binaries
  - Inferring black hole properties from electromagnetic observations
  - Testing accretion models against multi-wavelength data

**Technology stack:**

  - CUDA C++ for GPU-accelerated parallel photon tracking
  - C / OpenMP for host-side computation
  - Python for post-processing and visualization
  - Dependencies: CUDA Toolkit, GSL, HDF5, OpenMP, 

GPUmonty is based on [igrmonty](https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387). Please refer to the [documentation webpage](https://black-hole-group.github.io/gpumonty/) for more details.

## QUICKSTART

Before proceeding, make sure you have a NVIDIA GPU with the required drivers, CUDA toolkit, HDF5 library and GSL installed. 

(1) Compile (replace the number below with the number of CPU cores available): 

    cd gpumonty
    make -j 24

(2) Download the [test data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZECPF) from a GRMHD simulation:

    curl -L -o './data/dump_SANE.h5' 'https://dataverse.harvard.edu/api/access/datafile/12137142'

(3) Run gpumonty:

    ./gpumonty -par template.par

You should now have a spectrum data file in `output/sane_iharm.spec`.

(4) Visualize the spectrum. You will need python with numpy, matplotlib and astropy libraries:

    python python/example.py

If all goes well, you should now have a image in `output/example.png` with the spectrum emitted by a hot SANE RIAF. If not, keep reading.


## INSTALLATION INSTRUCTIONS

###  Prerequisites

Before compiling, ensure your system has the following libraries installed and accessible:

* **[CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/):** Required for the `nvcc` compiler and GPU kernels.
* **[GNU scientific library (GSL)](https://www.gnu.org/software/gsl/):** Required for various mathematical and statistical routines.
* **[Hierarchical Data Format v5 (HDF5)](https://www.hdfgroup.org/solutions/hdf5/):** Required for reading GRMHD simulation snapshots.

### Path to libraries

Locate the install paths for these libraries on your system and update the corresponding variables in the `Makefile`:

```makefile
CUDA_PATH     = /usr/local/cuda
GSL_PATH      = /usr/local
HDF5_INCLUDE  = /usr/include/hdf5/serial
HDF5_LIB      = /usr/lib/x86_64-linux-gnu/hdf5/serial
```

### Compute capability

The makefile is set to automatically find the compute capability of your GPU. Compute capability refers to the CUDA architecture version of your GPU (e.g., sm_86 for Ampere), which determines which GPU instructions and optimizations are used during compilation.

In case you want to do it yourself, set ```AUTO_CC ?= 0``` and look for your GPU’s compute capability on [NVIDIA’s website](https://developer.nvidia.com/cuda/gpus).

### Multi-Core Acceleration (OpenMP)

GPUmonty benefits from OpenMP for CPU-bound tasks such as data pre-processing and grid initialization. To enable multi-threaded CPU execution, add the following line to your `.bashrc` file:

    export OMP_NUM_THREADS=XX

Replace `XX` with the desired number of threads. It is recommended to set this value equal to the number of physical cores on your CPU.

### Compile

After you have configured the things above, compile with 

    make -j 15

where the number refers to the desired number of CPU cores to use. In case you want to compile it for debug: 

    make BUILD_TYPE=debug

### CUDA Number of Blocks Configuration

The build system includes an auto-tuning feature that detects the hardware specifications of your GPU (specifically Device 0).

During compilation, the `Makefile` triggers a probe (defined in `GetGPUBlocks.mk`) that calculates the optimal number of blocks based on the GPU's multiprocessor count and blocks-per-multiprocessor limit. This process automatically updates the `N_BLOCKS` definition located in `src/config.h`. By default, this feature is enabled. If you wish to manually set `N_BLOCKS` to a fixed value in the config file, you can disable the auto-tuner by setting the `GPU_TUNING` flag to 0:

```bash
make GPU_TUNING=0
```

---
**WARNING**  
If you are running on a HPC cluster, **do not compile on the login/head node**, as these nodes often lack GPUs or possess different hardware than the compute nodes. To ensure the auto-tuner detects the correct GPU architecture for your run, we recommend adding the compilation step directly inside your job submission script (e.g., Slurm or PBS script).

---


### Simulation Setup

Simulation parameters are passed via a `.par` file. You can find a baseline configuration in `template.par`. 

To run a simulation with custom parameters:

```
./gpumonty -par path/to/your_parameter_file.par
```

The following runtime parameters are supported:

| Parameter | Description |
| :--- | :--- |
| `Ns` | **Superphoton Count**: The approximated total number of photon packets to be generated. Higher values improve the signal-to-noise ratio in the resulting spectrum. |
| `dump` | **Data Path**: The relative or absolute path to the input GRMHD data file. |
| `spectrum` | **Output Name**: The filename for the output spectral data (e.g., `sane.spec`). |
| `MBH` | **Black Hole Mass**: Mass of the central black hole in Solar Masses ($M_\odot$). |
| `M_unit` | **Mass Unit Scale**: The normalization factor (in grams) used to scale dimensionless GRMHD density to physical CGS units. |
| `tp_over_te` | **Proton-to-Electron Temperature Ratio**: A constant ratio ($T_p/T_e$) used if a dynamic heating model is not active. |
| `Thetae_max` | **Temperature Ceiling**: A numerical cap for the dimensionless electron temperature ($\Theta_e = k_B T_e / m_e c^2$). |
| `scattering` | **Scattering boolean**: Enable or disable scattering processes in the simulation.|

## ANALYSIS

To facilitate data post-processing and visualization, [an example Jupyter Notebook is provided in the repository at `python/example.ipynb`](./python/example.ipynb). It contains a tutorial of how to process output files, extract spectra and generating plots.

When analyzing the raw results in Python, please note the relationship between luminosity and the observer's viewing angle: The luminosity array `nuLnu` is multi-dimensional; each index in the array corresponds directly to one of the `theta_bins` defined in your simulation.

## GRMHD DUMP FILE FOR TESTING

To reproduce the tests using the same GRMHD data used in our paper, download this [dump file] (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZECPF). This can be done with the command:

    curl -L -o './data/dump_SANE.h5' 'https://dataverse.harvard.edu/api/access/datafile/12137142'

This dump file corresponds to a snapshot from a SANE RIAF simulation around a black hole with $a_*=0.94$. After downloading the dump file, place it in `data/` directory (the command above already does this for you) and run 

    ./gpumonty -par template.par

## LICENSE

`gpumonty` is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either **version 2** of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the `LICENSE` file or the [GNU General Public License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) for more details.


