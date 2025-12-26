# GPUmonty: A GPU-Accelerated Relativistic Monte Carlo Code

GPUmonty is a high-performance, [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)-accelerated Monte Carlo radiative transfer (MCRT) code designed for the spectral modeling of accreting black holes based on [igrmonty](https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387).

Please refer to the [documentation webpage](https://pedronaethe.github.io/gpumonty/) for more details on the functions.

## 1. Prerequisites
Before compiling, ensure your system has the following libraries installed and accessible:

* **[CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/):** Required for the `nvcc` compiler and GPU kernels.
* **[GNU scientific library (GSL)](https://www.gnu.org/software/gsl/):** Required for various mathematical and statistical routines.
* **[Hierarchical Data Format v5 (HDF5)](https://www.hdfgroup.org/solutions/hdf5/):** Required for reading GRMHD simulation snapshots.

### Environment Configuration
Locate the installation paths for these libraries on your system and update the corresponding variables in the `Makefile`:

```makefile
CUDA_PATH     = /usr/local/cuda
GSL_PATH      = /usr/local
HDF5_INCLUDE  = /usr/include/hdf5/serial
HDF5_LIB      = /usr/lib/x86_64-linux-gnu/hdf5/serial
```
After you have changed all these things, compile with by typing ```make -j 15```. In case you want to compile it for debug, compile by typing ```make BUILD_TYPE=debug```.

### Multi-Core Acceleration (OpenMP)
GPUmonty benefits from **OpenMP** for CPU-bound tasks such as data pre-processing and grid initialization. To enable multi-threaded CPU execution:

```export OMP_NUM_THREADS=XX```

Replace ```XX``` with the desired number of threads. It is recommended to set this value equal to the number of physical cores on your CPU.

### Configuration Parameters

Simulation parameters are passed to the executable via a `.par` file. You can find a baseline configuration in `/gpumonty/template.par`. 

To run a simulation with your custom parameters:
```
./gpumonty -par path/to/your_file.par
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


## 2. Analyzing the Output

To facilitate data post-processing and visualization, an example Jupyter Notebook is provided in the repository.

* **Notebook Location:** `python/example.ipynb`
* **Workflow:** This tutorial guides you through the process of opening output files, extracting spectral arrays, and generating plots.

### Spectral Data Structure
When analyzing the raw results in Python, please note the relationship between luminosity and the observer's viewing angle:
* **Indexing:** The luminosity array (`nuLnu`) is multi-dimensional; each index in the array corresponds directly to one of the `theta_bins` defined in your simulation.

## 3. GRMHD data file for testing

To reproduce the tests using the same GRMHD input employed in the **GPUmonty paper**, download the GRMHD dataset from [Prather et al. (2023)](https://iopscience.iop.org/article/10.3847/1538-4357/acc586) via the Harvard Dataverse:

- Dataset: [Harvard Dataverse (DOI: 10.7910/DVN/XZECPF)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZECPF)

After downloading, place the GRMHD data file in the `data/` directory and run the ./gpumonty -par template.par

# LICENSE

`gpumonty` is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either **version 2** of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the `LICENSE` file or the [GNU General Public License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) for more details.


