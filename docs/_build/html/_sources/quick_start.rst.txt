======================================================
Quick Start Guide for GPUmonty
======================================================

GPUmonty is a high-performance, `CUDA <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_-accelerated Monte Carlo radiative transfer (MCRT) code designed for the spectral modeling of accreting black holes based on `igrmonty <https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387>`_.

Prerequisites
=============

Before compiling, ensure your system has the following libraries installed and accessible:

* **CUDA Toolkit:** Required for the ``nvcc`` compiler and GPU kernels (`Install Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`_).
* **GNU Scientific Library (GSL):** Required for various mathematical and statistical routines (`GSL Home <https://www.gnu.org/software/gsl/>`_).
* **Hierarchical Data Format v5 (HDF5):** Required for reading GRMHD simulation snapshots (`HDF5 Home <https://www.hdfgroup.org/solutions/hdf5/>`_).

Environment Configuration
-------------------------

Locate the installation paths for these libraries on your system and update the corresponding variables in the ``Makefile``:

.. code-block:: makefile

   CUDA_PATH      = /usr/local/cuda
   GSL_PATH       = /usr/local
   HDF5_INCLUDE   = /usr/include/hdf5/serial
   HDF5_LIB       = /usr/lib/x86_64-linux-gnu/hdf5/serial


.. note::
   The makefile is set to automatically find the **compute capability** of your GPU.

   Compute capability refers to the CUDA architecture version of your GPU (e.g., sm_86 for Ampere), which determines which GPU instructions and optimizations are used during compilation.
   
   In case you want to do it yourself, set ```AUTO_CC ?= 0``` and look for the compute capability on `Nvidia's website <https://developer.nvidia.com/cuda/gpus>`_. 
   
After you have changed these settings, compile by typing:

.. code-block:: bash

   make -j 15

In case you want to compile for debugging, use:

.. code-block:: bash

   make BUILD_TYPE=debug


CUDA Number of Blocks Configuration
-----------------------------------

The build system includes an auto-tuning feature that detects the hardware specifications of the GPU on your current machine (specifically Device 0).

During compilation, the ``Makefile`` triggers a probe (defined in ``GetGPUBlocks.mk``) that calculates the optimal number of blocks based on the GPU's multiprocessor count and blocks-per-multiprocessor limit. This process automatically updates the ``N_BLOCKS`` definition located in:

``src/config.h``

By default, this feature is **enabled**. If you wish to manually set ``N_BLOCKS`` to a fixed value in the config file, you can disable the auto-tuner by setting the ``GPU_TUNING`` flag to 0:

.. code-block:: bash

   make BLOCK_TUNING=0

.. warning::

   If you are running on a High Performance Computing (HPC) cluster, **do not compile on the login/head node**, as these nodes often lack GPUs or possess different hardware than the compute nodes.
   To ensure the auto-tuner detects the correct GPU architecture for your run, we recommend adding the compilation step directly inside your job submission script (e.g., Slurm or PBS script).



Multi-Core Acceleration (OpenMP)
--------------------------------

GPUmonty benefits from **OpenMP** for CPU-bound tasks such as data pre-processing and grid initialization. To enable multi-threaded CPU execution:

.. code-block:: bash

   export OMP_NUM_THREADS=XX

Replace ``XX`` with the desired number of threads (recommended: number of physical CPU cores).

Configuration Parameters
------------------------

Simulation parameters are passed to the executable via a ``.par`` file. You can find a baseline configuration in ``/gpumonty/template.par``. 

To run a simulation with your custom parameters:

.. code-block:: bash

   ./gpumonty -par path/to/your_file.par

.. list-table:: Runtime Parameters
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``Ns``
     - **Superphoton Count**: The approximated total number of photon packets to be generated.
   * - ``dump``
     - **Data Path**: Relative or absolute path to the input GRMHD data file.
   * - ``spectrum``
     - **Output Name**: Filename for the output spectral data (e.g., ``sane.spec``).
   * - ``MBH``
     - **Black Hole Mass**: Mass of the central black hole in Solar Masses (:math:`M_\odot`).
   * - ``M_unit``
     - **Mass Unit Scale**: Normalization factor (in grams) to scale dimensionless GRMHD density to physical CGS units.
   * - ``tp_over_te``
     - **Proton-to-Electron Temperature Ratio**: Constant ratio (:math:`T_p/T_e`) used if a dynamic heating model is not active.
   * - ``Thetae_max``
     - **Temperature Ceiling**: Numerical cap for the dimensionless electron temperature (:math:`\Theta_e = k_B T_e / m_e c^2`).
   * - ``scattering``
     - **Boolean for Scattering**: Enable or disable scattering processes in the simulation.
   * - ``bremsstrahlung``
     - **Boolean for Bremsstrahlung**: Enable or disable Bremsstrahlung emission processes in the simulation.
   * - ``synchrotron``
     - **Boolean for Synchrotron**: Enable or disable synchrotron emission processes in the simulation.
   * - ``fit_bias``
     - **Bias Fitting**: Enable or disable the bias fitting procedure to optimize photon packet generation.
   * - ``bias_guess``
     - **Initial Bias Guess**: Initial guess for the bias factor used in the bias fitting procedure.
   * - ``ratio``
     - **Bias Ratio**: Ratio used in the bias fitting procedure to adjust the bias factor iteratively.

Analyzing the Output
====================

To facilitate data post-processing and visualization, an example Jupyter Notebook is provided in the repository.

* **Notebook Location:** ``python/example.ipynb``
* **Workflow:** This tutorial guides you through opening output files, extracting spectral arrays, and generating plots.

Spectral Data Structure
-----------------------

When analyzing the raw results in Python, please note the relationship between luminosity and the observer's viewing angle:

* **Indexing:** The luminosity array (``nuLnu``) is multi-dimensional; each index in the array corresponds directly to one of the ``theta_bins`` defined in your simulation.

GRMHD Data File for Testing
===========================

To reproduce tests using the same GRMHD input employed in the **GPUmonty paper**, download the dataset from `Prather et al. (2023) <https://iopscience.iop.org/article/10.3847/1538-4357/acc586>`_ via the Harvard Dataverse:

* **Dataset:** `Harvard Dataverse (DOI: 10.7910/DVN/XZECPF) <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XZECPF>`_

After downloading, place the GRMHD data file in the ``data/`` directory and run:

.. code-block:: bash

   ./gpumonty -par template.par

The resulting spectrum should match the expected output shown below:


.. figure:: ./figures/expected_spectrum.png
   :width: 80%
   :align: center
   :alt: Expected spectrum output for GPUmonty

   **Expected Result:** The resulting spectrum showing the :math:`\nu L_\nu` distribution across frequencies.

LICENSE
=======

``GPUmonty`` is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either **version 2** of the License, or (at your option) any later version.

See the `GNU General Public License <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>`_ for more details.