Home
========

**GPUmonty** is a public, GPUmonty is a high-performance, `CUDA <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_-accelerated Monte Carlo radiative transfer (MCRT) code designed for the spectral modeling of accreting black holes based on `igrmonty <https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387>`_.-accelerated relativistic Monte Carlo radiative transfer (MCRT) code designed for the multi-wavelength spectral modeling of accreting black holes. 

The code is an evolution of the `grmonty` and `igrmonty` frameworks, assigning the most computationally expensive stages, such as superphoton generation, sampling, geodesic tracking, and scattering, directly to the GPU.

Theoretical Details
-------------------

For a comprehensive discussion of the governing equations, relativistic physics, and validation benchmarks, please refer to the GPUmonty paper:

* **Motta, P. N., Nemmen, R., & Joshi, A. V. (2025).** `GPUmonty: A public GPU accelerated relativistic Monte Carlo radiative transfer code. <https://www.google.com>`_

Public Availability
-------------------

The source code is open-source and available under the GNU General Public License v2 at:

* **GitHub Repository:** `https://github.com/pedronaethe/gpumonty <https://github.com/pedronaethe/gpumonty>`_

Getting Started
---------------

For a quick tutorial, system requirements, and build instructions, please refer to the **README** file in the GitHub repository. It contains the necessary steps to compile the code and run initial simulations.

Technical API Reference
-----------------------

This documentation site is provided as a reference for the **function specifics** only. It details the internal logic of the CUDA kernels, functions, and memory management structures.

* Use the sidebar to explore the specific modules.

**Developed by**: Pedro Naethe Motta

**Currently maintained by**: Pedro Naethe Motta

.. toctree::
   :maxdepth: 2
   :hidden:

   documentation