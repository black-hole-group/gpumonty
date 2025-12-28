Home
========

**GPUmonty** is a public, GPUmonty is a high-performance, `CUDA <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_-accelerated Monte Carlo radiative transfer (MCRT) code designed for the spectral modeling of accreting black holes based on `igrmonty <https://iopscience.iop.org/article/10.1088/0067-0049/184/2/387>`_.

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

For a quick tutorial, system requirements, and build instructions, please refer to the **README** file in the GitHub repository or the `Quick Start <quick_start.html>`_ documentation section. 

Technical API Reference
-----------------------

* Use the sidebar to explore the specific modules.


We welcome comments, questions, and contributions! Please feel free to open issues or pull requests on the GitHub repository.
email for contact: pedronaethemotta@usp.br

**Developed by**: Pedro Naethe Motta

**Currently maintained by**: Pedro Naethe Motta

.. toctree::
   :maxdepth: 2
   :hidden:

   documentation

.. toctree::
   :maxdepth: 2
   :hidden:

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:

   docs_guidelines
