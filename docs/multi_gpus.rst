======================================================
Multi-GPU Acceleration
======================================================

GPUmonty now uses multi-GPU acceleration to distribute the computational load of tracking millions of superphotons across multiple devices. By utilizing OpenMP for host-side threading and dedicated CUDA streams for device-side execution, the code scales efficiently with the number of available GPUs on a single node.

Overview
========

The multi-GPU implementation is built around a data-parallel architecture where each GPU operates almost entirely independently until the final data reduction phase, since we assume
the superphotons batches need no communcation during the tracking and scattering phases. The key features of this architecture include:

* **Basic Idea**: Each CPU thread locks to a specific device using ``cudaSetDevice()`` and manages its own independent CUDA stream (``local_stream``).
* **Workload Partitioning**: The total pool of generated superphotons is divided among the GPUs. Each GPU processes its assigned subset, tracks geodesics, and resolves the scattering events.
* **Spectrum Merging**: After all tracking is complete, the individual emission spectra from each GPU are reduced and merged on GPU 0 before being transferred back to the host.

Execution Flow
==============

The simulation follows

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Step
     - Phase
     - Description
   * - 1
     - Initialization
     - GPU 0 computes the total number of superphotons to be spawned in the whole grid dimension.
   * - 2
     - Partitioning
     - The host computes cumulative totals across zones and determines the exact slice of photons each GPU is responsible for tracking. The division occur within each cell to reduce load imbalance. If we divided photons from different locations to different gpus, we would have some gpus tracking photons from the inner region of the grid, which are more likely to scatter and thus require more time to track, while other gpus would be tracking photons from the outer region of the grid, which are less likely to scatter and thus require less time to track. 
   * - 3
     - Parallel Tracking
     - OpenMP spawns threads. Each thread copies the global grid/primitive variables to its assigned GPU's local VRAM, initializes random number generator states, and executes the main tracking and scattering kernels via asynchronous streams.
   * - 4
     - Reduction
     - Local spectra arrays are copied peer-to-peer (``cudaMemcpyPeer``) to GPU 0, where a master accumulation kernel merges them into a single global spectrum.

Architecture Diagram
====================

The following diagram illustrates the thread execution and memory architecture during the main simulation loop:

.. code-block:: text

   Host (CPU) - OpenMP Master Thread
         │
         ├─ Calculates total superphotons
         ├─ Partitions workload via Cumulative Arrays
         │
         ▼
   #pragma omp parallel for num_threads(num_gpus)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
   CPU Thread 0                              CPU Thread 1 (up to N)
   [cudaSetDevice(0)]                        [cudaSetDevice(1)]
         │                                         │
         ├─ Allocates local VRAM                   ├─ Allocates local VRAM
         ├─ Copies Grid & Primitives               ├─ Copies Grid & Primitives
         │                                         │
         ▼                                         ▼
   GPU 0 Stream                              GPU 1 Stream
         │                                         │
         ├─ sample_photons_batch<<<>>>             ├─ sample_photons_batch<<<>>>
         ├─ track<<<>>>                            ├─ track<<<>>>
         ├─ track_scat<<<>>>                       ├─ track_scat<<<>>>
         ├─ record<<<>>>                           ├─ record<<<>>>
         │                                         │
         ▼                                         ▼
   Local Spectrum 0                          Local Spectrum 1
         │                                         │
         └───────────────────┬─────────────────────┘
                             ▼
                 cudaMemcpyPeer to GPU 0
                             │
                             ▼
               AccumulateSpectrum<<<>>> (GPU 0)
                             │
                             ▼
                    Host (Final Output)

Memory Management and Batching
==============================

The memory algorithm remains the same from the single-gpu version. If a GPU's assigned workload exceeds its safe memory limits, the host thread automatically divides the GPU's slice into smaller, manageable partitions (``batch_divisions``). The GPU will fully process one partition, including all of its scattering layers, before freeing the memory and moving on to the next. This prevents out-of-memory (OOM) errors.

.. warning::

   When modifying the core tracking loops, developers must ensure that host-side variables used inside the OpenMP region (such as dynamic bias tuning parameters or global scattering counters) are either strictly **thread-local** or protected by ``#pragma omp atomic`` directives. Unintentionally modifying shared global state from multiple CPU threads simultaneously will result in race conditions and catastrophic data.

.. note::

   Multi-GPU scaling is currently limited to a single machine/node utilizing unified memory spaces and peer-to-peer transfers. Distributed computing across multiple nodes via MPI is not supported in this pipeline and to be honest, I, personally, don't see using MPI as being a great enhancement to the current version.