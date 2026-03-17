======================================================
Continuous Integration Testing
======================================================

GPUmonty uses a GPU-accelerated Continuous Integration (CI) pipeline to automatically validate the physics of the code on every push to the ``main`` or ``feature/test-ci`` branches. Since the code requires a real NVIDIA GPU to run, the CI pipeline offloads execution to `Kaggle <https://www.kaggle.com/>`_, which provides free GPU compute.

Overview
========

The CI pipeline is built on two components working together:

* **GitHub Actions**: Detects the push, prepares the test script, sends it to Kaggle, and polls for results.
* **Kaggle Kernels**: Clones the repository, compiles the code on a real GPU, runs the physics tests, and reports pass or fail.

The pipeline is entirely one-way: Kaggle runs the code in a virtual environment and discards all build artifacts after the session ends. Nothing is written back to the repository.

Trigger Conditions
==================

The CI pipeline triggers automatically under the following conditions:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Event
     - Behaviour
   * - Push to ``main``
     - Triggers CI unless commit message contains ``[skip-ci]``
   * - Push to ``feature/test-ci``
     - Always triggers CI
   * - Push to any other branch
     - CI is skipped entirely

Skipping CI
-----------

When pushing directly to ``main``, a local Git hook will prompt you interactively:

.. code-block:: text

   You are pushing to main. Run Kaggle CI tests? [y/N]

If you answer **N**, the hook automatically appends ``[skip-ci]`` to your commit message, and the GitHub Actions workflow will skip execution. If you answer **Y**, the pipeline runs normally.

.. note::

   The Git hook requires a one-time setup after cloning the repository. See :ref:`hook-setup` below.

   Note that direct pushes to ``main`` should be rare. Commits should generally reach ``main`` through a pull request after review. Direct pushes are only appropriate for minor changes such as documentation or README updates, in which case running the full GPU test suite is
   unnecessary and can be safely skipped.

Pipeline Architecture
=====================

The following diagram summarizes the full CI flow:

.. code-block:: text

   git push (main or feature/test-ci)
         │
         ▼
   GitHub Actions triggered
         │
         ├─ Clones repository at the pushed commit SHA
         ├─ Generates a Kaggle runner script (ci_script.py)
         └─ Pushes script to Kaggle via the Kaggle CLI
                   │
                   ▼
            Kaggle GPU Kernel
                   │
                   ├─ Clones repository
                   ├─ Installs dependencies (GSL, HDF5)
                   ├─ Compiles GPUmonty with nvcc
                   └─ Runs physics test suite
                             │
                             ▼
                  GitHub Actions polls status
                             │
                   ┌─────────┴──────────┐
                 PASS                 FAIL
                   │                    │
            Workflow succeeds    Workflow fails


Test Suite
==========

The physics tests are located in the ``tests-ci/`` directory. Each test runs GPUmonty with a known configuration and compares the output spectrum against an analytical solution and looks at the relative error. The following tests are currently included:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Test
     - Emission Process
     - Tolerance
   * - ``sphere-emission-test-synchrotron``
     - Synchrotron only
     - 10%
   * - ``sphere-emission-test-bremsstrahlung``
     - Bremsstrahlung only
     - 5%
   * - ``sphere-emission-test-mixed``
     - Synchrotron + Bremsstrahlung
     - 5%

All tests use the ``sphere`` model, as described in the paper, a uniform spherical plasma with analytically known emissivity, to isolate and validate the emission physics independently of the GRMHD model.

.. warning::

   Because the CI pipeline exclusively evaluates the ``sphere`` model, any modifications made to functions, or data-handling routines outside of this specific model (e.g., changes within ``harm_model`` or ``iharm_model``) are currently **not checked** by the automated test suite. Developers must validate GRMHD-specific changes locally before pushing.

Local Git Hook Setup
====================

The pre-push hook that enables the interactive CI prompt is stored in ``.git/hooks/pre-push`` and is version-controlled with the repository. To activate it, run the following command once after cloning:

.. code-block:: bash

   git config core.hooksPath .git/hooks
   chmod +x .git/hooks/pre-push

After this one-time setup, every push to ``main`` or ``feature/test-ci`` will trigger the prompt automatically.

.. note::

   This configuration is local to your clone and is not propagated automatically to other contributors. Each developer must run the setup command independently after cloning.