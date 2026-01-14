======================================================
Documentation Guidelines
======================================================

GPUmonty is a public, open-source project. To ensure the codebase remains accessible and maintainable for the scientific community, we provide this centralized documentation as a guide for both users and contributors.

The documentation and this site is built using a hybrid pipeline: `Doxygen <https://www.doxygen.nl/>`_ is used to parse the source code and extract structural metadata into XML format, while `Sphinx <https://www.sphinx-doc.org/en/master/>`_ serves as the primary engine to render this information into a web interface. Using the `Breathe <https://www.breathe-doc.org/>`_ extension, we ensure that the API reference is always in sync with the latest version of the code.
Prerequisites

File-Based Organization and Commenting
=======================================

To maintain a clear structure, the documentation is organized by source file. Each header file (``.h``) in the repository corresponds to a tab of the documentation. This allows users to browse functionality based on the specific file they are investigating.

Function Documentation
----------------------

All function-level documentation must be associated directly with the **function declaration in the header files**. By keeping the documentation in the headers rather than the source (``.cu``) files, we hope to keep the function definition cleaner. 

When adding or updating functions, use the Javadoc-style comment block immediately above the declaration:

.. code-block:: cpp

   /**
    * @brief A concise one-line description of the function.
    * * Detailed explanation of the physical model or numerical method used.
    * * @param parameter_name Description of the input.
    * @return Description of the return value.
    */
   __host__ __device__ double example_function(double parameter_name);

Guidelines:
-----------

* **Consistency**: Ensure the ``@brief`` tag is present; it is used to generate the summary tables in the sidebar.
* **Math**: Use LaTeX syntax (``\f$ ... \f$``) for physical variables (e.g., ``\f$ \Theta_e \f$``) to ensure they render correctly on the site.
* **Scope**: Documentation should live in the header files.


Update Workflow and Repository Structure
========================================

The GPUmonty documentation synchronizes with the latest updates from the `GitHub repository <https://github.com/pedronaethe/gpumonty/>`_. When changes are pushed to the main branch, the documentation pipeline processes the source code to ensure the web portal reflects the current state of the project.

Documentation Directories Organization
--------------------------------------

The documentation environment is organized into three primary folders:

.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Directory
     - Purpose
   * - ``/``
     - **Doxyfile**: Contains the Doxyfile configuration used by Doxygen to parse the source code.
   * - ``docs/``
     - **Source Directory**: Contains the Sphinx configuration (``conf.py``), and all ``.rst`` source files.
   * - ``html/``
     - **Web Output**: Stores the generated HTML files, CSS, and JavaScript that constitute the public website.
   * - ``latex/``
     - **Print Output**: Contains the LaTeX source files used to generate the PDF version of the manual.

Website Navigation and Tabs
---------------------------

The structure of the website is determined by the organization of ``.rst`` files within the ``docs/`` folder:

* **Primary Tabs**: Every ``.rst`` file located directly in the root of the ``docs/`` directory (such as ``quickstart.rst`` or ``guidelines.rst``) corresponds to a main navigation tab or section visible from the homepage.
* **API Pages**: The technical documentation for each code module is organized within the ``docs/api/`` subdirectory. For example, the documentation for weight calculations is defined in ``docs/api/weights.rst``.

How it Works
------------

**Extraction**: Doxygen scans the header files and generates XML metadata representing the code structure.
**Translation**: The **Breathe** extension acts as a bridge, allowing Sphinx to read that XML and insert it into the ``.rst`` files found in ``docs/api/``.
**Rendering**: Sphinx converts the ``.rst`` files into the final HTML layout.

Because the documentation is built directly from the source code, any update to a comment block in a header file (``.h``) will automatically update the corresponding page in the API section upon the next site build.

Local Testing and Building
==========================

Since any changes to the documentation affect the live website, we encourage local testing before committing and pushing to the repository.

Prerequisites
-------------

You must have **Doxygen** and **Sphinx** installed on your system.

**1. Install Doxygen** (Ubuntu/Debian):

.. code-block:: bash

   sudo apt update
   sudo apt install doxygen

**2. Install Sphinx and Extensions**:

.. code-block:: bash

   pip install sphinx sphinx-autobuild breathe exhale sphinx_rtd_theme

Testing Workflow
----------------

Follow these steps to preview your changes locally:

1. **Update Doxyfile/XML**: 
   If you have modified the ``Doxyfile`` or changed comments in the code headers, run Doxygen to update the XML metadata:
   
   .. code-block:: bash
   
      doxygen Doxyfile

2. **Launch Auto-build**:
   Open a second terminal and run the following command to keep the site updated automatically as you edit ``.rst`` files:
   
   .. code-block:: bash
   
      sphinx-autobuild docs docs/_build/html

3. **Preview Changes**:
   Open your browser and navigate to:
   
   ``http://127.0.0.1:8000``

4. **Clean Build**:
   Sphinx caches files to speed up building. If you want to force a total rebuild to ensure everything is fresh, delete the build directory:
   
   .. code-block:: bash
   
      rm -rf docs/_build

Because the documentation is built directly from the source code, any update to a comment block in a header file (``.h``) will automatically update the corresponding page in the API section upon the next build.