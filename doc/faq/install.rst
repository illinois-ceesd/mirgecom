Installation questions
======================

What is conda/anaconda/conda-forge/miniconda/miniforge?
-------------------------------------------------------

To install binary packages (including python and pocl), we use the `conda
<https://docs.conda.io/en/latest/>`__ package management system by default.
``conda`` was developed as part of the `anaconda <https://anaconda.org/>`__
Python distribution, but is now a separate project. ``conda`` can use
different channels to install packages from. By default, we use the
`conda-forge <https://conda-forge.org/>`__ channel. `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`__ is a minimal distribution
of packages (including Python and conda) to bootstrap the installation of the
other packages. `miniforge <https://github.com/conda-forge/miniforge>`__ is a
version of miniconda that uses the conda-forge channel by default.

Note that using conda is not strictly required for running mirgecom, but
simplifies the installation considerably.

Why are we installing (mostly) binary packages?
-----------------------------------------------

Many of the packages required by mirgecom are installed as binary packages by default, via conda.
These packages (python, pocl, pyopencl, and islpy, among others) are difficult or time-consuming
to install from source, and source builds are more easily impacted by other software present on the system.

How can I install pocl from source?
-----------------------------------

In some cases, it can be helpful to install certain packages from source, for debugging or to install
a git version. Most packages are straightforward to install from source. For pocl, you can follow this
`installation script <https://gist.github.com/matthiasdiener/838ccbdb5d8f4e4917b58fe3da811777>`__.

How can I use system OpenCL implementations, such as Nvidia CL or Apple CL?
---------------------------------------------------------------------------

To use OpenCL drivers other than pocl, you can access the ones installed on the
system by installing the following conda packages::

   $ conda install ocl-icd-system            # Linux
   $ conda install ocl_icd_wrapper_apple     # MacOS

.. _Pyopencl source installation:

How can I build pyopencl from source?
-------------------------------------

Pyopencl needs to be built against an OpenCL ICD loader (`libOpenCL.so`) which in turn loads the runtimes
(ICDs: installable client drivers). You therefore need to make sure
that the build process picks up the right one. This can be challenging especially on MacOS, since Apple provides its own CL runtime which does not easily compile against pyopencl.

You can build pyopencl against conda's OpenCL driver in the following way::

   $ conda install ocl-icd                    # Linux
   $ conda install khronos-opencl-icd-loader  # MacOS
   $ cd emirge/pyopencl
   # Apply this patch on MacOS: https://raw.githubusercontent.com/conda-forge/pyopencl-feedstock/master/recipe/osx_flags.diff
   $ ./configure.py --cl-inc-dir=$PWD/../miniforge3/envs/ceesd/include --cl-lib-dir=$PWD/../miniforge3/envs/ceesd/lib
   $ pip install -e .

.. _record pip packages:

How can I record the exact versions of Python packages that are currently installed and reinstall them at a later time?
------------------------------------------------------------------------------------------------------------------------

Running emirge's `version.sh` script creates a requirements.txt file that
stores the exact git commits of each emirge sub-repository that were used at
the time when `version.sh` was executed. You can use this file to install the
exact versions of the packages at a later time::

   $ cd emirge/
   $ ./version.sh --output-requirements=myreq.txt
   [...]
   *** Creating requirements file with current emirge module versions
   [...]
   *** Created file 'myreq.txt'. Install it with 'pip install --src . -r myreq.txt'.

   $ pip install --src . -r myreq.txt


.. note::

   This will build pyopencl by source, which can be challenging on some systems. Please
   see :ref:`Pyopencl source installation` for information on the prerequisites of Pyopencl,
   or remove/comment the pyopencl line from the generated requirements file.

.. note::

   You can also install the packages from the created requirements.txt with a new emirge installation::

      $ ./install.sh --pip-pkgs=myreq.txt

.. _record conda packages:

How can I record the exact versions of Conda packages that are currently installed and reinstall them at a later time?
------------------------------------------------------------------------------------------------------------------------

Running emirge's `version.sh` script creates a conda environment file that
stores the exact versions of the conda packages that were installed at
the time when `version.sh` was executed. You can use this file to install the
exact versions of the packages at a later time::

   $ cd emirge/
   $ ./version.sh --output-conda-env=myenv.yml
   [...]
   *** Conda env file with current conda package versions
   [...]
   *** Created file 'myenv.yml'. Install it with 'conda env create -f myenv.yml'

   $ conda env create -f=myenv.yml [--name my_new_env] [--force]

.. note::

   The filename must end in '.yml', otherwise conda refuses to install the file.

.. note::

   You can also install the conda packages from the created environment file with a new emirge installation::

      $ ./install.sh --conda-env=myenv.yml

   To restore package versions in your entire environment, you should combine this with :ref:`record pip packages`::

      $ ./install.sh --conda-env=myenv.yml --pip-pkgs=myreq.txt

.. warning::

   The environment file can **not** be used to install conda packages on a different architecture or OS. For example,
   an environment file created on MacOS won't be installable on Linux. The reasons are that conda package versions are
   unique to each OS/architecture, and that different systems require different packages (for example, the `pocl-cuda`
   package only exists on Linux, but not on MacOS).
