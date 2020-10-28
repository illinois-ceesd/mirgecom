Frequently Asked Questions
==========================

.. note::

   Have a question? Create a pull request adding the question to this file and tag
   one of the developers in the pull request to get them to add an answer. Thanks!

Why on earth does the code...?
------------------------------

Why does everything go through :mod:`loopy`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- We don't quite know what execution hardware we'll target, and even if we did, the
  answer to that might change over time. As of this writing, it's likely that at
  least GPUs by Nvidia, Intel, and AMD are in the mix. But large-vector ARM
  CPUs are not out of the question.

  Loopy lets us generate code for all these :class:`~loopy.TargetBase`\ s (e.g.
  CUDA, OpenCL) with ease.

- Likely, the loop structure for efficient code will not look the same on each
  machine. (E.g. because of vectorization, or GPU threads/blocks, or...) So we need a
  tool that lets us write reasonably readable code while being able to
  rewrite/transform that code to suit the loop structure that the machine likes.
  That tool is :mod:`loopy`.

- If we allowed some code to sidestep :mod:`loopy`, then all of those functions would
  have to be performed separately, and likely by hand, for that piece of code, which
  is likely not feasible.

What's the point of array contexts?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Arrays are the central large-scale data structure that :mod:`mirgecom` operates on.
:class:`meshmode.array_context.ArrayContext` allows us flexibility in how those are
created (and which ones we use) just by changing which array context we run with.
Some array types you might choose include :class:`pyopencl.array.Array`\ s on the GPU
through OpenCL, or lazily-evaluated :class:`pytato.Array`\ s.


Installation questions
----------------------

What is conda/anaconda/conda-forge/miniconda/miniforge?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the packages required by mirgecom are installed as binary packages by default, via conda.
These packages (python, pocl, pyopencl, and islpy, among others) are difficult or time-consuming
to install from source, and source builds are more easily impacted by other software present on the system.

How can I install pocl from source?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, it can be helpful to install certain packages from source, for debugging or to install
a git version. Most packages are straightforward to install from source. For pocl, you can follow this
`installation script <https://gist.github.com/matthiasdiener/838ccbdb5d8f4e4917b58fe3da811777>`__.

How can I build pyopencl from source?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyopencl needs to be built against an OpenCL runtime and you therefore need to make sure
that the build process picks up the right runtime. This can be challenging especially on MacOS, since Apple provides its own CL runtime which does not easily compile against pyopencl.

You can build pyopencl against conda's OpenCL driver in the following way::

   $ conda install ocl-icd # Linux
   $ conda install khronos-opencl-icd-loader # MacOS
   $ cd emirge/pyopencl
   $ ./configure.py --cl-inc-dir=$PWD/../miniforge3/envs/ceesd/include --cl-lib-dir=$PWD/../miniforge3/envs/ceesd/lib
   $ pip install -e .

