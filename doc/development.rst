Development How-To
==================

What Packages are Involved?
---------------------------

:mod:`mirgecom` relies on a number of other packages to do its job, which
depend on each other as illustrated in this graph:

.. graphviz::

   digraph deps {
   subgraph cluster_0 {
   label="emirge"
        mirgecom -> meshmode;
        grudge -> meshmode;
        mirgecom -> grudge;

        mirgecom -> loopy;
        grudge -> loopy;
        meshmode -> loopy;

        meshmode -> pyopencl;
        loopy -> pyopencl;

        meshmode -> modepy;

        loopy -> pymbolic;

        pyopencl -> pocl;

        mirgecom -> pytato;
        pytato -> loopy;
        pytato -> pymbolic;
        graph[style=dotted];
        }
   }

What do these packages do?

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Description
   * - :mod:`mirgecom` (`GitHub <https://github.com/illinois-ceesd/mirgecom>`__)
     - Where the main science code lives (flow, combustion, walls).
   * - :mod:`meshmode` (`GitHub <https://github.com/inducer/meshmode>`__)
     - (Unstructured, high-order) discontinuous piecewise polynomial discretizations.
   * - :mod:`grudge` (`GitHub <https://github.com/inducer/meshmode>`__)
     - 1/2/3D discontinuous Galerkin based on meshmode.
   * - :mod:`loopy` (`GitHub <https://github.com/inducer/loopy>`__)
     - A code generator for array-based code on CPUs and GPUs.
   * - :mod:`pyopencl` (`GitHub <https://github.com/inducer/pyopencl>`__)
     - OpenCL integration for Python.
   * - :mod:`modepy` (`GitHub <https://github.com/inducer/modepy>`__)
     - Modes and nodes for high-order discretizations.
   * - :mod:`pymbolic` (`GitHub <https://github.com/inducer/pymbolic>`__)
     - Expression tree and symbolic manipulation library.
   * - :mod:`pytato` (`GitHub <https://github.com/inducer/pytato>`__)
     - Lazily evaluated arrays in Python.
   * - pocl (`GitHub <https://github.com/pocl/pocl>`__)
     - OpenCL runtime for CPUs and GPUs written in C.
   * - emirge (`GitHub <https://github.com/illinois-ceesd/emirge>`__)
     - Scripts to manage a mirgecom installation and its dependencies.



The source repository (and current branch) of most of these packages
in use is determined by the file
`requirements.txt in mirgecom <https://github.com/illinois-ceesd/mirgecom/blob/master/requirements.txt>`__.

Installation
------------

The `emirge repository <https://github.com/illinois-ceesd/emirge>`__ contains some
scripts to help with installation and simultaneously has its checkout serve as a root
directory for development.

See the installation instructions for `emirge
<https://github.com/illinois-ceesd/emirge/>`_ for comprehensive instructions.
In most cases, running emirge's ``install.sh`` script should be sufficient to
set up a working version of mirgecom and all its dependencies:

.. code-block:: bash

   # Clone and install emirge
   $ git clone https://github.com/illinois-ceesd/emirge
   $ cd emirge
   $ ./install.sh

   # Activate the just installed packages
   $ source config/activate_env.sh

   # Run a quick test
   $ cd mirgecom/examples
   $ python ./wave-eager.py


.. note::

   These instructions work on macOS or Linux, including on clusters and DOE supercomputers.
   If you have a Windows machine, try
   `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
   If that works, please submit a pull request updating this documentation
   with a procedure that worked for you.

Proposing Changes
-----------------

.. todo::

   Write this.

Building this Documentation
---------------------------

The following should do the job::

    # make sure your conda env is active
    conda install sphinx graphviz
    cd mirgecom/doc
    make html

After that, point a browser at :file:`mirgecom/doc/_build/html/index.html` to
see your documentation.
