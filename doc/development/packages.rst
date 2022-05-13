What Packages are Involved?
===========================

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
        meshmode -> arraycontext;

        arraycontext -> pyopencl;
        arraycontext -> loopy;
        arraycontext -> pytato;

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
   * - :mod:`grudge` (`GitHub <https://github.com/inducer/grudge>`__)
     - 1/2/3D discontinuous Galerkin based on meshmode.
   * - :mod:`meshmode` (`GitHub <https://github.com/inducer/meshmode>`__)
     - (Unstructured, high-order) discontinuous piecewise polynomial discretizations.
   * - :mod:`arraycontext` (`GitHub <https://github.com/inducer/arraycontext>`__)
     - Creation of and operations on (lazy/GPU/...) arrays, array containers.
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
`requirements.txt in mirgecom <https://github.com/illinois-ceesd/mirgecom/blob/main/requirements.txt>`__.
