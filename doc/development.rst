Development How-To
==================

What Packages are Involved?
---------------------------

:mod:`mirgecom` relies on a number of other packages to do its job, which
depend on each other as illustrated in this graph:

.. graphviz::

   digraph deps {
        mirgecom -> meshmode;
        grudge -> meshmode;
        mirgecom -> grudge;

        mirgecom -> loopy;
        grudge -> loopy;
        meshmode -> loopy;

        meshmode -> pyopencl;
        loopy -> pyopencl;

        meshmode -> modepy;

        mirgecom -> pymbolic;
        loopy -> pymbolic;
   }

What do these pacakges do?

.. todo::

   Write this.

The source repository (and current branch) of each of these pacakges
in use is determined by the file
`requirements.txt in mirgecom <https://github.com/illinois-ceesd/mirgecom/blob/master/requirements.txt>`__.

Overview of the Setup
---------------------

The `emirge repository <https://github.com/illinois-ceesd/emirge>`__ contains some
scripts to help with installation and simultaneously has its checkout serve as a root
directory for development.

.. todo:

    - Conda environment
    - Editable installation

Installation
------------

See the installation instructions for the `emirge
<https://github.com/illinois-ceesd/emirge/>`_ installation infrastructure.

.. note::

    Should we move those here?

Installing on Your Personal Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    These instructions work on macOS or Linux. If you have a Windows machine, try
    `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
    If that works, please submit a pull request updating this documentation
    with a procedure that worked for you.

Installing on a Cluster/DOE Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

   Write this.

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
