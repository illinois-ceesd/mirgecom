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

What do these packages do?

.. todo::

   Write this.

The source repository (and current branch) of each of these packages
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
