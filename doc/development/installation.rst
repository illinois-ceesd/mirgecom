Installation
============

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
   $ python ./wave.py


.. note::

   These instructions work on macOS or Linux, including on clusters and DOE supercomputers.
   If you have a Windows machine, try
   `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
   If that works, please submit a pull request updating this documentation
   with a procedure that worked for you.
