OpenCL device selection
=======================

The :mod:`pyopencl` package supports selecting the device on which to run OpenCL code.
There are multiple ways in which this selection can happen:

- **Interactively.** When running mirgecom on the command line (i.e., not in a
  batch script), it will present a (potentially multiple-choice) dialog with
  the available OpenCL drivers (such as pocl) and the device (such as CPU or
  GPU). After a device has been selected, pyopencl will print the chosen
  device in the form of a value for the ``PYOPENCL_TEST`` environment
  variable that can be used for subsequent executions.

- **Through the PYOPENCL_TEST or PYOPENCL_CTX environment variables.**
  Pyopencl will use the value of the ``PYOPENCL_TEST`` or ``PYOPENCL_CTX``
  environment variables when set for device selection. The value of either variable
  can either be list indices (such as ``0:1`` for first driver, second device),
  or named abbreviations (such as ``port:tesla`` for pocl (=Portable OpenCL)
  and the first Nvidia Tesla GPU).

- **Default device.** When not using one of the options above, pyopencl will
  run on the first available device (which might not be the device you want)
  by default.

.. note::

   The device selection functionality described here is provided by the
   :func:`pyopencl.create_some_context`,
   :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`, and
   :func:`arraycontext.pytest_generate_tests_for_pyopencl_array_context`
   functions used in the default simulation drivers and tests. It is also
   possible to write your own device selection code with
   :func:`pyopencl.get_platforms`, :meth:`pyopencl.Platform.get_devices`, and
   :class:`pyopencl.Context`.

.. note::

   Each MPI rank (=Python process) can only use one device during the
   execution, even if there are multiple devices available. Although it is
   possible to run multiple MPI ranks on the same device, we do not recommend
   doing this, as it will lead to contention.

.. note::

   You can also use the ``clinfo`` command (automatically installed when installing
   emirge) to list all available OpenCL devices and their parameters.
