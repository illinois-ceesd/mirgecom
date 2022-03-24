Common error messages
=====================


How can I fix the ``clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR`` error?
----------------------------------------------------------------------------

This error appears at mirgecom startup when the OpenCL loader (ocl-icd) can not
find any valid OpenCL platform::

   $ python examples/wave.py
   Traceback (most recent call last):
     File "examples/wave.py", line 134, in <module>
       main(use_profiling=args.profile)
     File "examples/wave.py", line 62, in main
       cl_ctx = cl.create_some_context()
     File "/usr/WS1/diener3/Work/emirge/pyopencl/pyopencl/__init__.py", line 1432, in create_some_context
       platforms = get_platforms()
   pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR


This can be caused by multiple issues:

- The ``pocl`` package is not installed. You can install it with ``$ conda install pocl``.
  Consider installing ``pocl-cuda`` as well on Linux systems with Nvidia GPUs.
- The OpenCL loader cannot find the system OpenCL drivers. You can add
  support for the system CL platforms by installing the ``ocl-icd-system`` (on Linux) or ``khronos-opencl-icd-loader`` (on MacOS) package with ``conda``.
- The loader is unable to load the platform for other reasons. You can further
  debug such an issue by running ``$ export OCL_ICD_DEBUG=7`` before starting
  mirgecom, which will give more output about what the loader is doing.

  As an example, here is an error shown on Lassen::

     $ export OCL_ICD_DEBUG=7
     $ python examples/wave.py
     ocl-icd(ocl_icd_loader.c:776): __initClIcd: Reading icd list from '/g/g91/diener3/Work/emirge/miniforge3/envs/ceesd/etc/OpenCL/vendors'
     ocl-icd(ocl_icd_loader.c:234): _find_num_icds: return: 1/0x1
     ocl-icd(ocl_icd_loader.c:265): _open_driver: Considering file '/g/g91/diener3/Work/emirge/miniforge3/envs/ceesd/etc/OpenCL/vendors/pocl.icd'
     ocl-icd(ocl_icd_loader.c:239): _load_icd: Loading ICD '/g/g91/diener3/Work/emirge/miniforge3/envs/ceesd/lib/libpocl.so.2.5.0'
     ocl-icd(ocl_icd_loader.c:246): _load_icd: error while dlopening the IDL: '/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /g/g91/diener3/Work/emirge/miniforge3/envs/ceesd/lib/libpocl.so.2.5.0)',
       => skipping ICD
     ocl-icd(ocl_icd_loader.c:297): _open_driver: return: 0/0x0
     ocl-icd(ocl_icd_loader.c:320): _open_drivers: return: 0/0x0
     ocl-icd(ocl_icd_loader.c:1060): clGetPlatformIDs: return: -1001/0xfffffffffffffc17
     Traceback (most recent call last):
       File "examples/wave.py", line 134, in <module>
         main(use_profiling=args.profile)
       File "examples/wave.py", line 62, in main
         cl_ctx = cl.create_some_context()
       File "/usr/WS1/diener3/Work/emirge/pyopencl/pyopencl/__init__.py", line 1432, in create_some_context
         platforms = get_platforms()
     pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

  This error occurs because pyopencl was built by source with an incompatible
  gcc version. Load a newer gcc module (``$ ml load gcc/8.3.1`` should work),
  and recompile pyopencl.


What does ``clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES`` mean?
-------------------------------------------------------------------

This error message indicates that there is not enough memory available
to run the simulation::

     File "<generated code for 'invoke__pt_kernel_loopy_kernel'>", line 996, in invoke__pt_kernel_loopy_kernel
     File "<generated code for 'invoke__pt_kernel_loopy_kernel'>", line 62, in _lpy_host__pt_kernel
   pyopencl._cl.RuntimeError: clEnqueueNDRangeKernel failed: OUT_OF_RESOURCES


Try running on more nodes and/or devices.
