======================================
Welcome to |mirgecom|'s documentation!
======================================

.. module:: mirgecom

Math, Intermediate Representation, Compressible Flow, For Scramjets.
|mirgecom| aims to be a library of parts from which scramjet simulations
for the Center for Exascale-Enabled Scramjet Design (CEESD) at the
University of Illinois can be built.

Here’s an example, to give you an impression:

.. code-block:: python

   import numpy as np
   import pyopencl as cl
   from pytools.obj_array import flat_obj_array
   from grudge.eager import EagerDGDiscretization
   from grudge.shortcuts import make_visualizer
   from mirgecom.wave import wave_operator
   from mirgecom.integrators import rk4_step
   from meshmode.array_context import PyOpenCLArrayContext

   cl_ctx = cl.create_some_context()
   queue = cl.CommandQueue(cl_ctx)
   actx = PyOpenCLArrayContext(queue)

   dim = 2
   nel_1d = 16
   order = 3
   dt = 0.75 / (nel_1d * order ** 2)

   from meshmode.mesh.generation import generate_regular_rect_mesh
   mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim, n=(nel_1d,)*dim)

   print("%d elements" % mesh.nelements)

   discr = EagerDGDiscretization(actx, mesh, order=order)
   fields = flat_obj_array(
       [discr.zeros(actx)],
       [discr.zeros(actx) for i in range(discr.dim)]
       )
   vis = make_visualizer(discr, order + 3)

   def rhs(t, w):
       return wave_operator(discr, c=1, w=w)

   t = 0
   t_final = 3
   istep = 0

   while t < t_final:
       fields = rk4_step(fields, t, dt, rhs)
       if istep % 10 == 0:
           print(istep, t, discr.norm(fields[0], np.inf))
           vis.write_vtk_file("wave-eager-%04d.vtu" % istep,
                   [("u", fields[0]), ("v", fields[1:]), ])

       t += dt
       istep += 1

(This example is derived from
:download:`examples/wave.py <../examples/wave.py>` in the |mirgecom|
source distribution.)

Table of Contents
=================

.. toctree::
    :numbered:

    fluid
    discretization
    operators/operators
    support/support
    development/development
    running/running
    faq/faq
    misc
    🚀 Github <https://github.com/illinois-ceesd/mirgecom>
    💾 Download Releases <https://pypi.pythonorg/project/mirgecom>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
