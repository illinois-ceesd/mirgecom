"""Demonstrate wave serial example."""

__copyright__ = "Copyright (C) 2020 University of Illinos Board of Trustees"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.array as cla  # noqa
import pyopencl.tools as cl_tools

from pytools.obj_array import flat_obj_array

from grudge.shortcuts import make_visualizer
import grudge.op as op

from mirgecom.discretization import create_discretization_collection
from mirgecom.wave import wave_operator
from mirgecom.integrators import rk4_step
from mirgecom.utils import force_evaluation

from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from logpyle import IntervalTimer, set_dt

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)


def bump(actx, nodes, t=0):
    """Create a bump."""
    dim = len(nodes)
    source_center = np.array([0.2, 0.35, 0.1])[:dim]
    source_width = 0.05
    source_omega = 3

    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main(use_profiling=False, use_logmgr=False, lazy: bool = False):
    """Drive the example."""
    cl_ctx = cl.create_some_context()

    logmgr = initialize_logmgr(use_logmgr,
        filename="wave.sqlite", mode="wu")

    if use_profiling:
        if lazy:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        queue = cl.CommandQueue(cl_ctx)
        if lazy:
            actx = PytatoPyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
        else:
            actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    order = 3

    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(discr.nodes())

    current_cfl = 0.485
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, discr) / wave_speed
    dt = actx.to_numpy(current_cfl * op.nodal_min(discr, "vol",
                                                  nodal_dt))[()]

    print("%d elements" % mesh.nelements)

    fields = flat_obj_array(bump(actx, nodes),
                            [discr.zeros(actx) for i in range(dim)])

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(discr)

    def rhs(t, w):
        return wave_operator(discr, c=wave_speed, w=w)

    compiled_rhs = actx.compile(rhs)

    t = 0
    t_final = 1
    istep = 0
    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        fields = rk4_step(fields, t, dt, compiled_rhs)
        fields = force_evaluation(actx, fields)

        if istep % 10 == 0:
            if use_profiling:
                print(actx.tabulate_profiling_data())
            print(istep, t, actx.to_numpy(op.norm(discr, fields[0], 2)))
            vis.write_vtk_file("fld-wave-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ], overwrite=True)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wave (non-MPI version)")
    parser.add_argument("--profile", action="store_true",
        help="enable kernel profiling")
    parser.add_argument("--logging", action="store_true",
        help="enable logging")
    parser.add_argument("--lazy", action="store_true",
        help="enable lazy evaluation")
    args = parser.parse_args()

    main(use_profiling=args.profile, use_logmgr=args.logging, lazy=args.lazy)

# vim: foldmethod=marker
