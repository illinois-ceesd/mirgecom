"""Demonstrate linear wave equation example."""

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

from grudge.shortcuts import make_visualizer
import grudge.op as op

from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.utils import force_evaluation

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from logpyle import IntervalTimer, set_dt

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_mempool_usage)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.trace_pair import TracePair, interior_trace_pairs


def bump(actx, nodes):
    """Create a bump."""
    return actx.np.exp(-(nodes[0] - 0.0)**2/0.1**2)


def _flux(dcoll, c, w_tpair):
    u = w_tpair

    actx = w_tpair.int.array_context
    normal = actx.thaw(dcoll.normal(w_tpair.dd))

    flux_weak = normal[0]*u.avg

    # upwind
    flux_weak = flux_weak + 0.5*(u.ext-u.int)*normal[0]

    return op.project(dcoll, w_tpair.dd, "all_faces", c*flux_weak)


class _WaveTag:
    pass


def linear_wave_operator(dcoll, c, w, *, comm_tag=None):
    """Compute the RHS of the linear wave equation.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    c: float
        the (constant) wave speed
    w: numpy.ndarray
        an object array of DOF arrays, representing the state vector
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """
    u = w

    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_bval = dir_u
    dir_bc = -dir_u*0.0

    return -(
        op.inverse_mass(dcoll, -c*op.weak_local_grad(dcoll, u)[0]
        +  # noqa: W504
        op.face_mass(dcoll,
            _flux(dcoll, c=c,
                  w_tpair=TracePair(BTAG_ALL, interior=dir_bval,
                                    exterior=dir_bc))
            + sum(
                _flux(dcoll, c=c, w_tpair=tpair)
                for tpair in interior_trace_pairs(
                    dcoll, w, comm_tag=(_WaveTag, comm_tag)))
            )
        )
    )


def main(actx_class, use_profiling=False, use_logmgr=False, lazy: bool = False):
    """Drive the example."""
    cl_ctx = cl.create_some_context()

    logmgr = initialize_logmgr(use_logmgr,
        filename="wave.sqlite", mode="wo")

    from mirgecom.simutil import get_reasonable_memory_pool

    if use_profiling:
        if lazy:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        alloc = get_reasonable_memory_pool(cl_ctx, queue)
        actx = PyOpenCLProfilingArrayContext(queue, allocator=alloc)
    else:
        queue = cl.CommandQueue(cl_ctx)
        alloc = get_reasonable_memory_pool(cl_ctx, queue)

        if lazy:
            actx = actx_class(queue, allocator=alloc)
        else:
            actx = actx_class(queue, allocator=alloc,
                                        force_device_scalars=True)

    dim = 1
    nel_1d = 100
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-2,)*dim,
        b=(+2,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    order = 3

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    current_cfl = 0.2
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, dcoll) / wave_speed
    dt = actx.to_numpy(current_cfl * op.nodal_min(dcoll, "vol", nodal_dt))[()]

    print("%d elements" % mesh.nelements)

    fields = bump(actx, nodes)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_mempool_usage(logmgr, alloc)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    vis = make_visualizer(dcoll)

    def rhs(t, w):
        return linear_wave_operator(dcoll, c=wave_speed, w=w)

    compiled_rhs = actx.compile(rhs)
    fields = force_evaluation(actx, fields)

    t = 0
    t_final = 1
    istep = 0
    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        if istep % 10 == 0:
            print(istep, t, actx.to_numpy(op.norm(dcoll, fields, np.inf)))
            if use_profiling:
                print(actx.tabulate_profiling_data())
            vis.write_vtk_file("linear-wave-%04d.vtu" % istep,
                    [
                        ("u", fields),
                        ("x", nodes[0]),
                        ], overwrite=True)

        fields = rk4_step(fields, t, dt, compiled_rhs)
        fields = force_evaluation(actx, fields)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    vis.write_vtk_file("linear-wave-%04d.vtu" % istep,
            [
                ("u", fields),
                ("x", nodes[0]),
                ], overwrite=True)

    if logmgr:
        logmgr.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wave (non-MPI version)")
    parser.add_argument("--profiling", action="store_true",
        help="enable kernel profiling")
    parser.add_argument("--log", action="store_true",
        help="enable logging")
    parser.add_argument("--lazy", action="store_true",
        help="enable lazy evaluation")
    args = parser.parse_args()

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=False)

    main(actx_class, use_profiling=args.profiling,
         use_logmgr=args.log, lazy=args.lazy)

# vim: foldmethod=marker
