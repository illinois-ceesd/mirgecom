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

import grudge.op as op
import numpy as np
from grudge.shortcuts import make_visualizer
from logpyle import IntervalTimer, set_dt
from pytools.obj_array import flat_obj_array

from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_mempool_usage)
from mirgecom.utils import force_evaluation
from mirgecom.wave import wave_operator


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


def main(actx_class, use_logmgr: bool = False,
         casename="wave-nompi") -> None:
    """Drive the example."""
    logmgr = initialize_logmgr(use_logmgr,
        filename="wave.sqlite", mode="wu")

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, None)
    queue = getattr(actx, "queue", None)
    alloc = getattr(actx, "allocator", None)
    use_profiling = actx_class_is_profiling(actx_class)

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    order = 3

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    current_cfl = 0.485
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, dcoll) / wave_speed
    dt = actx.to_numpy(current_cfl
                       * op.nodal_min(dcoll, "vol",
                                    nodal_dt))[()]  # type: ignore[index]

    print("%d elements" % mesh.nelements)

    fields = flat_obj_array(bump(actx, nodes),
                            [dcoll.zeros(actx) for i in range(dim)])

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
        return wave_operator(dcoll, c=wave_speed, w=w)

    compiled_rhs = actx.compile(rhs)
    fields = force_evaluation(actx, fields)

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
                from mirgecom.profiling import PyOpenCLProfilingArrayContext
                assert isinstance(actx, PyOpenCLProfilingArrayContext)
                print(actx.tabulate_profiling_data())
            print(istep, t, actx.to_numpy(op.norm(dcoll, fields[0], 2)))
            vis.write_vtk_file(f"{casename}-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ], overwrite=True)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()


if __name__ == "__main__":
    from mirgecom.simutil import add_general_args
    import argparse
    parser = argparse.ArgumentParser(description="Wave (non-MPI version)")
    add_general_args(parser, leap=False, overintegration=False, restart_file=False)
    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=False,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

    main(actx_class, use_logmgr=args.log, casename=args.casename)

# vim: foldmethod=marker
