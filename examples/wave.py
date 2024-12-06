"""Demonstrate wave example."""

__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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
import logging

import sys
import grudge.op as op
import numpy as np
from grudge.shortcuts import make_visualizer
from logpyle import IntervalTimer, set_dt
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from pytools.obj_array import flat_obj_array

from mirgecom.array_context import initialize_actx
from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_mempool_usage)
from mirgecom.mpi import mpi_entry_point
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


@mpi_entry_point
def main(actx_class, casename="wave",
         restart_step=None, use_logmgr: bool = False, mpi: bool = True) -> None:
    """Drive the example."""

    if mpi:
        assert "mpi4py" in sys.modules
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_parts = comm.Get_size()
    else:
        assert "mpi4py" not in sys.modules
        comm = None
        rank = 0
        num_parts = 1

    snapshot_pattern = casename + "-{step:04d}-{rank:04d}.pkl"
    vizfile_pattern = casename + "-%03d-%04d.vtu"

    logmgr = initialize_logmgr(use_logmgr,
        filename="wave.sqlite", mode="wu", mpi_comm=comm)

    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    alloc = getattr(actx, "allocator", None)

    from mirgecom.array_context import actx_class_is_profiling
    use_profiling = actx_class_is_profiling(actx_class)

    if restart_step is None:

        dim = 2
        nel_1d = 16

        from functools import partial
        from meshmode.mesh.generation import generate_regular_rect_mesh

        generate_mesh = partial(generate_regular_rect_mesh,
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim)

        if comm:
            from mirgecom.simutil import distribute_mesh
            local_mesh, global_nelements = distribute_mesh(comm, generate_mesh)
        else:
            local_mesh = generate_mesh()
            global_nelements = local_mesh.nelements

        if rank == 0:
            print(f"{global_nelements} elements")

        fields = None

    else:
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(
            actx, snapshot_pattern.format(step=restart_step, rank=rank)
        )
        local_mesh = restart_data["local_mesh"]
        nel_1d = restart_data["nel_1d"]
        assert num_parts == restart_data["num_parts"]

    order = 3

    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    current_cfl = 0.485
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, dcoll) / wave_speed

    dt = actx.to_numpy(current_cfl
                       * op.nodal_min(dcoll,
                                      "vol", nodal_dt))[()]

    t_final = 1

    if restart_step is None:
        t = 0
        istep = 0

        fields = flat_obj_array(
            bump(actx, nodes),
            [dcoll.zeros(actx) for i in range(dim)]
            )

    else:
        t = restart_data["t"]
        istep = restart_step
        assert istep == restart_step
        restart_fields = restart_data["fields"]
        old_order = restart_data["order"]
        if old_order != order:
            old_dcoll = create_discretization_collection(
                actx, local_mesh, order=old_order)
            from meshmode.discretization.connection import \
                make_same_mesh_connection
            connection = make_same_mesh_connection(actx, dcoll.discr_from_dd("vol"),
                                                   old_dcoll.discr_from_dd("vol"))
            fields = connection(restart_fields)
        else:
            fields = restart_fields

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_mempool_usage(logmgr, alloc)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        try:
            logmgr.add_watches(
                ["memory_usage_mempool_managed.max",
                 "memory_usage_mempool_active.max"])
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

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        # restart must happen at beginning of step
        if istep % 100 == 0 and (
                # Do not overwrite the restart file that we just read.
                istep != restart_step):
            from mirgecom.restart import write_restart_file
            write_restart_file(
                actx, restart_data={
                    "local_mesh": local_mesh,
                    "order": order,
                    "fields": fields,
                    "t": t,
                    "step": istep,
                    "nel_1d": nel_1d,
                    "num_parts": num_parts},
                filename=snapshot_pattern.format(step=istep, rank=rank),
                comm=comm
            )

        if istep % 100 == 0:
            print(istep, t, actx.to_numpy(op.norm(dcoll, fields[0], 2)))
            vis.write_parallel_vtk_file(
                comm,
                vizfile_pattern % (rank, istep),
                [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                ], overwrite=True
            )

        fields = rk4_step(fields, t, dt, compiled_rhs)
        fields = force_evaluation(actx, fields)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()

    final_soln = actx.to_numpy(op.norm(dcoll, fields[0], 2))
    assert np.abs(final_soln - 0.04409852463947439) < 1e-14

    if mpi:
        assert "mpi4py" in sys.modules
    else:
        assert "mpi4py" not in sys.modules


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Wave")
    parser.add_argument("--profiling", action="store_true",
        help="enable kernel profiling")
    parser.add_argument("--log", action="store_true",
        help="enable logging")
    parser.add_argument("--lazy", action="store_true",
        help="enable lazy evaluation")
    parser.add_argument("--casename", help="casename to use for i/o")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--mpi", default=True, action=argparse.BooleanOptionalAction,
        help="use MPI")
    args = parser.parse_args()
    casename = args.casename or "wave"

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=args.mpi,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

    if args.mpi:
        main_func = main
    else:
        import inspect
        # run main without the mpi_entry_point wrapper
        main_func = inspect.unwrap(main)

    main_func(actx_class, use_logmgr=args.log, casename=casename, mpi=args.mpi)

# vim: foldmethod=marker
