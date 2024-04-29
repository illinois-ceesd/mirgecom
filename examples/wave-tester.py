"""Simple testing driver for debugging."""

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

from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_mempool_usage)
from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation
from mirgecom.wave import wave_operator
from mirgecom.continuity import continuity_operator

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
    logfile_pattern = casename + "-mpi.sqlite"
    logmgr = initialize_logmgr(use_logmgr,
        filename=logfile_pattern, mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    alloc = getattr(actx, "allocator", None)
    use_profiling = actx_class_is_profiling(actx_class)
    dim = 2
    nel_1d = 16

    if restart_step is None:

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

    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1.0
    wave_speed = np.sqrt(np.dot(velocity, velocity))

    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, dcoll) / wave_speed

    dt = actx.to_numpy(current_cfl
                       * op.nodal_min(dcoll,
                                      "vol", nodal_dt))[()]  # type: ignore[index]

    t_final = 0.01

    if restart_step is None:
        t = 0
        istep = 0
        if casename == "wave":
            fields = flat_obj_array(
                bump(actx, nodes),
                [dcoll.zeros(actx) for i in range(dim)]
            )
        else:
            fields = bump(actx, nodes)

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

    rhs_const = -1
    # rhs_const = 1
    # rhs_const = 0

    def wave_rhs(t, w):
        return wave_operator(dcoll, c=wave_speed, w=w)

    def simple_rhs(t, w):
        return rhs_const * w

    def continuity_rhs(t, w):
        return continuity_operator(dcoll, v=velocity, u=w)

    nviz = 0
    nrestart = 0
    nstatus = 1
    vis = make_visualizer(dcoll) if nviz else None

    if rhs_const < 0:
        if casename == "wave":
            compiled_rhs = actx.compile(wave_rhs)
        else:
            compiled_rhs = actx.compile(continuity_rhs)
    else:
        compiled_rhs = actx.compile(simple_rhs)

    fields = force_evaluation(actx, fields)

    while t < t_final:
        if istep % nstatus == 0:
            if casename == "wave":
                soln = fields[0]
            else:
                soln = fields
            print(f"{istep=}, {t=}, soln={actx.to_numpy(op.norm(dcoll, soln, 2))}")

        if logmgr:
            logmgr.tick_before()

        # restart must happen at beginning of step
        if (nrestart > 0) and (istep % nrestart == 0) and (
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
                    "num_parts": num_parts,
                    "casename": casename},
                filename=snapshot_pattern.format(step=istep, rank=rank),
                comm=comm
            )

        if (nviz > 0) and (istep % nviz == 0):
            if casename == "wave":
                viz_q = [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                ]
            else:
                viz_q = [("u", fields)]
            vis.write_parallel_vtk_file(
                comm,
                vizfile_pattern % (rank, istep),
                viz_q, overwrite=True
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

    if casename == "wave":
        soln = fields[0]
    else:
        soln = fields

    final_soln = actx.to_numpy(op.norm(dcoll, soln, 2))
    # if casename == "wave" and (rhs_const < 0):
    #     assert np.abs(final_soln - 0.04409852463947439) < 1e-14  # type: ignore[operator]

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
