"""Demonstrate Burger's equation example."""

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

import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
from grudge.trace_pair import TracePair, interior_trace_pairs
from pytools.obj_array import flat_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
)

from grudge.shortcuts import make_visualizer
import grudge.op as op

from mirgecom.discretization import create_discretization_collection
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.utils import force_evaluation
from mirgecom.operators import div_operator

from logpyle import IntervalTimer, set_dt

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage,
                                         logmgr_add_mempool_usage,)

import sys

def utx(actx, nodes, t=0):
    """Initialize the field for Burgers Eqn."""
    x = nodes[0]
    return actx.np.sin(2.0*np.pi*nodes[0]/2.0) 

#    """Create a bump."""
#    return actx.np.exp(-(nodes[0]-1.0)**2/0.1**2)

def _facial_flux(dcoll, u_tpair):

    actx = u_tpair.int.array_context
    normal = actx.thaw(dcoll.normal(u_tpair.dd))
    u2_pair = TracePair(dd=u_tpair.dd,
                        interior=0.5*u_tpair.int**2,
                        exterior=0.5*u_tpair.ext**2)

    # average
    flux_weak = 0.5*(u2_pair.ext + u2_pair.int)*normal[0]

#    # dissipation term
#    lamb = actx.np.abs( actx.np.maximum( u_tpair.int, u_tpair.ext ) )
#    flux_weak = flux_weak + .5*lamb*u_tpair.diff*normal[0]

    return op.project(dcoll, u_tpair.dd, "all_faces", flux_weak)


class _BurgTag:
    pass


def burgers_operator(dcoll, u, u_bc, *, comm_tag=None):
    """Compute the RHS of the inviscid Burger's Equation.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    u: DOF array representing the independent variable
    u_bc: DOF array representing the Dirichlet boundary value
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)

    u_bnd = op.project(dcoll, "vol", BTAG_ALL, u)
    itp = interior_trace_pairs(dcoll, u, comm_tag=(_BurgTag, comm_tag))
    
    #FIXME maybe flip the sign of u_bc to enforce 0 at the boundary
    el_bnd_flux = (
        _facial_flux(dcoll, u_tpair=TracePair(BTAG_ALL,
                                              interior=u_bnd, exterior=u_bc))
        + sum([_facial_flux(dcoll, u_tpair=tpair) for tpair in itp]))
    vol_flux = -op.weak_local_grad(dcoll, 0.5*u**2)[0]
    return -op.inverse_mass(dcoll, vol_flux +
                            op.face_mass(dcoll, el_bnd_flux))


@mpi_entry_point
def main(actx_class, snapshot_pattern="burgers-mpi-{step:04d}-{rank:04d}.pkl",
         restart_step=None, use_profiling=False, use_logmgr=False, lazy=False):
    """Drive the Burgers Equation example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="burgers-mpi.sqlite", mode="wo", mpi_comm=comm)
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    if restart_step is None:

        from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
        mesh_dist = MPIMeshDistributor(comm)

        dim = 1
        nel_1d = 200

        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                a=(0.0,)*dim, b=(2.0,)*dim,
                nelements_per_axis=(nel_1d,)*dim)

            print("%d elements" % mesh.nelements)
            part_per_element = get_partition_by_pymetis(mesh, num_parts)
            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()

    else:

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(
            actx, snapshot_pattern.format(step=restart_step, rank=rank)
        )
        local_mesh = restart_data["local_mesh"]
        nel_1d = restart_data["nel_1d"]
        assert comm.Get_size() == restart_data["num_parts"]

    order = 2
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    current_cfl = 0.485
    wave_speed = 1.0
    from grudge.dt_utils import characteristic_lengthscales
    nodal_dt = characteristic_lengthscales(actx, dcoll) / wave_speed

    dt = actx.to_numpy(current_cfl * op.nodal_min(dcoll, "vol", nodal_dt))[()]

    t_final = 1.0

    if restart_step is None:
        t = 0
        istep = 0
        u = utx(actx, nodes, t=0)

    else:
        t = restart_data["t"]
        istep = restart_step
        assert istep == restart_step
        restart_u = restart_data["u"]
        old_order = restart_data["order"]
        if old_order != order:
            old_dcoll = create_discretization_collection(
                actx, local_mesh, order=old_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, dcoll.discr_from_dd("vol"),
                                                   old_dcoll.discr_from_dd("vol"))
            u = connection(restart_u)
        else:
            u = restart_u

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

    def rhs(t, u):
        u_bc = op.project(dcoll, "vol", BTAG_ALL,
                          utx(actx, nodes, t=t))
        return burgers_operator(dcoll, u=u, u_bc=u_bc)

    u = force_evaluation(actx, u)
    compiled_rhs = actx.compile(rhs)

    restart_n = 1000
    viz_n = 10

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        # restart must happen at beginning of step
        if istep % restart_n == 0 and (
                # Do not overwrite the restart file that we just read.
                istep != restart_step):
            from mirgecom.restart import write_restart_file
            write_restart_file(
                actx, restart_data={
                    "local_mesh": local_mesh,
                    "order": order,
                    "u": u,
                    "t": t,
                    "step": istep,
                    "nel_1d": nel_1d,
                    "num_parts": num_parts},
                filename=snapshot_pattern.format(step=istep, rank=rank),
                comm=comm
            )

        if istep % viz_n == 0:
            print(istep, t, actx.to_numpy(op.norm(dcoll, u, np.inf)))
            vis.write_parallel_vtk_file(
                comm,
                "burgers-mpi-%04d-%04d.vtu" % (rank, istep),
                [
                    ("u", u),
                    ("x", nodes[0])
                ], overwrite=True
            )

        d = rk4_step(u, t, dt, compiled_rhs)
        u  = force_evaluation(actx, d)

#        break

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()

    final_u = actx.to_numpy(op.norm(dcoll, u, np.inf))
    assert np.abs(final_u) < 1e-14


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Burgers Equation (MPI version)")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true",
        help="enable logging")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    args = parser.parse_args()
    lazy = args.lazy

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    main(actx_class, use_profiling=args.profiling, use_logmgr=args.log, lazy=lazy)

# vim: foldmethod=marker
