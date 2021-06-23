"""Demonstrate wave-eager MPI example."""

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


import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from meshmode.array_context import thaw, PyOpenCLArrayContext

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.wave import wave_operator

import pyopencl.tools as cl_tools


def bump(actx, discr, t=0):
    """Create a bump."""
    source_center = np.array([0.2, 0.35, 0.1])[:discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, discr.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


@mpi_entry_point
def main(snapshot_pattern="wave-eager-{step:04d}-{rank:04d}.pkl", restart_step=None):
    """Drive the example."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    if restart_step is None:

        from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
        mesh_dist = MPIMeshDistributor(comm)

        dim = 2
        nel_1d = 16

        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dim, b=(0.5,)*dim,
                nelements_per_axis=(nel_1d,)*dim)

            print("%d elements" % mesh.nelements)
            part_per_element = get_partition_by_pymetis(mesh, num_parts)
            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()

        fields = None

    else:
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(
            actx, snapshot_pattern.format(step=restart_step, rank=rank)
        )
        local_mesh = restart_data["local_mesh"]
        nel_1d = restart_data["nel_1d"]
        assert comm.Get_size() == restart_data["num_parts"]

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                                  mpi_communicator=comm)

    if local_mesh.dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.7 / (nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.4 / (nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    t_final = 3

    if restart_step is None:
        t = 0
        istep = 0

        fields = flat_obj_array(
            bump(actx, discr),
            [discr.zeros(actx) for i in range(discr.dim)]
            )

    else:
        t = restart_data["t"]
        istep = restart_step
        assert istep == restart_step
        restart_fields = restart_data["fields"]
        old_order = restart_data["order"]
        if old_order != order:
            old_discr = EagerDGDiscretization(actx, local_mesh, order=old_order,
                                              mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, discr.discr_from_dd("vol"),
                                                   old_discr.discr_from_dd("vol"))
            fields = connection(restart_fields)
        else:
            fields = restart_fields

    vis = make_visualizer(discr)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    while t < t_final:
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

        if istep % 10 == 0:
            print(istep, t, discr.norm(fields[0]))
            vis.write_parallel_vtk_file(
                comm,
                "fld-wave-eager-mpi-%03d-%04d.vtu" % (rank, istep),
                [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                ]
            )

        fields = rk4_step(fields, t, dt, rhs)

        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
