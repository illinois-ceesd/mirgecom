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

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.symbolic.primitives import TracePair
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import diffusion_operator
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools


@mpi_entry_point
def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    nel_1d = 16

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim,
            boundary_tag_to_face={
                "dirichlet": ["+x", "-x"],
                "neumann": ["+y", "-y"]
                }
            )

        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.0025/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    source_width = 0.2

    nodes = thaw(actx, discr.nodes())

    u = discr.zeros(actx)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    dirichlet_btag = grudge_sym.DTAG_BOUNDARY("dirichlet")
    neumann_btag = grudge_sym.DTAG_BOUNDARY("neumann")

    def u_dirichlet(discr, u):
        dir_u = discr.project("vol", dirichlet_btag, u)
        return TracePair(dirichlet_btag, interior=dir_u, exterior=-dir_u)

    def q_dirichlet(discr, q):
        dir_q = discr.project("vol", dirichlet_btag, q)
        return TracePair(dirichlet_btag, interior=dir_q, exterior=dir_q)

    def u_neumann(discr, u):
        dir_u = discr.project("vol", neumann_btag, u)
        return TracePair(neumann_btag, interior=dir_u, exterior=dir_u)

    def q_neumann(discr, q):
        dir_q = discr.project("vol", neumann_btag, q)
        return TracePair(neumann_btag, interior=dir_q, exterior=-dir_q)

    u_boundaries = {
        dirichlet_btag: u_dirichlet,
        neumann_btag: u_neumann
    }

    q_boundaries = {
        dirichlet_btag: q_dirichlet,
        neumann_btag: q_neumann
    }

    def rhs(t, u):
        return (diffusion_operator(discr, alpha=1, u_boundaries=u_boundaries,
                q_boundaries=q_boundaries, u=u)
            + actx.np.exp(-np.dot(nodes, nodes)/source_width**2))

    rank = comm.Get_rank()

    t = 0
    t_final = 0.01
    istep = 0

    while True:
        if istep % 10 == 0:
            print(istep, t, discr.norm(u))
            vis.write_vtk_file("fld-heat-source-mpi-%03d-%04d.vtu" % (rank, istep),
                    [
                        ("u", u)
                        ])

        if t >= t_final:
            break

        u = rk4_step(u, t, dt, rhs)
        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
