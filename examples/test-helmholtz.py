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
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.symbolic.primitives import QTAG_NONE
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
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
            a=(-np.pi/2,)*dim,
            b=(np.pi/2,)*dim,
            n=(nel_1d,)*dim,
            boundary_tag_to_face={
                "bdy_x": ["+x", "-x"],
                "bdy_y": ["+y", "-y"]
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
        dt = 0.0008/(order**3)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    nodes = thaw(actx, discr.nodes())

    u = discr.zeros(actx)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    def sx2(actx, discr):
        nodes = thaw(actx, discr.nodes(dd=grudge_sym.DTAG_BOUNDARY("bdy_y")))
        return (actx.np.sin(nodes[0])) ** 2
    def cy2(actx, discr):
        nodes = thaw(actx, discr.nodes(dd=grudge_sym.DTAG_BOUNDARY("bdy_x")))
        return (actx.np.cos(nodes[1])) ** 2

    def source(actx, discr):
        a = (actx.np.sin(nodes[0]) * actx.np.cos(nodes[1])) ** 2
        b = 2 * actx.np.sin(nodes[0]) * actx.np.sin(nodes[0]) * actx.np.cos(2*nodes[1])
        c = - 2 * actx.np.cos(nodes[1]) * actx.np.cos(nodes[1]) * actx.np.cos(2*nodes[0])
        return a + b + c
    
    boundaries = {
        grudge_sym.DTAG_BOUNDARY("bdy_x"): NeumannDiffusionBoundary(0.),
        grudge_sym.DTAG_BOUNDARY("bdy_y"): DirichletDiffusionBoundary(0.)  
    }

    def rhs(t, u):
        return (diffusion_operator(discr, quad_tag=QTAG_NONE, alpha=1.0, 
            boundaries=boundaries, u=u) - u + source(actx, discr))

# nabla^2(E) - E - S(x,y) -> 0

    rank = comm.Get_rank()

    t = 0
    t_final = 2
    istep = 0

    while True:
        if istep % 10 == 0:
            print(istep, t, discr.norm(u))
            vis.write_vtk_file("fld-helmholtz-%03d-%04d.vtu" % (rank, istep),
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
