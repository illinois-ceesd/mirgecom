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

import os
import math
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import DOFArray, thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_BASE, DTAG_BOUNDARY
from mirgecom.sampling import query_eval
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
from pytools.obj_array import make_obj_array
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

    dim = 3

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(0,0,0),
            b=(10,10,10),
            nelements_per_axis=(3,3,3),
            boundary_tag_to_face={
                "bdy_x": ["+x", "-x"],
                "bdy_y": ["+y", "-y"],
                "bdy_z": ["+z", "-z"]
                }
            )
        n_elem = mesh.nelements
        print("%d elements" % n_elem)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    nodes = thaw(actx, discr.nodes())
    print(nodes)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    def simple_poly(x,y,z):
        return (x**3) + (y**2) + z

    def simple_poly_nodes(nodes):
        return simple_poly(nodes[0], nodes[1], nodes[2])

    u = simple_poly_nodes(nodes)
    qx = 4
    qy = 4
    qz = 4
    
    query_point = np.array([qx, qy, qz])
    u_query = simple_poly(qx,qy,qz)
    tol = 1e-5
    
    q_mapped = query_eval(query_point, actx, discr, dim, tol)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
