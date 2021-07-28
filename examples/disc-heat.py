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

    dim = 3

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.io import read_gmsh
        mesh = read_gmsh("/Users/dshtey2/Documents/Disc_Coeff/CADs/rod_dual.msh")
        
        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    if dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.005/(order**3)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    nodes = thaw(actx, discr.nodes())

    def sinus(actx, discr, t=0):
        nodes = thaw(actx, discr.nodes())
        return (
            actx.np.sin(math.pi/5 * nodes[2])
            )

    def steady(actx, discr, t=0):
        nodes = thaw(actx, discr.nodes())
        return (
            -40.0/3*abs(nodes[2]) 
            + 120.0/3*nodes[2] + 200.0/3
            )

    u = steady(actx, discr)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    # Generate alpha field
    left_region_bit = local_mesh.region_tag_bit("LeftHalf")
    right_region_bit = local_mesh.region_tag_bit("RightHalf")

    alpha = discr.empty(actx)
    alpha_np = [actx.to_numpy(alpha_i) for alpha_i in alpha]

    for igrp, grp in enumerate(local_mesh.groups):
        left_elems, = np.where((grp.regions & left_region_bit) != 0)
        right_elems, = np.where((grp.regions & right_region_bit) != 0)
        alpha_np[igrp][left_elems, :] = 1
        alpha_np[igrp][right_elems, :] = 0.5

    alpha = DOFArray(actx, tuple([
        actx.from_numpy(alpha_np_i) for alpha_np_i in alpha_np]))

    def flame_pulse(t):
        if t < 0.02:
            return 0.0
        elif t < 0.08:
            return 1200.0
        else:
            return 1200.0*math.exp(-t)
    
    #act2_boundaries = {
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Embedding"): DirichletDiffusionBoundary(0.),
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Outer"): DirichletDiffusionBoundary(0.01),
    #        grudge_sym.DTAG_BOUNDARY("Samp_Outer"): DirichletDiffusionBoundary(0.)
    #    }
    rod_boundaries = {
        grudge_sym.DTAG_BOUNDARY("LeftLat"): NeumannDiffusionBoundary(0.),
        grudge_sym.DTAG_BOUNDARY("RightLat"): NeumannDiffusionBoundary(0.),
        grudge_sym.DTAG_BOUNDARY("RightCap"): DirichletDiffusionBoundary(200.),
        grudge_sym.DTAG_BOUNDARY("LeftCap"):DirichletDiffusionBoundary(-200.)
    }


    # Run diffusion operator
    def rhs(t, u):
        # Set boundary conditions
        #act2_boundaries = {
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Embedding"): DirichletDiffusionBoundary(0.),
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Outer"): DirichletDiffusionBoundary(0.01),
    #        grudge_sym.DTAG_BOUNDARY("Samp_Outer"): DirichletDiffusionBoundary(0.)
    #    }
        return diffusion_operator(discr, quad_tag=QTAG_NONE, alpha=alpha, boundaries=rod_boundaries, u=u)

    rank = comm.Get_rank()

    t = 0
    t_final = 15
    istep = 0

    while True:
        if istep % 10 == 0:
            print(istep, t, discr.norm(u))
            vis.write_vtk_file("fld-heat-source-mpi-%03d-%04d.vtu" % (rank, istep),
                    [
                        ("u", u + 300.0),
                        ("a", alpha)
                        ])

        if t >= t_final:
            break

        u = rk4_step(u, t, dt, rhs)
        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
