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
from mirgecom.sampling import _find_src_unit_nodes
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
        from meshmode.mesh.io import read_gmsh
        mesh = read_gmsh("/Users/dshtey2/Documents/TT/CADs/act2_noholes.msh")
        
        n_elem = mesh.nelements
        print("%d elements" % n_elem)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        mesh_grp = mesh.groups

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 1

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    if dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 2.5e-4
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
            + 120.0/3*nodes[2] + 200.0/3)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    # Generate alpha field
    ramp_region_bit = local_mesh.region_tag_bit("Ramp_Vol")
    samp_region_bit = local_mesh.region_tag_bit("Samp_Vol")

    alpha = discr.empty(actx)
    alpha_np = [actx.to_numpy(alpha_i) for alpha_i in alpha]

    for igrp, grp in enumerate(local_mesh.groups):
        ramp_elems, = np.where((grp.regions & ramp_region_bit) != 0)
        samp_elems, = np.where((grp.regions & samp_region_bit) != 0)
        alpha_np[igrp][ramp_elems, :] = 4.2
        alpha_np[igrp][samp_elems, :] = 0.5

    alpha = DOFArray(actx, tuple([
        actx.from_numpy(alpha_np_i) for alpha_np_i in alpha_np]))
    
    # Surface Boundary condition functions
    def smooth_elliptic_jump(t,TL,TR,tl,tr):
        if TR == TL:
            return TR
        else:
            tm = 0.5*(tr + tl)
            if t < tm:
                return 0.5*(TL+TR + (TL-TR)*np.sqrt(1 - 4 * ((t-tl)/(tr-tl)) ** 2))
            else:
                return 0.5*(TL+TR - (TL-TR)*np.sqrt(1 - 4 * ((t-tr)/(tr-tl)) ** 2))

    def cube_left(t,TL,TR,tl,tr):
        a = 4 * (TL - TR)
        b = -(12 * tl * (TL - TR))
        c = 12 * (tl ** 2) * (TL - TR)
        d = -(3*TL*(tl**3) + TL*(tr**3) - 4*TR*(tl**3) - 3*TL*tl*tr*tr + 3*TL*tl*tl*tr)

        return 1/((tl-tr)**3)*(a*(t**3) + b*(t**2) + c*t + d)

    def cube_right(t,TL,TR,tl,tr):
        a = 4 * (TL - TR)
        b = -(12 * tr * (TL - TR))
        c = 12 * (tr ** 2) * (TL - TR)
        d = TR*(tl**3) - 4*TL*(tr**3) + 3*TR*(tr**3) + 3*TR*tl*(tr**2) - 3*TR*(tl**2)*tr

        return 1/((tl-tr)**3)*(a*(t**3) + b*(t**2) + c*t + d)

    def smooth_cubic_jump(t,TL,TR,tl,tr):
        if TR == TL:
            return TR
        else:
            tm = 0.5*(tr + tl)
            if t < tm:
                return cube_left(t,TL,TR,tl,tr)
            else:
                return cube_right(t,TL,TR,tl,tr)


    def flame_pulse(t,TL,TU,tl1,tr1,tl2,tr2):
        if t < tl1:
            return TL
        elif t < tr1:
            return smooth_cubic_jump(t,TL,TU,tl1,tr1)
        elif t < tl2:
            return TU
        elif t < tr2:
            return smooth_cubic_jump(t,TU,TL,tl2,tr2)
        else:
            return TL
    
    def act2_flame(t):
        return flame_pulse(t,300,1500,0.5,1,2,4)

    def act2_test(t):
        tol = 0.25
        b = 1/0.3 * np.log(1200.0/(tol*1500))
        A = tol*1500*np.exp(b*1.3)
        if t < 1:
            return 300.0
        elif t < 2:
            return 1500.0 - A*np.exp(-b*t)
        else:
            T0 = 1500.0 - A*np.exp(-b*2)
            return 300.0 + (T0-300.0)*np.exp(-(t-2)/0.91)

    def flame_tanh(t,TL,TU,dT,tr,tu):
        tm = 0.5*tr
        TH = 0.5*(TL + TU)
        a = 4.1033099028 #solve sinh(2at)/a = (tu-tr)*(TH-TL-dT)/(2*dT)
        A = (TH - TL - dT)/np.tanh(a*tm)

        if t - 0.5 < tr:
            return TH - dT + A*np.tanh(a*((t-0.5) - tm))
        elif t - 0.5 < tr*1.5:
            return TU - 2*dT * ((((t-0.5)-tu)/(tr-tu))**2)
        else:
            return TH - dT + A*np.tanh(a*(4*tm - (t-0.5)))


    def act2_tanh(t):
        return flame_tanh(t,300.0,1500.0,10,1.0,1.25)

    #act2_boundaries = {
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Embedding"): DirichletDiffusionBoundary(300),
    #        grudge_sym.DTAG_BOUNDARY("Ramp_Outer"): DirichletDiffusionBoundary(300),
    #        grudge_sym.DTAG_BOUNDARY("Samp_Outer"): DirichletDiffusionBoundary(300),
    #        grudge_sym.DTAG_BOUNDARY("Lines"): DirichletDiffusionBoundary(300)
    #    }
    

    # Run diffusion operator
    def rhs(t, u):
        #ramp_boundaries = {
        #grudge_sym.DTAG_BOUNDARY("Embedding"): DirichletDiffusionBoundary(200),
        #grudge_sym.DTAG_BOUNDARY("Insert"): DirichletDiffusionBoundary(200),
        #grudge_sym.DTAG_BOUNDARY("Outer"): DirichletDiffusionBoundary(200)
    #}
        # Set boundary conditions
        TL = 1500.0
        TR = 300.0
        tl = 1.0
        tr = 2.0
        act2_boundaries = {
            DTAG_BOUNDARY("Ramp_Embedding"): DirichletDiffusionBoundary(act2_tanh(0)),
            DTAG_BOUNDARY("Ramp_Outer"): DirichletDiffusionBoundary(act2_tanh(t)),
            DTAG_BOUNDARY("Samp_Outer"): DirichletDiffusionBoundary(act2_tanh(t))
        }
        return diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=alpha, 
            boundaries=act2_boundaries, u=u)

    rank = comm.Get_rank()

    u = discr.zeros(actx) + act2_tanh(0)

    t = 0
    t_final = dt
    istep = 0

    while True:

        if istep % 50 == 0:
            print(istep, t, discr.norm(u))
            vis.write_vtk_file("29-fld-act2-bdy-%03d-%04d.vtu" % (rank, istep),
                    [
                        ("u", u),
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
