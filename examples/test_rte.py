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
from meshmode.discretization.connection import (
    NodalToModalDiscretizationConnection as nod2mod,
    ModalToNodalDiscretizationConnection as mod2nod)
from meshmode.discretization import Discretization

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_BASE, DTAG_BOUNDARY
from mirgecom.integrators import rk4_step, rk4_step_grad
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary,
    RobinDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

#from matplotlib import pyplot as plt


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
    order_des = 2
    err_inf = []
    nx_plot = []

    for i in range(1,2):
        nz = 10
        tau_L = float(i)
        tau_max = 0.5*tau_L

        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                a=(0,0,-tau_max),
                b=(10,10,tau_max),
                nelements_per_axis=(3,3,nz),
                boundary_tag_to_face={
                    "bdy_x": ["+x", "-x"],
                    "bdy_y": ["+y", "-y"],
                    "bdy_z+": ["+z","-z"]
                    }
                )

            print("%d elements" % mesh.nelements)

            part_per_element = get_partition_by_pymetis(mesh, num_parts)

            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()
        # converge on successive orders of elements
        for ord in range(order_des, order_des+1):
            order = ord
            
            discr = EagerDGDiscretization(actx, local_mesh, order=order,
                        mpi_communicator=comm)

            nodes = thaw(actx, discr.nodes())

            def gam_coeff(omega, A1):
                return np.sqrt( (1.0-omega) * (3-A1*omega) )

            T0 = 300.0
            Tw = 600.0
            SB = 5.67e-8
            for A1 in range(1,2):
                Alab = A1 + 2
                for om in range(1,2):
                    omega = om/2
                    gamma = gam_coeff(omega,A1)

                    def Ibw(n,T):
                        return 4 * (n**2) * SB * (T**4)
                    def s_G(gam,n,T):
                        return -(gam**2) * Ibw(n,T)
                    param = 0.5*(3-A1*omega)
                    source_G = s_G(gamma,1.0,T0)

                    boundaries = {
                        DTAG_BOUNDARY("bdy_x"): NeumannDiffusionBoundary(0.),
                        DTAG_BOUNDARY("bdy_y"): NeumannDiffusionBoundary(0.),
                        DTAG_BOUNDARY("bdy_z+"): RobinDiffusionBoundary(param,param*Ibw(1.0,Tw))
                    }

                    def G_analytical(actx, discr):
                        tau = nodes[2]
                        C_htrig = 1/(np.cosh(gamma*tau_max) + 2*gamma/(3 - omega*A1)*np.sinh(gamma*tau_max))
                        return Ibw(1.0,T0) + (Ibw(1.0,Tw) - Ibw(1.0,T0))*C_htrig*actx.np.cosh(gamma * tau)

                    def dG_analytical(actx, discr):
                        tau = nodes[2]
                        C_htrig = 1/(np.cosh(gamma*tau_max) + 2*gamma/(3 - omega*A1)*np.sinh(gamma*tau_max))
                        return gamma*(Ibw(1.0,Tw) - Ibw(1.0,T0))*C_htrig*actx.np.sinh(gamma * tau)

                    def heatflux_analytical(actx, discr):
                        tau = nodes[2]
                        return 2*actx.np.sinh(gamma*tau)/(np.sinh(gamma*tau_max) + 
                            0.5*np.sqrt((3 - A1*omega)/(1 - omega))*np.cosh(gamma*tau_max))
                    def heatflux_wall(tau_w, A1, omega):
                        return 2*np.sinh(gamma*tau_w)/(np.sinh(gamma*tau_max) + 
                            0.5*np.sqrt((3 - A1*omega)/(1 - omega))*np.cosh(gamma*tau_max))

                    if order == 1:
                        u_0 = G_analytical(actx,discr)#discr.zeros(actx) + 18000.0

                    if dim == 3:
                        dt = (np.sqrt(10)**(3-order))*1e-5
                    else:
                        raise ValueError("don't have a stable time step guesstimate")

                    vis = make_visualizer(discr, order+3 if dim == 2 else order)

                #def sx2(actx, discr):
                #    nodes = thaw(actx, discr.nodes(dd=grudge_sym.DTAG_BOUNDARY("bdy_y")))
                #    return (actx.np.sin(nodes[0])) ** 2
                #def cy2(actx, discr):
                #    nodes = thaw(actx, discr.nodes(dd=grudge_sym.DTAG_BOUNDARY("bdy_x")))
                #    return (actx.np.cos(nodes[1])) ** 2

                    def source(actx, discr):
                        #a = (actx.np.sin(nodes[0]) * actx.np.cos(nodes[1])) ** 2
                        #b = 2 * actx.np.sin(nodes[0]) * actx.np.sin(nodes[0]) * actx.np.cos(2*nodes[1])
                        #c = - 2 * actx.np.cos(nodes[1]) * actx.np.cos(nodes[1]) * actx.np.cos(2*nodes[0])
                        return 4*actx.np.sin(nodes[0]) * actx.np.cos(nodes[1]) * actx.np.sin(nodes[2])

                    def rhs(t, u):
                        #diff_u, grad_u = diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=1.0, 
                            #boundaries=boundaries, u=u, return_grad_u=True) # GRAD_U TESTING
                        return (diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=1.0, 
                            boundaries=boundaries, u=u, return_grad_u=False) - gamma*gamma * u - source_G)

                # nabla^2(E) - E - S(x,y) -> 0

                    rank = comm.Get_rank()

                    u = discr.zeros(actx) + 18000.0
                    t = 0
                    istep = 0

                    u_analytical = G_analytical(actx, discr)
                    #du_analytical = dG_analytical(actx, discr) # GRAD_U TESTING
                    u_new = u
                    u_old = u + 20000
                    q_analytical = heatflux_analytical(actx, discr)
                    #grad_u = du_analytical # GRAD_U TESTING

                    while True:
                        u_old = u_new

                        if istep % 500 == 0:
                            #discr_mod = discr.discr_from_dd("vol")
                            #print(discr_mod.groups[0].basis_obj().functions[0])
                            print(istep, t, discr.norm(u_old))
                            diff_u, grad_u = diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=1.0, 
                                boundaries=boundaries, u=u_old, return_grad_u=True)

                            q_nondim = grad_u[2]/(( Ibw(1.0,Tw) - Ibw(1.0,T0) )/4 * (3 - omega*A1))
                            q_err = q_nondim - q_analytical

                            vis.write_vtk_file("fld-rte-2q_test-10-%03d-%03d-%04d.vtu" % (Alab, om, istep),
                                [
                                    ("u", u_old),
                                    ("u_err", u_old - u_analytical),
                                    ("q", q_nondim),
                                    ("q_err", q_err)
                                    ])

                        u_new = rk4_step(u_old, t, dt, rhs)
                        t += dt
                        istep += 1

                        if discr.norm(u_old - u_new)/discr.norm(u_old) < 1e-7 and istep > 500:
                            u_0 = u_old
                            diff_u, grad_u = diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=1.0, 
                                boundaries=boundaries, u=u_old, return_grad_u=True)

                            q_nondim = grad_u[2]/(( Ibw(1.0,Tw) - Ibw(1.0,T0) )/4 * (3 - omega*A1))
                            q_err = q_nondim - q_analytical

                            vis.write_vtk_file("soln-rte-2q_test-10-%03d-%03d-%04d.vtu" % (Alab, om, i),
                                [
                                    ("u", u_old),
                                    ("u_err", u_old - u_analytical),
                                    ("q", q_nondim),
                                    ("q_err", q_err)
                                    ])
                            break
            
    #err = np.abs(np.max(u_old - u_analytical))
    #err_inf.append(err)
    #print(err_inf)
    #plt.plot(nx_plot,err_inf)

### Find query point elements routine

### Interpolation Routine Here
qpoint_des = find_src_unit_nodes(tgt_bdy_nodes, src_bdy_nodes, src_grp, tol)
###

if __name__ == "__main__":
    main()

# vim: foldmethod=marker
