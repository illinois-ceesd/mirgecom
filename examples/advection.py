"""Advection serial example."""

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
import pyopencl.array as cla  # noqa
from grudge.eager import EagerDGDiscretization
from mirgecom.integrators import rk4_step
from meshmode.array_context import PyOpenCLArrayContext
import pyopencl.tools as cl_tools

from grudge import bind, sym

from mirgecom.profiling import PyOpenCLProfilingArrayContext


def main(use_profiling=False):
    """Drive the example."""
    cl_ctx = cl.create_some_context()
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # domain [-d/2, d/2]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 20
    # grid spacing
    h = d / npoints

    dim = 2
    order = 3

    # cfl
    dt_factor = 2.0
    # final time
    t_final = 1.0
    # compute number of steps
    dt = dt_factor * h/order**2
    nsteps = int(t_final // dt) + 1
    dt = t_final/nsteps + 1.0e-15

    # velocity field
    c = np.array([0.5] * dim)
    norm_c = la.norm(c)
    # flux
    flux_type = "central"

    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(
        [np.linspace(-d/2, d/2, npoints) for _ in range(dim)],
        order=order)

    discr = EagerDGDiscretization(actx, mesh, order=order)

    print(f"{mesh.nelements} elements, dim={dim}, order={order}")

    def f(x):
        return sym.sin(3 * x)

    def u_analytic(x):
        t = sym.var("t", sym.DD_SCALAR)
        return f(-np.dot(c, x) / norm_c + t * norm_c)

    from grudge.models.advection import WeakAdvectionOperator
    op = WeakAdvectionOperator(c,
        inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
        flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, u_analytic(sym.nodes(dim)))(actx, t=0)

    def rhs(t, u):
        return bound_op(t=t, u=u)

    istep = 0
    t = 0

    while t < t_final:
        u = rk4_step(u, t, dt, rhs)

        if istep % 10 == 0:
            if use_profiling:
                print(actx.tabulate_profiling_data())
            print(istep, t)

        t += dt
        istep += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wave-eager (non-MPI version)")
    parser.add_argument("--profile", action="store_true",
        help="enable kernel profiling")
    args = parser.parse_args()

    main(use_profiling=args.profile)
