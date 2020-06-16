__copyright__ = (
    "Copyright (C) 2020 University of Illinos Board of Trustees"
)

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
import pyopencl.clmath as clmath
from pytools.obj_array import join_fields

# TODO: Remove grudge dependence?
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.wave import wave_operator
from mirgecom.euler import inviscid_operator
from mirgecom.euler import Lump
from mirgecom.euler import Vortex2D
from mirgecom.euler import BoundaryBoss
from mirgecom.integrators import rk4_step
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


def bump(discr, queue, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[: discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = discr.nodes().with_queue(queue)
    center_dist = join_fields(
        [nodes[i] - source_center[i] for i in range(discr.dim)]
    )

    return np.cos(source_omega * t) * clmath.exp(
        -np.dot(center_dist, center_dist) / source_width ** 2
    )


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75 / (nel_1d * order ** 2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45 / (nel_1d * order ** 2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)

    #    initializer = Vortex2D()
    initializer = Lump()
    initializer.SetBoundaryTag(BTAG_ALL)
    boundaryboss = BoundaryBoss()
    boundaryboss.AddBoundary(initializer)

    fields = initializer(0, nodes)

    vis = make_visualizer(
        discr, discr.order + 3 if dim == 2 else discr.order
    )

    def rhs(t, w):
        return inviscid_operator(
            discr, w=w, t=t, boundaries=boundaryboss
        )

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, inviscid_operator)

        if istep % 10 == 0:
            print(istep, t, la.norm(fields[0].get()))
            vis.write_vtk_file(
                "fld-euler-eager-%04d.vtu" % istep,
                [
                    ("rho", fields[0]),
                    ("rhoe", fields[1]),
                    ("rhov", fields[2:]),
                ],
            )

        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
