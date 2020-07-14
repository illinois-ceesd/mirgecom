__copyright__ = "Copyright (C) 2020 CEESD Developers"

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
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.wave import wave_operator
from mirgecom.integrators import rk4_step


def bump(discr, queue, t=0):
    source_center = np.array([0., 0., 5.])[:discr.dim]
    source_width = 0.5
    source_omega = 3

    nodes = discr.nodes().with_queue(queue)
    center_dist = join_fields([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * clmath.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def import_pseudo_y0_mesh():
    from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
    mesh = generate_gmsh(
            ScriptWithFilesSource("""
                Merge "pseudoY0.brep";
                Mesh.CharacteristicLengthMin = 1;
                Mesh.CharacteristicLengthMax = 10;
                Mesh.ElementOrder = 2;
                Mesh.CharacteristicLengthExtendFromBoundary = 0;

                // Inside and end surfaces of nozzle/scramjet
                Field[1] = Distance;
                Field[1].NNodesByEdge = 100;
                Field[1].FacesList = {5,7,8,9,10};
                Field[2] = Threshold;
                Field[2].IField = 1;
                Field[2].LcMin = 1;
                Field[2].LcMax = 10;
                Field[2].DistMin = 0;
                Field[2].DistMax = 20;

                // Edges separating surfaces with boundary layer refinement from those without
                // (Seems to give a smoother transition)
                Field[3] = Distance;
                Field[3].NNodesByEdge = 100;
                Field[3].EdgesList = {5,10,14,16};
                Field[4] = Threshold;
                Field[4].IField = 3;
                Field[4].LcMin = 1;
                Field[4].LcMax = 10;
                Field[4].DistMin = 0;
                Field[4].DistMax = 20;

                // Min of the two sections above
                Field[5] = Min;
                Field[5].FieldsList = {2,4};

                Background Field = 5;
                """, ["pseudoY0.brep"]), 3, target_unit='MM')
    return mesh


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 3
    mesh = import_pseudo_y0_mesh()

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(100*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(100*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    fields = join_fields(
            bump(discr, queue),
            [discr.zeros(queue) for i in range(discr.dim)]
            )

    vis = make_visualizer(discr, discr.order+3 if dim == 2 else discr.order)

    vis.write_vtk_file("fld-import-mesh-initial.vtu",
            [
                ("u", fields[0]),
                ("v", fields[1:]),
                ])

#     def rhs(t, w):
#         return wave_operator(discr, c=1, w=w)

#     t = 0
#     t_final = 3
#     istep = 0
#     while t < t_final:
#         fields = rk4_step(fields, t, dt, rhs)

#         if istep % 10 == 0:
#             print(istep, t, la.norm(fields[0].get()))
#             vis.write_vtk_file("fld-wave-eager-%04d.vtu" % istep,
#                     [
#                         ("u", fields[0]),
#                         ("v", fields[1:]),
#                         ])

#         t += dt
#         istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
