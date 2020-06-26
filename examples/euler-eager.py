__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__author__ = """
Center for Exascale-Enabled Scramjet Design
University of Illinois, Urbana, IL 61801
"""

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

# TODO: Remove grudge dependence?
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.euler import get_inviscid_timestep
from mirgecom.euler import inviscid_operator
from mirgecom.initializers import Lump
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas
from mirgecom.integrators import rk4_step
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "EaEu] "

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-5.0,) * dim, b=(5.0,) * dim, n=(nel_1d,) * dim
    )

    order = 3

    t = 0
    t_final = 0.1
    istep = 0

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    nodes = discr.nodes().with_queue(queue)

    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[0] = 1.0
    vel[1] = 1.0

    casename = 'Vortex'
    if casename == 'Vortex':
        initializer = Vortex2D(center=orig, velocity=vel)
    elif casename == 'Lump':
        initializer = Lump(center=orig, velocity=vel)
    else:
        print(f"{iotag}Error: Unknown init case ({casename})")
        assert(False)

    boundaries = {BTAG_ALL: initializer}
    eos = IdealSingleGas()

    fields = initializer(0, nodes)

    cfl = 1.0
    sdt = get_inviscid_timestep(discr, fields, c=cfl, eos=eos)
    constantcfl = False
    dt = 0.001
    nstep_status = 10

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    message = (
        f"{iotag}Num {dim}d elements: {mesh.nelements}\n"
        f"{iotag}Timestep:        {dt}\n"
        f"{iotag}Final time:      {t_final}\n"
        f"{iotag}Status freq:     {nstep_status}\n"
        f"{iotag}Initialization:  {initname}\n"
        f"{iotag}EOS:             {eosname}"
    )

    print(message)
    vis = make_visualizer(
        discr, discr.order + 3 if dim == 2 else discr.order
    )

    def write_soln():
        expected_result = initializer(t, nodes)
        result_resid = fields - expected_result
        maxerr = [
            np.max(np.abs(result_resid[i].get()))
            for i in range(dim + 2)
        ]

        dv = eos(fields)
        mindv = [np.min(dvfld.get()) for dvfld in dv]
        maxdv = [np.max(dvfld.get()) for dvfld in dv]

        statusmsg = (
            f"{iotag}Status: Step({istep}) Time({t})\n"
            f"{iotag}------   P({mindv[0]},{maxdv[0]})\n"
            f"{iotag}------   T({mindv[1]},{maxdv[1]})\n"
            f"{iotag}------   dt,cfl = ({dt},{cfl})\n"
            f"{iotag}------   Err({maxerr})"
        )
        print(statusmsg)

        vis.write_vtk_file(
            "fld-euler-eager-%04d.vtu" % istep,
            [
                ("density", fields[0]),
                ("energy", fields[1]),
                ("momentum", fields[2:]),
                ("pressure", dv[0]),
                ("temperature", dv[1]),
                ("expected_solution", expected_result),
                ("residual", result_resid),
            ],
        )

    def rhs(t, w):
        return inviscid_operator(
            discr, w=w, t=t, boundaries=boundaries, eos=eos
        )

    while t < t_final:

        if constantcfl is True:
            dt = sdt
        else:
            cfl = dt / sdt

        if istep % nstep_status == 0:
            write_soln()

        fields = rk4_step(fields, t, dt, rhs)
        t += dt
        istep += 1

        sdt = get_inviscid_timestep(discr, fields, c=cfl, eos=eos)

    print(f"{iotag}Writing final dump.")
    write_soln()

    print(f"{iotag}Goodbye!")


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
