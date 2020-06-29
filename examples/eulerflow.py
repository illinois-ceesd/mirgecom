__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
import logging
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
from mpi4py import MPI
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.euler import get_inviscid_timestep
from mirgecom.euler import inviscid_operator
from mirgecom.euler import split_fields
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import Lump
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas
from mirgecom.integrators import rk4_step
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw


mirge_params = {

    "numdim": 2,
    "nel_1d": 16,
    "box_lower_left": -5,
    "box_upper_right": 5,
    "order": 3,
    "origin": (0, 0, 0),
    "velocity": (0, 0.0),
    "t_final": 0.1,
    "cfl": 1.0,
    "dt": 0.001,
    "nstatus": 10,
    "constantcfl": False,
    "casename": "Vortex",
}


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = mirge_params["numdim"]
    nel_1d = mirge_params["nel_1d"]
    box_ll = mirge_params["box_lower_left"]
    box_ur = mirge_params["box_upper_right"]
    order = mirge_params["order"]
    t_final = mirge_params["t_final"]
    cfl = mirge_params["cfl"]
    dt = mirge_params["dt"]
    constantcfl = mirge_params["constant_cfl"]
    nstepstatus = mirge_params["nstatus"]
    casename = mirge_params["casename"]
    visfileroot = casename

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    num_parts = nproc

    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh

        mesh = generate_regular_rect_mesh(
            a=(box_ll,) * dim, b=(box_ur,) * dim, n=(nel_1d,) * dim
        )

        logging.info(f"Total {dim}d elements: {mesh.nelements}")

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(
            mesh, part_per_element, num_parts
        )
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    t = 0
    istep = 0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    
    j = 0
    for veli in mirge_params["velocity"]:
        vel[j] = veli
        j += 1

    j = 0
    for origi in mirge_params["origin"]:
        orig[j] = origi
        j += 1

    if casename == "Vortex":
        initializer = Vortex2D(center=orig, velocity=vel)
    elif casename == "Lump":
        initializer = Lump(center=orig, velocity=vel)
    else:
        logging.error(f"Error: Unknown init case ({casename})")
        assert False

    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    eos = IdealSingleGas()

    fields = initializer(0, nodes)

    # todo: needs parallelization, maybe just reduction
    sdt = get_inviscid_timestep(discr, fields, c=cfl, eos=eos)

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__

    # todo: should report min/max num elements / partition over procs
    message = (
        f"Num partitions:      {nproc}\n"
        f"Num {dim}d elements: {local_mesh.nelements}\n"
        f"Timestep:            {dt}\n"
        f"Final time:          {t_final}\n"
        f"Status freq:         {nstepstatus}\n"
        f"Initialization:      {initname}\n"
        f"EOS:                 {eosname}"
    )
    comm.Barrier()

    if rank == 0:
        logging.info(message)

    vis = make_visualizer(discr, discr.order + 3 if dim == 2 else discr.order)

    def write_soln():
        expected_result = initializer(t, nodes)
        result_resid = fields - expected_result
        maxerr = [np.max(np.abs(result_resid[i].get())) for i in range(dim + 2)]

        dv = eos(fields)
        mindv = [np.min(dvfld.get()) for dvfld in dv]
        maxdv = [np.max(dvfld.get()) for dvfld in dv]

        statusmsg = (
            f"Status: Step({istep}) Time({t})\n"
            f"------   P({mindv[0]},{maxdv[0]})\n"
            f"------   T({mindv[1]},{maxdv[1]})\n"
            f"------   dt,cfl = ({dt},{cfl})\n"
            f"------   Err({maxerr})"
        )

        if rank == 0:  # todo: need parallel status
            logging.info(statusmsg)


        visfilename = visfileroot + "-{iorank:04d}-{iostep:04d}.vtu"
        visfilename.Format(iorank=rank, iostep=istep)

        io_fields = split_fields(dim, fields)
        io_fields += eos.split_fields(dim, dv)
        io_fields.append(("exact_soln", expected_result))
        io_fields.append(("residual", result_resid))
        vis.write_vtk_file("fld-euler-eager-%04d.vtu" % istep, io_fields)

    def rhs(t, w):
        return inviscid_operator(discr, w=w, t=t, boundaries=boundaries, eos=eos)

    while t < t_final:

        if constantcfl is True:
            dt = sdt
        else:
            cfl = dt / sdt

        if istep % nstepstatus == 0:
            write_soln()

        fields = rk4_step(fields, t, dt, rhs)
        t += dt
        istep += 1

        # todo: reduce min over ranks
        sdt = get_inviscid_timestep(discr, fields, c=cfl, eos=eos)

        comm.Barrier()

    if rank == 0:
        logging.info("Writing final dump.")
    write_soln()

    comm.Barrier()

    if rank == 0:
        logging.info("Goodbye!")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
