"""Demonstrate a 3D periodic box mesh generation."""

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
import pyopencl as cl
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from mirgecom.discretization import create_discretization_collection

from mirgecom.mpi import mpi_entry_point


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None, periodic=None):
    if periodic is None:
        periodic = (False,)*dim

    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,
               periodic=periodic)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_overintegration=False, lazy=False,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=None, use_esdg=False):
    """Drive the example."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    # from mirgecom.simutil import global_reduce as _global_reduce
    # global_reduce = partial(_global_reduce, comm=comm)

    # logmgr = initialize_logmgr(use_logmgr,
    #     filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # some geometry setup
    dim = 2

    left_boundary_location = tuple([0. for _ in range(dim)])
    right_boundary_location = tuple([2*np.pi for _ in range(dim)])
    periodic = (True,)*dim

    n_refine = 1
    pts_per_axis = 16
    npts_axis = tuple([n_refine * pts_per_axis for _ in range(dim)])
    # npts_axis = (npts_x, npts_y)
    box_ll = left_boundary_location
    box_ur = right_boundary_location
    generate_mesh = partial(_get_box_mesh, dim=dim, a=box_ll, b=box_ur, n=npts_axis,
                            periodic=periodic)
    print(f"{left_boundary_location=}")
    print(f"{right_boundary_location=}")
    print(f"{npts_axis=}")
    from mirgecom.simutil import generate_and_distribute_mesh
    local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                generate_mesh)
    local_nelements = local_mesh.nelements

    # from meshmode.mesh.processing import rotate_mesh_around_axis
    # local_mesh = rotate_mesh_around_axis(local_mesh, theta=-np.pi/4)

    order = 1
    dcoll = create_discretization_collection(actx, local_mesh, order=order,
                                             quadrature_order=order+2)
    nodes = actx.thaw(dcoll.nodes())

    print(f"{rank=}/{nparts=}")
    print(f"{global_nelements=}")
    print(f"{local_nelements=}")
    print(f"{nodes=}")


if __name__ == "__main__":
    import argparse
    casename = "poiseuille"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from warnings import warn
    if args.esdg:
        if not args.lazy:
            warn("ESDG requires lazy-evaluation, enabling --lazy.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    lazy = args.lazy or args.esdg
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
         use_overintegration=args.overintegration or args.esdg, lazy=lazy,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class,
         use_esdg=args.esdg)

# vim: foldmethod=marker
