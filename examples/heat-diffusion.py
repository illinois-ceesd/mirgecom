"""Demonstrate orthotropic heat diffusion example."""

__copyright__ = "Copyright (C) 2023 University of Illinois Board of Trustees"

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
import grudge.op as op
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import diffusion_operator, DirichletDiffusionBoundary
from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation

from mirgecom.logging_quantities import (initialize_logmgr,
                                         logmgr_add_cl_device_info,
                                         logmgr_add_device_memory_usage)

from logpyle import IntervalTimer, set_dt


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_overintegration=False,
         use_leap=False, casename=None, rst_filename=None):
    """Run the example."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    logmgr = initialize_logmgr(True,
        filename="heat-diffusion.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    viz_path = "viz_data/"
    vizname = viz_path+casename

    dim = 2
    nel_x = 61
    nel_y = 31

    t = 0
    t_final = 0.00025
    istep = 0

    factor = 4.0
    gaussian_width = 0.1
    _kappa = np.zeros(2,)
    _kappa[1] = 0.5
    _kappa[0] = factor*_kappa[1]

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-1.0*np.sqrt(factor), -1.0),
            b=(+1.0*np.sqrt(factor), +1.0),
            nelements_per_axis=(nel_x, nel_y),
            boundary_tag_to_face={
                "x": ["+x", "-x"],
                "y": ["+y", "-y"]
                }
            )

        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 2

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None  # noqa

    if dim == 2:
        # no deep meaning here, just a fudge factor
        # this is significantly smaller than the maximum allowable CFL
        dt = 0.05/(nel_x*order)**2
        print(dt)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    nodes = actx.thaw(dcoll.nodes())

    kappa = force_evaluation(actx, _kappa + nodes[0]*0.0)

    boundaries = {
        BoundaryDomainTag("x"): DirichletDiffusionBoundary(0.),
        BoundaryDomainTag("y"): DirichletDiffusionBoundary(0.),
    }

    r2 = np.dot(nodes/actx.np.sqrt(kappa), nodes/actx.np.sqrt(kappa))
    u = dcoll.zeros(actx) + 30.0*actx.np.exp(-r2/(2.0*gaussian_width**2))

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches(["step.max", "t_step.max", "t_log.max"])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

    def rhs(t, u):
        return diffusion_operator(dcoll, kappa=kappa, boundaries=boundaries,
                                  u=u, quadrature_tag=quadrature_tag)

    compiled_rhs = actx.compile(rhs)

    from mirgecom.simutil import write_visfile
    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        if istep % 10 == 0:
            amp = 30.0/(1.0 + 2.0*t/gaussian_width**2)
            r2 = np.dot(nodes/np.sqrt(_kappa), nodes/np.sqrt(_kappa))
            exact = amp*actx.np.exp(-r2/(2.0*gaussian_width**2 + 4.0*t))
            error = u - exact
            viz_fields = [
                ("u", u),
                ("exact", exact),
                ("error", error),
                ("kappa_x", kappa[0]),
                ("kappa_y", kappa[1]),
            ]
            write_visfile(dcoll, viz_fields, visualizer,
                          vizname=vizname+"heat-diffusion", step=istep, t=t,
                          overwrite=True, comm=comm)

        u = rk4_step(u, t, dt, compiled_rhs)
        u = force_evaluation(actx, u)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    final_error = actx.to_numpy(op.norm(dcoll, error, np.inf))
    print(1.0*np.sqrt(factor)/nel_x, 1.0/nel_y, final_error)

    if logmgr:
        logmgr.close()


if __name__ == "__main__":
    import argparse
    casename = "heat-diffusion"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--overintegration", action="store_true",
        help="turn on overintegration.")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profiling, numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_leap=args.leap,
         use_overintegration=args.overintegration,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
