r"""Demonstrate Ablation Workshop case \#2.1."""

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
import pyopencl as cl

from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection

from mirgecom.integrators.ssprk import ssprk43_step

from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary,
    PrescribedFluxDiffusionBoundary
)

from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh,
    write_visfile
)

from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation

import logging
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage
)

from pytools.obj_array import make_obj_array

import sys  # noqa

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         restart_file=None):
    """.XXX."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename="ablation.sqlite", mode="wo", mpi_comm=comm)

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
        actx = actx_class(comm, queue, allocator=alloc,
                          force_device_scalars=True)

    viz_path = "viz_data/"
    vizname = viz_path+casename

    t_final = 60.0

    order = 2
    dt = 1.0e-7
    pressure_scaling_factor = 0.1  # noqa N806

    dt = dt/pressure_scaling_factor

    nviz = 1000
    ngarbage = 100
    nrestart = 10000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )
    if restart_file:  # read the grid from restart data
        rst_filename = f"{restart_file}"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts

    else:  # generate the grid from scratch
        from functools import partial
        box_ll = (-.0025, 0.0)
        box_ur = (+.0025, .05)
        num_elements = (5+1, 100+1)

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                periodic=(True, False),
                                boundary_tag_to_face={"prescribed": ["+y"],
                                                      "neumann": ["-y"]})
        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_BASE

    quadrature_tag = DISCR_TAG_BASE

    nodes = actx.thaw(dcoll.nodes())

#    zeros = nodes[0]*0.0

    from grudge.dof_desc import DD_VOLUME_ALL
    dd_vol = DD_VOLUME_ALL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#    from grudge.op import nodal_min, nodal_max

#    def vol_min(x):
#        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

#    def vol_max(x):
#        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

#    from grudge.dt_utils import characteristic_lengthscales
#    length_scales = characteristic_lengthscales(actx, dcoll, dd=dd_vol)

#    h_min = vol_min(length_scales)
#    h_max = vol_max(length_scales)

    if rank == 0:
        print("----- Discretization info ----")
        #  print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#    # ablation workshop case #1.0
#    def my_presc_func(**kwargs):
#        time = kwargs['time']

#        if time >= 0.0 and time < 0.1:
#            surface_temperature = (1644-300)*(time/0.1) + 300.0

#        if time >= 0.1 and time < 60.1:
#            surface_temperature = 1644.0

#        return surface_temperature

    # ablation workshop case #2.1
    def my_presc_flux_func(u_minus, kappa_minus, grad_u_minus, normal, **kwargs):
        time = kwargs["time"]

        flux = actx.np.where(actx.np.less(time, 0.1),
            0.3*1.5e6*(time/0.1),
            0.3*1.5e6
        )

        # FIXME make emissivity function of tau
        emissivity = 0.8
        return make_obj_array([
                flux - emissivity*5.67e-8*(u_minus**4 - 300**4),
                u_minus*0.0])

    def my_presc_grad_func(u_minus, grad_u_minus, **kwargs):
        return 0.0

    energy_boundaries = {
        BoundaryDomainTag("prescribed"):
            PrescribedFluxDiffusionBoundary(my_presc_flux_func),
        BoundaryDomainTag("neumann"):
            NeumannDiffusionBoundary(my_presc_grad_func)
    }

    def my_presc_pres_func(u_minus, **kwargs):
        return 101325.0

    pressure_boundaries = {
        BoundaryDomainTag("prescribed"):
            DirichletDiffusionBoundary(my_presc_pres_func),
        BoundaryDomainTag("neumann"):
            NeumannDiffusionBoundary(my_presc_grad_func)
    }

    def my_presc_velocity_func(u_minus, **kwargs):
        return np.zeros((2,))

    def my_dummy_velocity_func(u_minus, **kwargs):
        return u_minus

    velocity_boundaries = {
        BoundaryDomainTag("prescribed"):
            DirichletDiffusionBoundary(my_dummy_velocity_func),
        BoundaryDomainTag("neumann"):
            DirichletDiffusionBoundary(my_presc_velocity_func)
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import mirgecom.phenolics.phenolics as wall
    import mirgecom.phenolics.tacot as my_composite
    pyrolysis = my_composite.Pyrolysis()

    from mirgecom.phenolics.gas import GasProperties
    my_gas = GasProperties()

    eos = wall.PhenolicsEOS(composite=my_composite, gas=my_gas)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
    import mirgecom.phenolics.phenolics as wall

    solid_species_mass = np.empty((3,), dtype=object)
    solid_species_mass[0] = 30.0 + nodes[0]*0.0
    solid_species_mass[1] = 90.0 + nodes[0]*0.0
    solid_species_mass[2] = 160. + nodes[0]*0.0

    pressure = 101325.0 + nodes[0]*0.0

    temperature = 300.0 + nodes[0]*0.0

    pressure = force_evaluation(actx, pressure)
    temperature = force_evaluation(actx, temperature)
    solid_species_mass = force_evaluation(actx, solid_species_mass)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    if restart_file:
        t = restart_data["t"]
        istep = restart_data["step"]
        wall_vars = restart_data["wall_vars"]
    else:
        t = 0
        istep = 0
        wall_vars = wall.initializer(eos=eos,
            solid_species_mass=solid_species_mass,
            pressure=pressure, temperature=temperature, progress=0.0)

    wdv = eos.dependent_vars(wv=wall_vars, temperature_seed=temperature)

    wall_vars = force_evaluation(actx, wall_vars)
    wdv = force_evaluation(actx, wdv)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if logmgr:
        from mirgecom.logging_quantities import logmgr_set_time
        logmgr_set_time(logmgr, istep, t)

        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value:8d}, "),
            ("dt.max", "dt: {value:1.3e} s, "),
            ("t_sim.max", "sim time: {value:12.8f} s, "),
            ("t_step.max", "step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max", "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["multiply_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.phenolics.operator import phenolics_operator

    def _rhs(t, state):

        boundaries = make_obj_array([pressure_boundaries, energy_boundaries,
                                     velocity_boundaries])

        rhs = phenolics_operator(
            dcoll=dcoll, state=state, boundaries=boundaries, time=t,
            wall=wall, eos=eos, pyrolysis=pyrolysis,
            pressure_scaling_factor=pressure_scaling_factor,
            quadrature_tag=quadrature_tag, dd_wall=dd_vol)

        # ~~~~~
        return make_obj_array([rhs, tseed*0.0])

    compiled_rhs = actx.compile(_rhs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, wall_vars, dep_vars):

        wv = wall_vars
        wdv = dep_vars
        gas_pressure_diffusivity = \
            eos.gas_pressure_diffusivity(wdv.temperature, wdv.tau)
        _, grad_pressure = diffusion_operator(dcoll,
            kappa=wv.gas_density*gas_pressure_diffusivity,
            boundaries=pressure_boundaries, u=wdv.gas_pressure, time=t,
            return_grad_u=True)

        velocity = -gas_pressure_diffusivity*grad_pressure

        viz_fields = [("gas_density", wall_vars.gas_density),
                      ("DV_velocity", velocity),
                      ("grad_p", grad_pressure),
                      ("DV", dep_vars)]

        viz_fields.extend((
            ("phase_1", wall_vars.solid_species_mass[0]),
            ("phase_2", wall_vars.solid_species_mass[1]),
            ("phase_3", wall_vars.solid_species_mass[2]),
        ))

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, wall_vars, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "wall_vars": wall_vars,
                "tseed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "nparts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _eval_dep_vars(wall_vars, tseed):
        return eos.dependent_vars(wv=wall_vars, temperature_seed=tseed)

    eval_dep_vars = actx.compile(_eval_dep_vars)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from warnings import warn
    warn("Running gc.collect() to work around memory growth issue ")
    import gc
    gc.collect()

    my_write_viz(step=istep, t=t, wall_vars=wall_vars, dep_vars=wdv)

    while t < t_final:

        if logmgr:
            logmgr.tick_before()

        try:

            tseed = wdv.temperature
            state = make_obj_array([wall_vars, tseed])

            state = ssprk43_step(state, t, dt, compiled_rhs)
            state = force_evaluation(actx, state)

            wall_vars, tseed = state

            wdv = eval_dep_vars(wall_vars, tseed)

            t += dt
            istep += 1

            if check_naninf_local(dcoll, "vol", wdv.temperature):
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                raise MyRuntimeError("Failed simulation health check.")

            if istep % nviz == 0:
                my_write_viz(step=istep, t=t, wall_vars=wall_vars,
                             dep_vars=wdv)

            if istep % ngarbage == 0:
                gc.collect()

            if istep % nrestart == 0:
                my_write_restart(step=istep, t=t, wall_vars=wall_vars,
                             tseed=wdv.temperature)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=istep, t=t, wall_vars=wall_vars, dep_vars=wdv)
            raise

        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()

    if logmgr:
        logmgr.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":

    import argparse
    casename = "ablation"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    lazy = args.lazy
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
        rst_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {rst_filename}")

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename,
         restart_file=rst_filename)

# vim: foldmethod=marker
