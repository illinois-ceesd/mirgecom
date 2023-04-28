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
from grudge.dof_desc import (
    BoundaryDomainTag,
    DISCR_TAG_BASE
)
from mirgecom.discretization import create_discretization_collection

from mirgecom.integrators.ssprk import ssprk43_step

from mirgecom.diffusion import (
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary
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

    t_final = 1.0

    dim = 1

    order = 2
    dt = 1.0e-8
    pressure_scaling_factor = 1.0  # noqa N806

    dt = dt/pressure_scaling_factor

    nviz = 200
    ngarbage = 50
    nrestart = 10000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    rst_pattern = rst_path + "{cname}-{step:09d}-{rank:04d}.pkl"
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
        nel_1d = 201

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
            a=(0.0,)*dim, b=(0.05,)*dim,
            nelements_per_axis=(nel_1d,)*dim,
            boundary_tag_to_face={"prescribed": ["+x"], "neumann": ["-x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    quadrature_tag = DISCR_TAG_BASE

    nodes = actx.thaw(dcoll.nodes())

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
#    def my_presc_energy_func(u_minus, **kwargs):
#        time = kwargs['time']

#        surface_temperature = actx.np.where(actx.np.less(time, 0.1),
#            (1644-300)*(time/0.1) + 300,
#            1644
#        )
#        surface_temperature = u_minus*0.0 + 1200

#        return surface_temperature

#    # ablation workshop case #2.1
#    def my_presc_flux_func(kappa_tpair, u_tpair, grad_u_tpair,
#                           lengthscales_tpair, normal, penalty_amount):

#        flux = actx.np.where(actx.np.less(time, 0.1),
#            0.3*(time/0.1)*1.5e6*(time/0.1),
#            0.3*1.5e6
#        )
#
#        # FIXME add blowing correction
#        Bsurf = m_dot_g/conv_coeff
#
#        temperature = u_tpair.int
#

#        H11 = bprime_class._H[ idx_B ,  idx_T ]
#        H21 = bprime_class._H[ idx_B , idx_T+1]
#        H12 = bprime_class._H[idx_B+1,  idx_T ]
#        H22 = bprime_class._H[idx_B+1, idx_T+1]

#        print(H11, H21, H12, H22)

#        # FIXME make emissivity function of tau
#        emissivity = 0.8
#        radiation = emissivity*5.67e-8*(temperature**4 - 300**4)
#        return make_obj_array([flux - radiation])

    pressure_boundaries = {
        BoundaryDomainTag("prescribed"):
            DirichletDiffusionBoundary(101325),
        BoundaryDomainTag("neumann"):
            NeumannDiffusionBoundary(0.0)
    }

    # FIXME this is just a dummy work around. Make this the right way...
    velocity_boundaries = {
        BoundaryDomainTag("prescribed"):
            DirichletDiffusionBoundary(-1234567.0),
        BoundaryDomainTag("neumann"):
            DirichletDiffusionBoundary(-1234567.0)
    }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import mirgecom.phenolics.phenolics as wall
    import mirgecom.phenolics.tacot as my_composite

    pyrolysis = my_composite.Pyrolysis()
    bprime_class = my_composite.BprimeTable()
    my_solid = my_composite.SolidProperties()
    my_gas = my_composite.GasProperties()

    eos = wall.PhenolicsEOS(solid_data=my_solid, gas_data=my_gas)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
    import mirgecom.phenolics.phenolics as wall

    solid_species_mass = np.empty((3,), dtype=object)

    solid_species_mass[0] = 30.0 + nodes[0]*0.0
    solid_species_mass[1] = 90.0 + nodes[0]*0.0
    solid_species_mass[2] = 160. + nodes[0]*0.0
    temperature = 300.0 + 0.0*nodes[0]
    pressure = 101325.0 + nodes[0]*0.0

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
            pressure=pressure, temperature=temperature)

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

        boundaries = make_obj_array([pressure_boundaries, velocity_boundaries])

        rhs = phenolics_operator(
            dcoll=dcoll, state=state, boundaries=boundaries, time=t,
            eos=eos, pyrolysis=pyrolysis, quadrature_tag=quadrature_tag,
            dd_wall=dd_vol, pressure_scaling_factor=pressure_scaling_factor,
            bprime_class=bprime_class, penalty_amount=1.0)

        # ~~~~~
        return make_obj_array([rhs, tseed*0.0])

    compiled_rhs = actx.compile(_rhs)

#    def _get_gradients(wv, wdv):
#        gas_pressure_diffusivity = \
#            eos.gas_pressure_diffusivity(wdv.temperature, wdv.tau)
#        _, grad_pressure = diffusion_operator(dcoll,
#            kappa=wv.gas_density*gas_pressure_diffusivity,
#            boundaries=pressure_boundaries, u=wdv.gas_pressure, time=t,
#            penalty_amount=1.0,
#            return_grad_u=True)

#        _, grad_temperature = diffusion_operator(dcoll,
#            kappa=wdv.thermal_conductivity,
#            boundaries=energy_boundaries, u=wdv.temperature, time=t,
#            penalty_amount=1.0,
#            return_grad_u=True)

#        return make_obj_array([grad_pressure, grad_temperature])

#    get_gradients = actx.compile(_get_gradients)

#    def _get_rhs_terms(t, wv, wdv):

#        boundaries = make_obj_array([pressure_boundaries, energy_boundaries,
#                                     velocity_boundaries])
#
#        rhs_I, rhs_V, rhs_S = phenolics_operator(
#            dcoll=dcoll, state=make_obj_array([wv, wdv.temperature]),
#            boundaries=boundaries, time=t,
#            wall=wall, eos=eos, pyrolysis=pyrolysis,
#            quadrature_tag=quadrature_tag, dd_wall=dd_vol,
#            pressure_scaling_factor=pressure_scaling_factor,
#            penalty_amount=1.0,
#            split=True)

#        return make_obj_array([rhs_I, rhs_V, rhs_S])

#    get_rhs_terms = actx.compile(_get_rhs_terms)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, wall_vars, dep_vars):

        wdv = force_evaluation(actx, dep_vars)
        wv = force_evaluation(actx, wall_vars)

        viz_fields = [("x", nodes[0]),
                      ("WV_gas_density", wv.gas_density),
                      ("WV_energy", wv.energy),
                      ("WV_phase_1", wall_vars.solid_species_mass[0]),
                      ("WV_phase_2", wall_vars.solid_species_mass[1]),
                      ("WV_phase_3", wall_vars.solid_species_mass[2]),
                      ("DV", wdv)]

#        gas_pressure_diffusivity = \
#            eos.gas_pressure_diffusivity(wdv.temperature, wdv.tau)
#        gradients = get_gradients(wv, wdv)
#        grad_P, grad_T = gradients
#        velocity = -gas_pressure_diffusivity*grad_P
#        mass_flux = wv.gas_density*velocity

#        rhs = get_rhs_terms(t, wv, wdv)
#        rhs_I, rhs_V, rhs_S = rhs

#        viz_fields.extend((
#            ("DV_velocity", velocity),
#            ("DV_mass_flux", mass_flux),
#            ("grad_P", grad_P),
#            ("grad_T", grad_T),
#            ("RHS_I", rhs_I),
#            ("RHS_V", rhs_V),
#            ("RHS_S", rhs_S)))

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

#    def _eval_temperature(wall_vars, tseed):
#        tau = eos.eval_tau(wall_vars)
#        return eos.eval_temperature(wall_vars, tseed, tau)

#    eval_temperature = actx.compile(_eval_temperature)

    def _eval_dep_vars(wall_vars, tseed):
        return eos.dependent_vars(wv=wall_vars, temperature_seed=tseed)

    eval_dep_vars = actx.compile(_eval_dep_vars)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from warnings import warn
    warn("Running gc.collect() to work around memory growth issue ")
    import gc
    gc.collect()

    my_write_viz(step=istep, t=t, wall_vars=wall_vars, dep_vars=wdv)

#    rhs = get_rhs_terms(t, wall_vars, wdv)
#    rhs_I, rhs_V, rhs_S = rhs

    freeze_gc_flag = True
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

            if freeze_gc_flag is True:
                print("Freeze gc")
                freeze_gc_flag = False

                import gc
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                gc.freeze()

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
