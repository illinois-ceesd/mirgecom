"""Demonstrate double mach reflection."""

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
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import BoundaryDomainTag
from grudge.shortcuts import make_visualizer

from mirgecom.euler import euler_operator
from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import (
    av_laplacian_operator,
    smoothness_indicator
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step, euler_step
from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    PressureOutflowBoundary,
    AdiabaticSlipBoundary
)
from mirgecom.initializers import DoubleMachReflection
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import (
    SimpleTransport,
    ArtificialViscosityTransportDiv
)
from mirgecom.simutil import get_sim_timestep
from mirgecom.utils import force_evaluation

from logpyle import set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def get_doublemach_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        doubleMach.msh (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (doubleMach.msh).
    """
    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptSource,
    )
    import os
    meshfile = "doubleMach.msh"
    if not os.path.exists(meshfile):
        mesh = generate_gmsh(
            ScriptSource("""
                x0=1.0/6.0;
                setsize=0.025;
                Point(1) = {0, 0, 0, setsize};
                Point(2) = {x0,0, 0, setsize};
                Point(3) = {4, 0, 0, setsize};
                Point(4) = {4, 1, 0, setsize};
                Point(5) = {0, 1, 0, setsize};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(5) = {3, 4};
                Line(6) = {4, 5};
                Line(7) = {5, 1};
                Line Loop(8) = {-5, -6, -7, -1, -2};
                Plane Surface(8) = {8};
                Physical Surface('domain') = {8};
                Physical Curve('flow') = {1, 6, 7};
                Physical Curve('wall') = {2};
                Physical Curve('out') = {5};
        """, "geo"), force_ambient_dim=2, dimensions=2, target_unit="M",
            output_file_name=meshfile)
    else:
        mesh = read_gmsh(meshfile, force_ambient_dim=2)

    return mesh


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_leap=False, use_overintegration=False,
         casename=None, rst_filename=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    # logging and profiling
    log_path = "log_data/"
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(True,
        filename=logname, mode="wo", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # Timestepping control
    current_step = 0
    timestepper = rk4_step
    timestepper = euler_step
    force_eval = True
    t_final = 5.e-4
    current_cfl = 0.1
    current_dt = 2.5e-5
    current_t = 0
    constant_cfl = False

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = _compiled_stepper_wrapper
    force_eval = False

    # default health status bounds
    health_pres_min = 0.7
    health_pres_max = 19.
    health_temp_min = 1e-4
    health_temp_max = 100

    # Some i/o frequencies
    nviz = 250
    nrestart = 1000
    nhealth = 1
    nstatus = 1

    viz_path = "viz_data/"
    vizname = viz_path + casename
    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        gen_grid = partial(get_doublemach_mesh)
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    gen_grid)
        local_nelements = local_mesh.nelements

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    from mirgecom.discretization import create_discretization_collection
    order = 3
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None  # noqa

    dim = 2
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, ")
        ])

    # which kind of artificial viscosity?
    #    0 - none
    #    1 - laplacian diffusion
    #    2 - physical viscosity based, div(velocity) indicator
    use_av = 2

    # Solution setup and initialization
    # {{{ Initialize simple transport model
    kappa = 1.0
    s0 = np.log10(1.0e-4 / np.power(order, 4))
    alpha = 0.03
    kappa_t = 0.
    sigma_v = 0.
    # }}}

    # Shock strength
    shock_location = 1.0/6.0
    shock_speed = 4.0
    shock_sigma = 0.01

    initializer = DoubleMachReflection(shock_location=shock_location,
                                       shock_speed=shock_speed,
                                       shock_sigma=shock_sigma)

    from mirgecom.gas_model import GasModel, make_fluid_state
    physical_transport = SimpleTransport(
        viscosity=sigma_v, thermal_conductivity=kappa_t)
    if use_av < 2:
        transport_model = physical_transport
    else:
        transport_model = ArtificialViscosityTransportDiv(
            physical_transport=physical_transport,
            av_mu=1.0, av_prandtl=0.75)

    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=transport_model)

    def get_fluid_state(cv, smoothness_mu=None):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                smoothness_mu=smoothness_mu)

    create_fluid_state = actx.compile(get_fluid_state)

    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, dcoll)

    from mirgecom.navierstokes import grad_cv_operator

    # compiled wrapper for grad_cv_operator
    def _grad_cv_operator(fluid_state, time):
        return grad_cv_operator(dcoll=dcoll, gas_model=gas_model,
                                boundaries=boundaries,
                                state=fluid_state,
                                time=time,
                                quadrature_tag=quadrature_tag)

    grad_cv_operator_compiled = actx.compile(_grad_cv_operator) # noqa

    def compute_smoothness(cv, grad_cv):

        actx = cv.array_context
        from mirgecom.fluid import velocity_gradient
        div_v = np.trace(velocity_gradient(cv, grad_cv))

        # kappa_h = 1.5
        kappa_h = 5
        gamma = gas_model.eos.gamma()
        r = gas_model.eos.gas_const()
        static_temp = 0.015
        c_star = actx.np.sqrt(gamma*r*(2/(gamma+1)*static_temp))
        indicator = -kappa_h*length_scales*div_v/c_star

        # steepness of the smoothed function
        alpha = 100
        # cutoff, smoothness below this value is ignored
        beta = 0.01
        smoothness = actx.np.log(
            1 + actx.np.exp(alpha*(indicator - beta)))/alpha
        return smoothness*kappa_h*length_scales

    compute_smoothness_compiled = actx.compile(compute_smoothness) # noqa

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = initializer(nodes)

    smoothness = None
    no_smoothness = None
    if use_av > 0:
        smoothness = smoothness_indicator(dcoll, current_cv.mass,
                                          kappa=kappa, s0=s0)
        no_smoothness = 0.*smoothness

    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model,
                                     smoothness_mu=smoothness)
    force_evaluation(actx, current_state)

    def _boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(cv=initializer(x_vec=nodes, eos=gas_model.eos,
                                               **kwargs),
                                gas_model=gas_model,
                                smoothness_mu=state_minus.dv.smoothness_mu)

    flow_boundary = PrescribedFluidBoundary(
        boundary_state_func=_boundary_state)

    boundaries = {
        BoundaryDomainTag("flow"): flow_boundary,
        BoundaryDomainTag("wall"): AdiabaticSlipBoundary(),
        BoundaryDomainTag("out"): PressureOutflowBoundary(boundary_pressure=1.0),
    }

    visualizer = make_visualizer(dcoll, order)

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(
        dim=dim,
        order=order,
        nelements=local_nelements,
        global_nelements=global_nelements,
        dt=current_dt,
        t_final=t_final,
        nstatus=nstatus,
        nviz=nviz,
        cfl=current_cfl,
        constant_cfl=constant_cfl,
        initname=initname,
        eosname=eosname,
        casename=casename,
    )
    if rank == 0:
        logger.info(init_message)

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    def my_write_status(cv, dv, dt, cfl):
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        p_min = vol_min(dv.pressure)
        p_max = vol_max(dv.pressure)
        t_min = vol_min(dv.temperature)
        t_max = vol_max(dv.temperature)

        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({p_min:1.9e}, {p_max:1.9e})")
        dv_status_msg += (
            f"\n-------- T (min, max) (K)  = ({t_min:7g}, {t_max:7g})")

        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    from mirgecom.viscous import get_viscous_timestep, get_viscous_cfl

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * get_viscous_timestep(dcoll, state=state)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(dcoll, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = get_viscous_cfl(dcoll, dt=dt, state=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(dcoll, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_write_viz(step, t, fluid_state):
        cv = fluid_state.cv
        dv = fluid_state.dv
        mu = fluid_state.viscosity

        """
        exact_cv = initializer(x_vec=nodes, eos=gas_model.eos, time=t)
        exact_smoothness = smoothness_indicator(dcoll, exact_cv.mass,
                                                  kappa=kappa, s0=s0)
        exact_state = create_fluid_state(cv=exact_cv,
                                         smoothness_mu=exact_smoothness)

        # try using the divergence to compute the smoothness field
        #exact_grad_cv = grad_cv_operator_compiled(fluid_state=exact_state,
                                                  #time=t)
        exact_grad_cv = grad_cv_operator(dcoll, gas_model, boundaries,
                                         exact_state, time=current_t,
                                         quadrature_tag=quadrature_tag)
        from mirgecom.fluid import velocity_gradient
        exact_grad_v = velocity_gradient(exact_cv, exact_grad_cv)

        # make a smoothness indicator
        # try using the divergence to compute the smoothness field
        exact_smoothness = compute_smoothness(exact_cv, exact_grad_cv)

        exact_state = create_fluid_state(cv=exact_cv,
                                         smoothness_mu=exact_smoothness)
        """

        viz_fields = [("cv", cv),
                      ("dv", dv),
                      # ("exact_cv", exact_state.cv),
                      # ("exact_grad_v_x", exact_grad_v[0]),
                      # ("exact_grad_v_y", exact_grad_v[1]),
                      # ("exact_dv", exact_state.dv),
                      ("mu", mu)]
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv):
        # Note: This health check is tuned s.t. it is a test that
        #       the case gets the expected solution.  If dt,t_final or
        #       other run parameters are changed, this check should
        #       be changed accordingly.
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(dcoll, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(dcoll, "vol", dv.pressure,
                                           health_pres_min, health_pres_max),
                         op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(dcoll, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if global_reduce(
                check_range_local(dcoll, "vol", dv.temperature,
                                  health_temp_min, health_temp_max),
                op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.temperature))
            t_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.temperature))
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        return health_error

    def my_pre_step(step, t, dt, state):
        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if any([do_viz, do_restart, do_health, do_status, constant_cfl]):
                fluid_state = create_fluid_state(cv=state,
                                                 smoothness_mu=no_smoothness)
                if use_av > 1:
                    # recompute the dv to have the correct smoothness
                    # this is forcing a recompile, only do it at dump time
                    # not sure why the compiled version of grad_cv doesn't work
                    if do_viz:
                        # use the divergence to compute the smoothness field
                        force_evaluation(actx, t)
                        grad_cv = grad_cv_operator_compiled(fluid_state, time=t)
                        smoothness = compute_smoothness_compiled(state, grad_cv)

                        # this works, but seems a lot of work,
                        # not sure if it's really faster
                        # avoids re-computing the temperature
                        from dataclasses import replace
                        new_dv = replace(fluid_state.dv, smoothness_mu=smoothness)
                        fluid_state = replace(fluid_state, dv=new_dv)
                        new_tv = gas_model.transport.transport_vars(
                            cv=state, dv=new_dv, eos=gas_model.eos)
                        fluid_state = replace(fluid_state, tv=new_tv)

                # if the time integrator didn't force_eval, do so now
                if not force_eval:
                    fluid_state = force_evaluation(actx, fluid_state)
                dv = fluid_state.dv

                if do_viz:
                    my_write_viz(step=step, t=t, fluid_state=fluid_state)

                ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state)

                if do_health:
                    health_errors = \
                        global_reduce(my_health_check(state, dv), op="lor")

                    if health_errors:
                        if rank == 0:
                            logger.info("Fluid solution failed health check.")
                            raise MyRuntimeError(
                                "Failed simulation health check.")

                if do_status:
                    my_write_status(dt=dt, cfl=cfl, dv=dv, cv=state)

                if do_restart:
                    my_write_restart(step=step, t=t, state=state)

            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt,
                                      current_cfl, t_final, constant_cfl)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, fluid_state=fluid_state)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def _my_rhs(t, state):

        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)

        return (
            euler_operator(dcoll, state=fluid_state, time=t,
                           boundaries=boundaries,
                           gas_model=gas_model, quadrature_tag=quadrature_tag,
                           use_esdg=use_esdg)
        )

    def _my_rhs_av(t, state):

        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)

        return (
            euler_operator(dcoll, state=fluid_state, time=t,
                           boundaries=boundaries,
                           gas_model=gas_model, quadrature_tag=quadrature_tag,
                           use_esdg=use_esdg)
            + av_laplacian_operator(dcoll, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    time=t, gas_model=gas_model,
                                    alpha=alpha, s0=s0, kappa=kappa,
                                    quadrature_tag=quadrature_tag)
        )

    def _my_rhs_phys_visc_av(t, state):

        smoothness = smoothness_indicator(dcoll, state.mass,
                                          kappa=kappa, s0=s0)
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model,
                                       smoothness_mu=smoothness)

        return (
            ns_operator(dcoll, state=fluid_state, time=t,
                        boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag,
                        use_esdg=use_esdg)
        )

    def _my_rhs_phys_visc_div_av(t, state):

        fluid_state = make_fluid_state(cv=state, gas_model=gas_model,
                                       smoothness_mu=no_smoothness)

        # use the divergence to compute the smoothness field
        grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, fluid_state,
                                   time=t, quadrature_tag=quadrature_tag)
        smoothness = compute_smoothness(state, grad_cv)

        from dataclasses import replace
        new_dv = replace(fluid_state.dv, smoothness_mu=smoothness)
        fluid_state = replace(fluid_state, dv=new_dv)
        new_tv = gas_model.transport.transport_vars(
            cv=state, dv=new_dv, eos=gas_model.eos)
        fluid_state = replace(fluid_state, tv=new_tv)

        return (
            ns_operator(dcoll, state=fluid_state, time=t,
                        boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag,
                        grad_cv=grad_cv, use_esdg=use_esdg)
        )

    my_rhs = (_my_rhs if use_av == 0 else _my_rhs_av if use_av == 1 else
              _my_rhs_phys_visc_div_av)

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      dt=current_dt,
                      state=current_state.cv, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    if use_av < 2:
        current_state = create_fluid_state(cv=current_cv)
    else:
        current_state = create_fluid_state(cv=current_cv,
                                           smoothness_mu=no_smoothness)

        # use the divergence to compute the smoothness field
        current_grad_cv = grad_cv_operator_compiled(current_state,
                                                    time=current_t)
        smoothness = compute_smoothness_compiled(current_cv,
                                                 current_grad_cv)
        from dataclasses import replace
        new_dv = replace(current_state.dv, smoothness_mu=smoothness)
        current_state = replace(current_state, dv=new_dv)

    final_dv = current_state.dv
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state)
    my_write_status(dt=dt, cfl=cfl, cv=current_cv, dv=final_dv)
    my_write_viz(step=current_step, t=current_t, fluid_state=current_state)
    my_write_restart(step=current_step, t=current_t, state=current_cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "doublemach"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--esdg", action="store_true",
        help="use flux-differencing/entropy stable DG for inviscid computations.")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()

    from warnings import warn
    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy,
                                                    distributed=True,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_leap=args.leap, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, rst_filename=rst_filename)

# vim: foldmethod=marker
