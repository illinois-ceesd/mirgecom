"""Production-like (nozzle-like) case in a box."""

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
import os
import yaml
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
import math

from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from meshmode.array_context import (
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import make_conserved
from mirgecom.artificial_viscosity import (av_operator, smoothness_indicator)
from mirgecom.simutil import (
    check_step,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    get_sim_timestep
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedInviscidBoundary,
    IsothermalNoSlipBoundary
)
from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def get_pseudo_y0_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        data/pseudoY0.brep  (for mesh gen)
        -or-
        data/pseudoY0.msh   (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (data/pseudoY0.msh), but
    note that if the grid is generated in millimeters,
    then the solution initialization and BCs need to be
    adjusted or the grid needs to be scaled up to meters
    before being used with the current main driver in this
    example.
    """
    from meshmode.mesh.io import (read_gmsh, generate_gmsh,
                                  ScriptWithFilesSource)
    if os.path.exists("data/pseudoY1nozzle.msh") is False:
        mesh = generate_gmsh(ScriptWithFilesSource(
            """
            Merge "data/pseudoY1nozzle.brep";
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

            // Edges separating surfaces with boundary layer
            // refinement from those without
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
        """, ["data/pseudoY1nozzle.brep"]), 3, target_unit="MM")
    else:
        mesh = read_gmsh("data/pseudoY1nozzle.msh")

    return mesh


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         use_profiling=False, use_logmgr=False, user_input_file=None,
         actx_class=PyOpenCLArrayContext, casename=None):
    """Drive the Y0 nozzle example."""
    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # Most of these can be set by the user input file

    # default i/o junk frequencies
    nviz = 100
    nrestart = 100
    nhealth = 100
    nstatus = 1
    log_dependent = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 5.0e-8
    t_final = 5.0e-6
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    current_step = 0

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6

    # discretization and model control
    order = 1
    alpha_sc = 0.5
    s0_sc = -5.0
    kappa_sc = 0.5
    weak_scale = 1

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            weak_scale = float(input_data["wscale"])
        except KeyError:
            pass
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            log_dependent = int(input_data["logDependent"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if rank == 0:
        print("#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tShock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")
        print(f"\tTime integration {integrator}")
        if log_dependent:
            print("\tDependent variable logging is ON.")
        else:
            print("\tDependent variable logging is OFF.")
        print("#### Simluation control data: ####")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    dim = 3
    vel_inflow = np.zeros(shape=(dim, ))
    vel_outflow = np.zeros(shape=(dim, ))

    # working gas: CO2 #
    #   gamma = 1.289
    #   MW=44.009  g/mol
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma_co2 = 1.289
    r_co2 = 8314.59/44.009

    # background
    #   100 Pa
    #   298 K
    #   rho = 1.77619667e-3 kg/m^3
    #   velocity = 0,0,0
    rho_bkrnd = 1.77619667e-3
    pres_bkrnd = 100
    temp_bkrnd = 298

    # nozzle inflow #
    #
    # stagnation tempertuare 298 K
    # stagnation pressure 1.5e Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=13e-3m)
    # and the throat (r=6.3e-3)
    #
    # calculate the inlet Mach number from the area ratio
    nozzle_inlet_radius = 13.0e-3
    nozzle_throat_radius = 6.3e-3
    nozzle_inlet_area = math.pi*nozzle_inlet_radius*nozzle_inlet_radius
    nozzle_throat_area = math.pi*nozzle_throat_radius*nozzle_throat_radius
    inlet_area_ratio = nozzle_inlet_area/nozzle_throat_area

    def get_mach_from_area_ratio(area_ratio, gamma, mach_guess=0.01):
        error = 1.0e-8
        next_error = 1.0e8
        g = gamma
        m0 = mach_guess
        while next_error > error:
            r_gas = (
                (((2/(g + 1) + ((g - 1)/(g + 1)*m0*m0))**(((g + 1)/(2*g - 2))))/m0
                 - area_ratio)
            )
            drdm = (
                (2*((2/(g + 1) + ((g - 1)/(g + 1)*m0*m0))**(((g + 1)/(2*g - 2))))
                 / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*m0*m0))
                 - ((2/(g + 1) + ((g - 1)/(g + 1)*m0*m0))**(((g + 1)/(2*g - 2))))
                 * m0**(-2))
            )
            m1 = m0 - r_gas/drdm
            next_error = abs(r_gas)
            m0 = m1

        return m1

    def get_isentropic_pressure(mach, p0, gamma):
        pressure = (1. + (gamma - 1.)*0.5*math.pow(mach, 2))
        pressure = p0*math.pow(pressure, (-gamma / (gamma - 1.)))
        return pressure

    def get_isentropic_temperature(mach, t0, gamma):
        temperature = (1. + (gamma - 1.)*0.5*math.pow(mach, 2))
        temperature = t0*math.pow(temperature, -1.0)
        return temperature

    inlet_mach = get_mach_from_area_ratio(area_ratio=inlet_area_ratio,
                                          gamma=gamma_co2,
                                          mach_guess=0.01)
    # ramp the stagnation pressure
    start_ramp_pres = 1000
    ramp_interval = 1.0e-3
    t_ramp_start = 1.0e-5
    pres_inflow = get_isentropic_pressure(mach=inlet_mach,
                                        p0=start_ramp_pres,
                                        gamma=gamma_co2)
    temp_inflow = get_isentropic_temperature(mach=inlet_mach,
                                           t0=298,
                                           gamma=gamma_co2)
    rho_inflow = pres_inflow / temp_inflow / r_co2
    end_ramp_pres = 150000
    pres_inflow_final = get_isentropic_pressure(mach=inlet_mach,
                                              p0=end_ramp_pres,
                                              gamma=gamma_co2)
    vel_inflow[0] = inlet_mach * math.sqrt(
        gamma_co2 * pres_inflow / rho_inflow)

    if rank == 0:
        print(f"inlet Mach number {inlet_mach}")
        print(f"inlet temperature {temp_inflow}")
        print(f"inlet pressure {pres_inflow}")
        print(f"final inlet pressure {pres_inflow_final}")

    mu = 1.e-5
    kappa = rho_bkrnd*mu/0.75
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    eos = IdealSingleGas(
        gamma=gamma_co2,
        gas_const=r_co2,
        transport_model=transport_model
    )
    bulk_init = PlanarDiscontinuity(dim=dim, disc_location=-.30, sigma=0.005,
        temperature_left=temp_inflow, temperature_right=temp_bkrnd,
        pressure_left=pres_inflow, pressure_right=pres_bkrnd,
        velocity_left=vel_inflow, velocity_right=vel_outflow)

    # pressure ramp function
    def inflow_ramp_pressure(
        t,
        start_p=start_ramp_pres,
        final_p=end_ramp_pres,
        ramp_interval=ramp_interval,
        t_ramp_start=t_ramp_start
    ):
        return actx.np.where(
            actx.np.greater(t, t_ramp_start),
            actx.np.minimum(
                final_p,
                start_p + (t - t_ramp_start) / ramp_interval * (final_p - start_p)),
            start_p)

    class IsentropicInflow:
        def __init__(self, *, dim=1, direc=0, t0=298, p0=1e5, mach=0.01, p_fun=None):

            self._p0 = p0
            self._t0 = t0
            self._dim = dim
            self._direc = direc
            self._mach = mach
            if p_fun is not None:
                self._p_fun = p_fun

        def __call__(self, x_vec, *, time=0, eos, **kwargs):

            if self._p_fun is not None:
                p0 = self._p_fun(time)
            else:
                p0 = self._p0
            t0 = self._t0

            gamma = eos.gamma()
            gas_const = eos.gas_const()
            pressure = get_isentropic_pressure(
                mach=self._mach,
                p0=p0,
                gamma=gamma
            )
            temperature = get_isentropic_temperature(
                mach=self._mach,
                t0=t0,
                gamma=gamma
            )
            rho = pressure/temperature/gas_const

            velocity = np.zeros(self._dim, dtype=object)
            velocity[self._direc] = self._mach*actx.np.sqrt(gamma*pressure/rho)

            mass = 0.0*x_vec[0] + rho
            mom = velocity*mass
            energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
            return make_conserved(
                dim=self._dim,
                mass=mass,
                momentum=mom,
                energy=energy
            )

    inflow_init = IsentropicInflow(
        dim=dim,
        t0=298,
        p0=start_ramp_pres,
        mach=inlet_mach,
        p_fun=inflow_ramp_pressure
    )
    outflow_init = Uniform(
        dim=dim,
        rho=rho_bkrnd,
        p=pres_bkrnd,
        velocity=vel_outflow
    )

    inflow = PrescribedInviscidBoundary(fluid_solution_func=inflow_init)
    outflow = PrescribedInviscidBoundary(fluid_solution_func=outflow_init)
    wall = IsothermalNoSlipBoundary()

    boundaries = {
        DTAG_BOUNDARY("Inflow"): inflow,
        DTAG_BOUNDARY("Outflow"): outflow,
        DTAG_BOUNDARY("Wall"): wall
    }

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]
    else:
        boundary_tag_to_face = {
            "Inflow": ["-x"],
            "Outflow": ["+x"],
            "Wall": ["-y", "+y", "-z", "+z"]
        }
        scale = np.power(weak_scale, 1/dim)
        box_ll = 0
        box_ur = .5*scale
        nel_1d = int(8*scale)
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                                b=(box_ur,) * dim,
                                nelements_per_axis=(nel_1d,) * dim,
                                boundary_tag_to_face=boundary_tag_to_face)
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm,
            generate_mesh
        )
        local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")

    discr = EagerDGDiscretization(actx,
                                  local_mesh,
                                  order=order,
                                  mpi_communicator=comm)

    nodes = thaw(actx, discr.nodes())

    # initialize the sponge field
    def gen_sponge():
        thickness = 0.15
        amplitude = 1.0/current_dt/25.0
        x0 = 0.05

        return amplitude * actx.np.where(
            actx.np.greater(nodes[0], x0),
            zeros + ((nodes[0] - x0) / thickness) * ((nodes[0] - x0) / thickness),
            zeros + 0.0,
        )

    zeros = 0 * nodes[0]
    sponge_sigma = gen_sponge()
    ref_state = bulk_init(x_vec=nodes, eos=eos, time=0.0)

    if restart_filename:
        if rank == 0:
            logging.info("Restarting soln.")
        current_state = restart_data["state"]
        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol")
            )
            restart_state = restart_data["state"]
            current_state = connection(restart_state)
    else:
        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_state = bulk_init(x_vec=nodes, eos=eos, time=0.0)
        # for uniform background initial condition
        # current_state = bulk_init(nodes, eos=eos)

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)
        logmgr.add_quantity(log_cfl, interval=nstatus)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("cfl.max", "cfl = {value:1.4f}\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s\n")
        ])

        if log_dependent:
            logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                                                      extract_vars_for_logging,
                                                      units_for_logging)
            logmgr.add_watches([
                ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
                ("max_pressure", "{value:1.9e})\n"),
                ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
                ("max_temperature", "{value:7g})\n"),
            ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr)

    initname = "pseudoY0"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim,
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
                                     casename=casename)
    if rank == 0:
        logger.info(init_message)

    def sponge(cv, cv_ref, sigma):
        return (sigma*(cv_ref - cv))

    def my_rhs(t, state):
        return (
            ns_operator(discr, cv=state, t=t, boundaries=boundaries, eos=eos)
            + make_conserved(
                dim, q=av_operator(discr, q=state.join(), boundaries=boundaries,
                                   boundary_kwargs={"time": t, "eos": eos},
                                   alpha=alpha_sc, s0=s0_sc, kappa=kappa_sc)
            ) + sponge(cv=state, cv_ref=ref_state, sigma=sponge_sigma)
        )

    def my_write_viz(step, t, dt, state, dv=None, tagged_cells=None, ts_field=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if tagged_cells is None:
            tagged_cells = smoothness_indicator(discr, state.mass, s0=s0_sc,
                                                kappa=kappa_sc)
        if ts_field is None:
            ts_field, cfl, dt = my_get_timestep(t, dt, state)

        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("sponge_sigma", gen_sponge()),
                      ("tagged_cells", tagged_cells),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_range_local(discr, "vol", dv.pressure,
                             health_pres_min, health_pres_max):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation.")

        return health_error

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            from mirgecom.viscous import get_viscous_timestep
            ts_field = current_cfl * get_viscous_timestep(discr, eos=eos, cv=state)
            from grudge.op import nodal_min
            dt = nodal_min(discr, "vol", ts_field)
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            ts_field = get_viscous_cfl(discr, eos=eos, dt=dt, cv=state)
            from grudge.op import nodal_max
            cfl = nodal_max(discr, "vol", ts_field)

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, state)
            log_cfl.set_quantity(cfl)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(dv), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_viz(step=step, t=t, dt=dt, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    if rank == 0:
        logging.info("Stepping.")

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state, dt=current_dt,
                      t_final=t_final, t=current_t, istep=current_step)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    my_write_viz(step=current_step, t=current_t, dt=current_dt, state=current_state,
                 dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename",
                        nargs="?", action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    args = parser.parse_args()

    # for writing output
    casename = "nozzle"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    if args.profile:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Reading user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, use_profiling=args.profile,
         use_logmgr=args.log, user_input_file=input_file,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
