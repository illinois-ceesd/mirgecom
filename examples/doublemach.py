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

import time
import logging
import numpy as np
from functools import partial

from mirgecom.fluid import (
    make_conserved
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.dof_desc import BoundaryDomainTag
from grudge.shortcuts import make_visualizer


from mirgecom.euler import euler_operator
from mirgecom.artificial_viscosity import (
    av_laplacian_operator,
    smoothness_indicator,
    AdiabaticNoSlipWallAV,
    PrescribedFluidBoundaryAV
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.initializers import DoubleMachReflection
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.simutil import get_sim_timestep, ApplicationOptionsError
from logpyle import set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)
from pytato.transform.parameter_study import ParameterStudyAxisTag
from arraycontext.parameter_study import pack_for_parameter_study, unpack_parameter_study

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass

class DoubleMachParameterStudy(ParameterStudyAxisTag):
    """ Parameter study altering the shock initial conditions. """
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
                Physical Curve('ic1') = {6};
                Physical Curve('ic2') = {7};
                Physical Curve('ic3') = {1};
                Physical Curve('wall') = {2};
                Physical Curve('out') = {5};
        """, "geo"), force_ambient_dim=2, dimensions=2, target_unit="M",
            output_file_name=meshfile)
    else:
        mesh = read_gmsh(meshfile, force_ambient_dim=2)

    return mesh


@mpi_entry_point
def main(actx_class, use_esdg=False,
         use_overintegration=False, use_leap=False,
         casename=None, rst_filename=None, nel_1d:int = 16, order:int=1,
         num_uq=None):
    """Drive the example."""
    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    logmgr = initialize_logmgr(True,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    from mirgecom.array_context import initialize_actx, actx_class_is_profiling
    actx = initialize_actx(actx_class, comm)
    queue = getattr(actx, "queue", None)
    use_profiling = actx_class_is_profiling(actx_class)

    # Timestepping control
    current_step = 0
    timestepper = rk4_step
    t_final = 1.0e-3
    current_cfl = 0.1
    current_dt = 1.0e-5
    current_t = 0
    constant_cfl = False

    # Some i/o frequencies
    nstatus = 10
    nviz = 1
    nrestart = -100
    nhealth = -1

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
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, gen_grid)
        local_nelements = local_mesh.nelements

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    dim = 2
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, dcoll, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K) = ({value:1.9e}, "),
            ("max_temperature",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    # Solution setup and initialization
    s0 = -6.0
    kappa = 1.0
    alpha = 2.0e-2
    # {{{ Initialize simple transport model
    kappa_t = 1e-5
    sigma_v = 1e-5
    # }}}

    from mirgecom.gas_model import GasModel, make_fluid_state
    cvs = []
    source_locs = np.random.random(num_uq) # U [0, 1]
    shock_speed = np.random.random(num_uq)*4 + 4 # U [4, 8]
    shock_sigma = np.random.random(num_uq)*0.05 + 0.025 # U [0.025, 0.075]
    visualizer = make_visualizer(dcoll)
    transport_model = SimpleTransport(viscosity=sigma_v,
                                  thermal_conductivity=kappa_t)
    eos = IdealSingleGas()
    gas_model = GasModel(eos=eos, transport=transport_model)
    bounds = []
    for uq in range(num_uq):
        initializer = None
        if uq == 0:
            initializer = DoubleMachReflection()
        else:
            initializer = DoubleMachReflection(shock_location=source_locs[uq],
                                           shock_speed=shock_speed[uq],
                                           shock_sigma=shock_sigma[uq])


        def _boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = dcoll.discr_from_dd(dd_bdry)
            nodes = actx.thaw(bnd_discr.nodes())
            return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model)

        boundaries = {
            BoundaryDomainTag("ic1"):
                PrescribedFluidBoundaryAV(boundary_state_func=_boundary_state),
            BoundaryDomainTag("ic2"):
                PrescribedFluidBoundaryAV(boundary_state_func=_boundary_state),
            BoundaryDomainTag("ic3"):
                PrescribedFluidBoundaryAV(boundary_state_func=_boundary_state),
            BoundaryDomainTag("wall"): AdiabaticNoSlipWallAV(),
            BoundaryDomainTag("out"): AdiabaticNoSlipWallAV(),
        }
        bounds.append(boundaries)
        # Set the current state from time 0
        current_cv = initializer(nodes)
        cvs.append(current_cv)
    
    current_cv = pack_for_parameter_study(actx, DoubleMachParameterStudy, *[cv for cv in cvs])
    single_states = [make_fluid_state(cv, gas_model=gas_model) for cv in cvs]
    current_state = make_fluid_state(current_cv, gas_model=gas_model)


    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    """
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
    """

    UNCERTAIN = True
    def my_write_viz(step, t, state, dv, tagged_cells):
        from mirgecom.simutil import write_visfile
        from pytato import unify_axes_tags
        if UNCERTAIN:
            mass_vals = unpack_parameter_study(unify_axes_tags(state.mass[0]), DoubleMachParameterStudy)
            energy_vals = unpack_parameter_study(unify_axes_tags(state.energy[0]), DoubleMachParameterStudy)
            momentum_x_vals = unpack_parameter_study(unify_axes_tags(state.momentum[0][0]), DoubleMachParameterStudy)
            momentum_y_vals = unpack_parameter_study(unify_axes_tags(state.momentum[1][0]), DoubleMachParameterStudy)
            actx = state.array_context
            for i in range(num_uq):
                q = np.empty(2 + state.dim, dtype="O")
                if len(mass_vals) < num_uq:
                    breakpoint()
                q[0] = DOFArray(actx, (mass_vals[i],))
                q[1] = DOFArray(actx, (energy_vals[i],))
                q[2] = DOFArray(actx, (momentum_x_vals[i],))
                q[3] = DOFArray(actx, (momentum_y_vals[i],))

                tmp_state = make_conserved(dim=state.dim, q=q)
                dv = eos.dependent_vars(tmp_state)
                tagged_cells = smoothness_indicator(dcoll, tmp_state.mass, s0=s0,
                                                    kappa=kappa)
                viz_fields = [("cv", tmp_state),
                              ("dv", eos.dependent_vars(tmp_state)),
                              ("tagged_cells", tagged_cells)]
                write_visfile(dcoll, viz_fields, visualizer, vizname=casename + "_uq_" + str(i),
                              step=step, t=t, overwrite=True,
                              comm=comm)

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

        if global_reduce(check_range_local(dcoll, "vol", dv.pressure, .9, 18.6),
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
                check_range_local(dcoll, "vol", dv.temperature, 2.48e-3, 1.16e-2),
                op="lor"):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.temperature))
            t_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.temperature))
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state, gas_model)
        dv = fluid_state.dv
        try:

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = \
                    global_reduce(my_health_check(state, dv), op="lor")

                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, dv=dv,
                             tagged_cells=None)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            tagged_cells = smoothness_indicator(dcoll, state.mass, s0=s0,
                                                kappa=kappa)
            my_write_viz(step=step, t=t, state=state,
                         tagged_cells=tagged_cells, dv=dv)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                              current_cfl, t_final, constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        #if logmgr:
        #    set_dt(logmgr, dt)
        #    set_sim_state(logmgr, dim, state, eos)
        #    logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(state, gas_model)
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

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    start_clock = time.monotonic_ns()
    # State needs to be the current_cv so that we can build the fluid state from it.
    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state.cv, t=current_t, t_final=t_final)
    end_clock = time.monotonic_ns()


    print(f"Completed the parallel execution in {end_clock - start_clock} (ns)")
    packed_final_time = current_t
    packed_final_cv = current_cv
    packed_final_step = current_step

    def assert_arrays_close(arr1, arr2):

        mask1 = np.isfinite(arr1)
        mask2 = np.isfinite(arr2)

        assert np.allclose(mask1, mask2) # Both NANs in the same locations.
        assert np.allclose(arr1[mask1], arr2[mask2])

    time_data = []
    for i in range(num_uq):
        current_step = 0
        timestepper = rk4_step
        t_final = 1.0e-3
        current_cfl = 0.1
        current_dt = 1.0e-5
        current_t = 0
        constant_cfl = False
        current_state=single_states[i]

        UNCERTAIN=False # turn off splitting. 
        current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                      current_cfl, t_final, constant_cfl)

        start_single = time.monotonic_ns()
        current_step, current_t, current_cv = \
            advance_state(rhs=my_rhs, timestepper=timestepper,
                          pre_step_callback=my_pre_step,
                          post_step_callback=my_post_step, dt=current_dt,
                          state=current_state.cv, t=current_t, t_final=t_final)
        end_single = time.monotonic_ns()
        time_data.append(end_single - start_single)
        if rank == 0:
            print(f"Advancing state 1 by 1 took: {end_single - start_single} (ns) for {i+1} of {num_uq} states.")

        assert current_step == packed_final_step
        assert current_t == packed_final_time

        for j in range(dim):
            saved_momentum = actx.to_numpy(packed_final_cv.momentum[j][0][...,i])
            correct_momentum = actx.to_numpy(current_cv.momentum[j][0])
            assert_arrays_close(saved_momentum, correct_momentum)
        assert np.allclose(actx.to_numpy(packed_final_cv.nspecies),
                     actx.to_numpy(current_cv.nspecies))
        # Index into the DoFArray
        saved_mass = actx.to_numpy(packed_final_cv.mass[0])[..., i]
        correct_mass = actx.to_numpy(current_cv.mass[0])
        assert_arrays_close(saved_mass, correct_mass)
        
        saved_energy = actx.to_numpy(packed_final_cv.energy[0])[...,i]
        correct_energy = actx.to_numpy(current_cv.energy[0])
        assert_arrays_close(saved_energy, correct_energy)
        
        saved_speed = actx.to_numpy(packed_final_cv.speed[0])[..., i]
        correct_speed = actx.to_numpy(current_cv.speed[0])
        assert_arrays_close(saved_speed, correct_speed)


    # Dump the final data
    if rank == 0:
        print(f"Advancing the states together took: {end_clock - start_clock} (ns)")
        print(f"Average cost for 1 by 1: {np.mean(time_data)} (ns)")
        logger.info("Checkpointing final state ...")

    current_state = make_fluid_state(packed_final_cv, gas_model)
    final_dv = current_state.dv
    """
    def my_smooth_indicator(mass, s0, kappa):
        return smoothness_indicator(dcoll, mass, s0=s0, kappa=kappa)

    smooth_out = actx.compile(my_smooth_indicator)
    tagged_cells = smooth_out(current_cv.mass, s0=s0, kappa=kappa)
    my_write_viz(step=current_step, t=current_t, state=current_cv, dv=final_dv,
                 tagged_cells=tagged_cells)
    my_write_restart(step=current_step, t=current_t, state=current_cv)
    """

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol

    if rank == 0:
        nicks_timing_data_file = "doublemach_timing_data_file.txt"
        # Then, we know we can just add the next data line.
        # Num 1d elem, Order, Num Uq, Num Timesteps, All Together time (ns),
        # Avg 1-by-1 time (ns)
        with open(nicks_timing_data_file, "a+") as my_file:
            my_file.write(str(nel_1d))
            my_file.write(",")
            my_file.write(str(order))
            my_file.write(",")
            my_file.write(str(num_uq))
            my_file.write(",")
            my_file.write(str(current_step))
            my_file.write(",")
            my_file.write(str(end_clock - start_clock))
            my_file.write(",")
            my_file.write(str(np.mean(time_data)))
            my_file.write("\r\n")



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
    parser.add_argument("--elms",type=int, help="Number of 1d elements.")
    parser.add_argument("--order",type=int, help="Order of the elements.")
    parser.add_argument("--uncertain",type=int, help="Number of uncertain trails.")
    args = parser.parse_args()

    from warnings import warn
    if args.esdg:
        if not args.lazy and not args.numpy:
            raise ApplicationOptionsError("ESDG requires lazy or numpy context.")
        if not args.overintegration:
            warn("ESDG requires overintegration, enabling --overintegration.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True,
                                                    profiling=args.profiling,
                                                    numpy=args.numpy)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_esdg=args.esdg,
         use_overintegration=args.overintegration or args.esdg,
         use_leap=args.leap,
         casename=casename, rst_filename=rst_filename,
         nel_1d=args.elms, order=args.order, num_uq=args.uncertain)
# vim: foldmethod=marker
