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

import sys  # noqa
import logging
import numpy as np
import pyopencl as cl

from meshmode.discretization.connection import FACE_RESTR_ALL

from grudge.trace_pair import (
    TracePair, interior_trace_pairs, tracepair_with_discr_tag
)
from grudge import op
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    BoundaryDomainTag,
    DISCR_TAG_BASE
)

from mirgecom.discretization import create_discretization_collection
from mirgecom.integrators import ssprk43_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    PrescribedFluxDiffusionBoundary,
    NeumannDiffusionBoundary
)
from mirgecom.simutil import (
    check_naninf_local,
    generate_and_distribute_mesh,
    write_visfile
)
from mirgecom.mpi import mpi_entry_point
from mirgecom.utils import force_evaluation
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage
)
from mirgecom.eos import GasDependentVars
from mirgecom.fluid import ConservedVars
from mirgecom.transport import GasTransportVars
from mirgecom.gas_model import PorousFluidState
from mirgecom.multiphysics.phenolics import WallConservedVars
from mirgecom.multiphysics.wall_model import PorousTransportVars

from logpyle import IntervalTimer, set_dt

from pytools.obj_array import make_obj_array

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _MyGradTag:
    pass


class _PresDiffTag:
    pass


class _TempDiffTag:
    pass


@mpi_entry_point
def main(actx_class, use_logmgr=True, use_profiling=False, casename=None,
         lazy=False, restart_file=None):
    """Demonstrate the ablation workshop test case 2.1."""
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

    t_final = 1.0e-7

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

    wall_sample_mask = nodes[0]*0.0 + 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if rank == 0:
        print("----- Discretization info ----")
        #  print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # {{{ Initialize wall model

    from mirgecom.materials.tacot import WallTabulatedEOS
    import mirgecom.multiphysics.phenolics as wall
    import mirgecom.materials.tacot as my_composite

    my_solid = my_composite.SolidProperties()
    my_gas = my_composite.GasProperties()
    pyrolysis = my_composite.Pyrolysis()

#    # FIXME there must be a way to get the table from the "materials" directory
    bprime_table = \
        (np.genfromtxt("aw_Bprime.dat", skip_header=1)[:, 2:6]).reshape((25, 151, 4))
    bprime_class = my_composite.BprimeTable(table=bprime_table)

    def _get_wall_enthalpy(temperature, tau):
        wall_sample_h = my_solid.solid_enthalpy(temperature, tau)
        return wall_sample_h * wall_sample_mask

    def _get_wall_heat_capacity(temperature, tau):
        wall_sample_cp = my_solid.solid_heat_capacity(temperature, tau)
        return wall_sample_cp * wall_sample_mask

    def _get_wall_thermal_conductivity(temperature, tau):
        scaled_sample_kappa = \
            my_solid.solid_thermal_conductivity(temperature, tau)
        return scaled_sample_kappa * wall_sample_mask

    wall_model = WallTabulatedEOS(
        wall_material=my_solid,
        wall_sample_mask=wall_sample_mask,
        enthalpy_func=_get_wall_enthalpy,
        heat_capacity_func=_get_wall_heat_capacity,
        thermal_conductivity_func=_get_wall_thermal_conductivity)

    # }}}

    from mirgecom.gas_model import GasModel
    gas_model = GasModel(eos=my_gas, wall=wall_model)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # soln setup and init
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
        cv, wall_vars = wall.initializer(dcoll=dcoll, gas_model=gas_model,
            solid_species_mass=solid_species_mass,
            pressure=pressure, temperature=temperature)

    temperature_seed = nodes[0]*0.0 + 300.0

    def _make_state(cv, gas_model, temperature_seed, wall_vars):
        """Return the fluid+wall state for porous media flow.

        Ideally one would use the gas_model.make_fluid_state but, since this
        case use tabulated data and equilibrium gas assumption, it was
        implemented in this stand-alone function. Note that some functions
        have slightly different calls and the absence of species.
        """

        zeros = cv.mass*0.0

        tau = gas_model.wall.decomposition_progress(wall_vars)
        epsilon = gas_model.wall.void_fraction(tau)
        temperature = gas_model.wall.get_temperature(cv=cv, wv=wall_vars,
            tseed=temperature_seed, tau=tau, eos=my_gas)

        pressure = 1.0/epsilon*gas_model.eos.pressure(cv=cv,
                                                      temperature=temperature)

        dv = GasDependentVars(
            temperature=temperature,
            pressure=pressure,
            speed_of_sound=zeros,
            smoothness_mu=zeros,
            smoothness_kappa=zeros,
            smoothness_beta=zeros,
        )

        # gas only transport vars
        gas_mu = gas_model.eos.gas_viscosity(temperature)
        gas_kappa = gas_model.eos.gas_thermal_conductivity(temperature)
        gas_tv = GasTransportVars(
            bulk_viscosity=zeros,
            viscosity=gas_mu,
            thermal_conductivity=gas_kappa,
            species_diffusivity=np.empty((0,), dtype=object))

        # coupled solid-gas thermal conductivity
        kappa = wall_model.thermal_conductivity(cv, wall_vars,
                                                temperature, tau, gas_tv)
        # pressure diffusivity for Darcy flow
        pressure_diffusivity = wall_model.pressure_diffusivity(cv, tau, gas_tv)

        # return modified transport vars
        tv = PorousTransportVars(
            bulk_viscosity=zeros,
            viscosity=gas_mu,
            thermal_conductivity=kappa,
            species_diffusivity=np.empty((0,), dtype=object),
            pressure_diffusivity=pressure_diffusivity
        )

        wdv = gas_model.wall.dependent_vars(wall_vars, temperature)
        return PorousFluidState(cv=cv, dv=dv, tv=tv, wv=wall_vars, wdv=wdv)

    def make_state(cv, temperature_seed, wall_vars):
        return _make_state(cv, gas_model, temperature_seed, wall_vars)

    compiled_make_state = actx.compile(make_state)

    fluid_state = compiled_make_state(cv, temperature_seed, wall_vars)

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

    def compute_div(actx, dcoll, quadrature_tag, field, velocity,
                    boundaries, dd_vol):
        """Return divergence for inviscid term."""
        dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
        dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

        itp_f = interior_trace_pairs(dcoll, field, volume_dd=dd_vol,
                                     comm_tag=_MyGradTag)

        itp_u = interior_trace_pairs(dcoll, velocity, volume_dd=dd_vol,
                                     comm_tag=_MyGradTag)

        def interior_flux(f_tpair, u_tpair):
            dd_trace_quad = f_tpair.dd.with_discr_tag(quadrature_tag)
            normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
            bnd_u_tpair_quad = tracepair_with_discr_tag(dcoll, quadrature_tag,
                                                        u_tpair)
            bnd_f_tpair_quad = tracepair_with_discr_tag(dcoll, quadrature_tag,
                                                        f_tpair)

            wavespeed_int = actx.np.sqrt(np.dot(bnd_u_tpair_quad.int,
                                                bnd_u_tpair_quad.int))
            wavespeed_ext = actx.np.sqrt(np.dot(bnd_u_tpair_quad.ext,
                                                bnd_u_tpair_quad.ext))
            lam = actx.np.maximum(wavespeed_int, wavespeed_ext)
            jump = bnd_f_tpair_quad.int - bnd_f_tpair_quad.ext
            numerical_flux = (bnd_f_tpair_quad*bnd_u_tpair_quad).avg + 0.5*lam*jump

            return op.project(dcoll, dd_trace_quad, dd_allfaces_quad,
                              numerical_flux@normal_quad)

        def boundary_flux(bdtag, bdry):
            dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
            normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad))
            int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, field*velocity)

            if bdtag.tag == "prescribed":
                ext_soln_quad = +1.0*int_soln_quad
            if bdtag.tag == "neumann":
                ext_soln_quad = -1.0*int_soln_quad

            bnd_tpair = TracePair(dd_bdry_quad,
                interior=int_soln_quad, exterior=ext_soln_quad)

            return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad,
                              bnd_tpair.avg@normal_quad)

        # pylint: disable=invalid-unary-operand-type
        return -op.inverse_mass(
            dcoll, dd_vol,
            op.weak_local_div(dcoll, dd_vol, field*velocity)
            - op.face_mass(dcoll, dd_allfaces_quad,
                (sum(interior_flux(f_tpair, u_tpair) for f_tpair, u_tpair in
                    zip(itp_f, itp_u))
                + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in
                    boundaries.items()))
            )
        )

    def ablation_workshop_flux(dcoll, state, wall_model, velocity, bprime_class,
                               quadrature_tag, dd_wall, time):
        """Evaluate the prescribed heat flux to be applied at the boundary.

        Function specific for verification against the ablation workshop test
        case 2.1.
        """
        cv = state.cv
        dv = state.dv
        wdv = state.wdv

        actx = cv.mass.array_context

        # restrict variables to the domain boundary
        dd_vol_quad = dd_wall.with_discr_tag(quadrature_tag)
        bdtag = dd_wall.trace("prescribed").domain_tag
        normal_vec = actx.thaw(dcoll.normal(bdtag))
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)

        temperature_bc = op.project(dcoll, dd_wall, dd_bdry_quad, dv.temperature)
        m_dot_g = op.project(dcoll, dd_wall, dd_bdry_quad, cv.mass*velocity)
        emissivity = op.project(dcoll, dd_wall, dd_bdry_quad, wdv.solid_emissivity)

        m_dot_g = np.dot(m_dot_g, normal_vec)

        # time-dependent function
        weight = actx.np.where(actx.np.less(time, 0.1), (time/0.1)+1e-7, 1.0)

        h_e = 1.5e6*weight
        conv_coeff_0 = 0.3*weight

        # this is only valid for the non-ablative case (2.1)
        m_dot_c = 0.0
        m_dot = m_dot_g + m_dot_c + 1e-13

        # ~~~~
        # blowing correction: few iterations to converge the coefficient
        conv_coeff = conv_coeff_0*1.0
        lambda_corr = 0.5
        for _ in range(0, 3):
            phi = 2.0*lambda_corr*m_dot/conv_coeff
            blowing_correction = phi/(actx.np.exp(phi) - 1.0)
            conv_coeff = conv_coeff_0*blowing_correction

        Bsurf = m_dot_g/conv_coeff  # noqa N806

        # ~~~~
        # get the wall enthalpy using spline interpolation
        bnds_T = bprime_class._bounds_T  # noqa N806
        bnds_B = bprime_class._bounds_B  # noqa N806

        # couldn't make lazy work without this
        sys.setrecursionlimit(10000)

        # using spline for temperature interpolation
        # while using "nearest neighbor" for the "B" parameter
        h_w = 0.0
        for j in range(0, 24):
            for k in range(0, 15):
                h_w = \
                    actx.np.where(actx.np.greater_equal(temperature_bc, bnds_T[j]),
                    actx.np.where(actx.np.less(temperature_bc, bnds_T[j+1]),
                    actx.np.where(actx.np.greater_equal(Bsurf, bnds_B[k]),
                    actx.np.where(actx.np.less(Bsurf, bnds_B[k+1]),
                          bprime_class._cs_Hw[k, 0, j]*(temperature_bc-bnds_T[j])**3
                        + bprime_class._cs_Hw[k, 1, j]*(temperature_bc-bnds_T[j])**2
                        + bprime_class._cs_Hw[k, 2, j]*(temperature_bc-bnds_T[j])
                        + bprime_class._cs_Hw[k, 3, j], 0.0), 0.0), 0.0), 0.0) + h_w

        h_g = wall_model.eos.gas_enthalpy(temperature_bc)

        flux = conv_coeff*(h_e - h_w) - m_dot*h_w + m_dot_g*h_g

        radiation = emissivity*5.67e-8*(temperature_bc**4 - 300**4)

        return make_obj_array([flux - radiation])

    def phenolics_operator(dcoll, fluid_state, boundaries, gas_model, pyrolysis,
                           quadrature_tag, dd_wall, time=0.0, bprime_class=None,
                           pressure_scaling_factor=1.0, penalty_amount=1.0):
        """Return the RHS of the composite wall."""

        cv = fluid_state.cv
        wv = fluid_state.wv
        tv = fluid_state.tv
        dv = fluid_state.dv
        wdv = fluid_state.wdv

        zeros = wdv.tau*0.0
        actx = zeros.array_context

        pressure_boundaries, velocity_boundaries = boundaries

        # ~~~~~
        # viscous RHS
        pressure_viscous_rhs, grad_pressure = diffusion_operator(dcoll,
            kappa=tv.pressure_diffusivity,
            boundaries=pressure_boundaries, u=dv.pressure,
            penalty_amount=penalty_amount, return_grad_u=True,
            comm_tag=_PresDiffTag)

        velocity = -(tv.pressure_diffusivity/cv.mass)*grad_pressure

        boundary_flux = ablation_workshop_flux(dcoll, fluid_state, gas_model,
            velocity, bprime_class, quadrature_tag, dd_wall, time)

        # FIXME try to use NS operator for the fluid inside the wall

        energy_boundaries = {
            BoundaryDomainTag("prescribed"):
                PrescribedFluxDiffusionBoundary(boundary_flux),
            BoundaryDomainTag("neumann"):
                NeumannDiffusionBoundary(0.0)
        }

        energy_viscous_rhs = diffusion_operator(dcoll,
            kappa=tv.thermal_conductivity, boundaries=energy_boundaries,
            u=dv.temperature, penalty_amount=penalty_amount,
            comm_tag=_TempDiffTag)

        viscous_rhs = ConservedVars(
            mass=pressure_scaling_factor*pressure_viscous_rhs,
            momentum=cv.momentum*0.0,
            energy=energy_viscous_rhs,
            species_mass=cv.species_mass)

        # ~~~~~
        # inviscid RHS, energy equation only
        field = cv.mass*gas_model.eos.gas_enthalpy(temperature)
        energy_inviscid_rhs = compute_div(actx, dcoll, quadrature_tag, field,
                                          velocity, velocity_boundaries, dd_wall)

        inviscid_rhs = ConservedVars(
            mass=zeros,
            momentum=cv.momentum*0.0,
            energy=-energy_inviscid_rhs,
            species_mass=cv.species_mass)

        # ~~~~~
        # decomposition for each component of the resin
        resin_pyrolysis = pyrolysis.get_source_terms(temperature=dv.temperature,
                                                     chi=wv.mass)

        # flip sign due to mass conservation
        gas_source_term = -pressure_scaling_factor*sum(resin_pyrolysis)

        # viscous dissipation due to friction inside the porous
        visc_diss_energy = tv.viscosity*wdv.void_fraction**2*(
            (1.0/wdv.solid_permeability)*np.dot(velocity, velocity))

        source_terms = ConservedVars(
            mass=gas_source_term,
            momentum=cv.momentum*0.0,
            energy=visc_diss_energy,
            species_mass=cv.species_mass)  # this should be empty in this case

        cv_rhs = inviscid_rhs + viscous_rhs + source_terms
        wv_rhs = WallConservedVars(mass=resin_pyrolysis)

        return cv_rhs, wv_rhs

    def _rhs(t, state):

        cv, wv, tseed = state

        fluid_state = _make_state(cv, gas_model, temperature_seed, wall_vars)

        boundaries = make_obj_array([pressure_boundaries, velocity_boundaries])

        cv_rhs, wv_rhs = phenolics_operator(
            dcoll=dcoll, fluid_state=fluid_state, boundaries=boundaries,
            gas_model=gas_model, pyrolysis=pyrolysis, quadrature_tag=quadrature_tag,
            dd_wall=dd_vol, time=t, bprime_class=bprime_class,
            pressure_scaling_factor=pressure_scaling_factor, penalty_amount=1.0)

        # ~~~~~
        return make_obj_array([cv_rhs, wv_rhs, tseed*0.0])

    compiled_rhs = actx.compile(_rhs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    visualizer = make_visualizer(dcoll)

    def my_write_viz(step, t, state):

        cv = state.cv
        dv = state.dv
        wv = state.wv
        wdv = state.wdv

        viz_fields = [("CV_density", cv.mass),
                      ("CV_energy", cv.energy),
                      ("DV", dv),
                      ("WV_phase_1", wv.mass[0]),
                      ("WV_phase_2", wv.mass[1]),
                      ("WV_phase_3", wv.mass[2]),
                      ("WDV", wdv)
                      ]

        # depending on the version, paraview may complain without this
        viz_fields.append(("x", nodes[0]))

        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
            step=step, t=t, overwrite=True, vis_timer=vis_timer, comm=comm)

    def my_write_restart(step, t, state):
        cv = state.cv
        dv = state.dv
        wv = state.wv
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "wv": wv,
                "tseed": dv.temperature,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "nparts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from warnings import warn
    warn("Running gc.collect() to work around memory growth issue ")
    import gc
    gc.collect()

    my_write_viz(step=istep, t=t, state=fluid_state)

    cv = fluid_state.cv
    dv = fluid_state.dv
    wv = fluid_state.wv

    freeze_gc_flag = True
    while t < t_final:

        if logmgr:
            logmgr.tick_before()

        try:

            state = make_obj_array([cv, wv, fluid_state.temperature])

            state = ssprk43_step(state, t, dt, compiled_rhs)
            state = force_evaluation(actx, state)

            cv, wv, tseed = state
            fluid_state = compiled_make_state(cv, tseed, wv)

            t += dt
            istep += 1

            if check_naninf_local(dcoll, "vol", dv.temperature):
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                raise MyRuntimeError("Failed simulation health check.")

            if istep % nviz == 0:
                my_write_viz(step=istep, t=t, state=fluid_state)

            if istep % ngarbage == 0:
                gc.collect()

            if istep % nrestart == 0:
                my_write_restart(step=istep, t=t, state=fluid_state)

            if freeze_gc_flag is True:
                print("Freeze gc")
                freeze_gc_flag = False

                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                gc.freeze()

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=istep, t=t, state=fluid_state)
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

    main(actx_class, use_logmgr=args.log, lazy=lazy,
         use_profiling=args.profiling, casename=casename,
         restart_file=rst_filename)

# vim: foldmethod=marker
