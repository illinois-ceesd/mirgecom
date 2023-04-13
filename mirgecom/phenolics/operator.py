r""":mod:`mirgecom.phenolics.operator`

.. autofunction:: my_derivative_function
.. autofunction:: phenolics_operator
"""

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

from grudge.trace_pair import TracePair
from grudge.trace_pair import interior_trace_pairs, tracepair_with_discr_tag
from grudge import op
from meshmode.discretization.connection import FACE_RESTR_ALL

from mirgecom.diffusion import diffusion_operator


class _MyGradTag:
    pass


def my_derivative_function(actx, dcoll, quadrature_tag, field, u,
                           boundaries, dd_vol):
    """."""
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    itp_f = interior_trace_pairs(dcoll, field, volume_dd=dd_vol,
                                     comm_tag=_MyGradTag)

    itp_u = interior_trace_pairs(dcoll, u, volume_dd=dd_vol,
                                     comm_tag=_MyGradTag)

    flux = field*u

    def interior_flux(f_tpair, u_tpair):
        dd_trace_quad = f_tpair.dd.with_discr_tag(quadrature_tag)
        normal_quad = actx.thaw(dcoll.normal(dd_trace_quad))
        bnd_u_tpair_quad = tracepair_with_discr_tag(dcoll,
            quadrature_tag, u_tpair)
        bnd_f_tpair_quad = tracepair_with_discr_tag(dcoll,
            quadrature_tag, f_tpair)

        numerical_flux = (bnd_f_tpair_quad*bnd_u_tpair_quad).avg

        wavespeed_int = actx.np.sqrt(
            bnd_u_tpair_quad.int[0]**2 + bnd_u_tpair_quad.int[1]**2)
        wavespeed_ext = actx.np.sqrt(
            bnd_u_tpair_quad.ext[0]**2 + bnd_u_tpair_quad.ext[1]**2)
        lam = actx.np.maximum(wavespeed_int, wavespeed_ext)
        jump = bnd_f_tpair_quad.int - bnd_f_tpair_quad.ext
        numerical_flux = numerical_flux + 0.5*lam*(jump)

        flux_int = numerical_flux@normal_quad

        return op.project(dcoll, dd_trace_quad, dd_allfaces_quad, flux_int)

    def boundary_flux(bdtag, bdry):
        dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)
        normal_quad = actx.thaw(dcoll.normal(dd_bdry_quad))
        int_soln_quad = op.project(dcoll, dd_vol, dd_bdry_quad, flux)

        # FIXME make this more organized
        if (bdtag.tag == "prescribed"):
            ext_soln_quad = +1.0*int_soln_quad
        if (bdtag.tag == "neumann"):
            ext_soln_quad = -1.0*int_soln_quad

        bnd_tpair = TracePair(dd_bdry_quad,
            interior=int_soln_quad, exterior=ext_soln_quad)

        flux_bnd = bnd_tpair.avg@normal_quad

        return op.project(dcoll, dd_bdry_quad, dd_allfaces_quad, flux_bnd)

    return -op.inverse_mass(
        dcoll, dd_vol,
        op.weak_local_div(dcoll, dd_vol, flux)
        - op.face_mass(dcoll, dd_allfaces_quad,
            (sum(interior_flux(f_tpair, u_tpair) for f_tpair, u_tpair in
                zip(itp_f, itp_u))
            + sum(boundary_flux(bdtag, bdry) for bdtag, bdry in
                boundaries.items()))
        )
    )


def phenolics_operator(dcoll, state, boundaries, time, wall, eos, pyrolysis,
                       pressure_scaling_factor, quadrature_tag, dd_wall):
    """."""
    wv, tseed = state
    wdv = eos.dependent_vars(wv=wv, temperature_seed=tseed)

    tau = wdv.tau
    kappa = wdv.thermal_conductivity
    temperature = wdv.temperature
    pressure = wdv.gas_pressure

    zeros = tau*0.0

    pressure_boundaries, energy_boundaries, velocity_boundaries = boundaries

    # ~~~~~
    gas_pressure_diffusivity = eos.gas_pressure_diffusivity(temperature, tau)
    pressure_viscous_rhs, grad_pressure = diffusion_operator(dcoll,
        kappa=wv.gas_density*gas_pressure_diffusivity,
        boundaries=pressure_boundaries, u=pressure, time=time,
        return_grad_u=True)

    energy_viscous_rhs = diffusion_operator(dcoll, kappa=kappa,
        boundaries=energy_boundaries, u=temperature, time=time)

    viscous_rhs = wall.make_conserved(
        solid_species_mass=wv.solid_species_mass*0.0,
        gas_density=pressure_scaling_factor*pressure_viscous_rhs,
        energy=energy_viscous_rhs)

    # ~~~~~
    velocity = -gas_pressure_diffusivity*grad_pressure

    actx = velocity[0].array_context
    field = wv.gas_density*wdv.gas_enthalpy
    energy_inviscid_rhs = my_derivative_function(actx, dcoll, quadrature_tag,
            field, velocity, velocity_boundaries, dd_wall)

    inviscid_rhs = wall.make_conserved(
        solid_species_mass=wv.solid_species_mass*0.0,
        gas_density=zeros,
        energy=-energy_inviscid_rhs
    )

    # ~~~~~
    resin_pyrolysis = pyrolysis.get_sources(temperature,
                                            wv.solid_species_mass)

    # flip sign due to mass conservation
    gas_source_term = -pressure_scaling_factor*sum(resin_pyrolysis)

    source_terms = wall.make_conserved(
        solid_species_mass=resin_pyrolysis,
        gas_density=gas_source_term,
        energy=zeros)

    return inviscid_rhs + viscous_rhs + source_terms
