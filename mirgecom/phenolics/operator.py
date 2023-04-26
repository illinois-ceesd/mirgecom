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

import numpy as np
from grudge.trace_pair import (
    TracePair, interior_trace_pairs, tracepair_with_discr_tag
)
from grudge import op
from meshmode.discretization.connection import FACE_RESTR_ALL

from mirgecom.diffusion import (
    diffusion_operator,
    PrescribedFluxDiffusionBoundary,
    NeumannDiffusionBoundary
)

from pytools.obj_array import make_obj_array

from grudge.dof_desc import BoundaryDomainTag

import sys

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

        if dcoll.dim == 1:
            wavespeed_int = bnd_u_tpair_quad.int[0]
            wavespeed_ext = bnd_u_tpair_quad.ext[0]
        if dcoll.dim == 2:
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
                       quadrature_tag, dd_wall, bprime_class,
                       pressure_scaling_factor=1.0, penalty_amount=1.0,
                       split=False):
    """."""
    wv, tseed = state
    wdv = eos.dependent_vars(wv=wv, temperature_seed=tseed)

    tau = wdv.tau
    kappa = wdv.thermal_conductivity
    temperature = wdv.temperature
    pressure = wdv.gas_pressure

    zeros = tau*0.0

    actx = zeros.array_context

    pressure_boundaries, velocity_boundaries = boundaries

    gas_pressure_diffusivity = eos.gas_pressure_diffusivity(temperature, tau)

    # ~~~~~
    pressure_viscous_rhs, grad_pressure = diffusion_operator(dcoll,
        kappa=wv.gas_density*gas_pressure_diffusivity,
        boundaries=pressure_boundaries, u=pressure, time=time,
        penalty_amount=penalty_amount, return_grad_u=True)

    velocity = -gas_pressure_diffusivity*grad_pressure

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # restrict temperature and momentum to the domain boundary
    dd_vol_quad = dd_wall.with_discr_tag(quadrature_tag)
    bdtag = dd_wall.trace('neumann').domain_tag
    normal_vec = actx.thaw(dcoll.normal(bdtag))
    dd_bdry_quad = dd_vol_quad.with_domain_tag(bdtag)

    temperature_bc = op.project(dcoll, dd_wall, dd_bdry_quad, temperature)
    m_dot_g = op.project(dcoll, dd_wall, dd_bdry_quad,
                         wv.gas_density*velocity)

    # TODO double-check this
    m_dot_g = np.dot(m_dot_g, normal_vec)

    # ~~~~
    # time-dependent function
    weight = actx.np.where(actx.np.less(time, 0.1), (time/0.1)+1e-13, 1.0 )

    h_e = 1.5e6*weight
    conv_coeff_0 = 0.3*weight

    # ~~~~
    m_dot_c = 0.0
    m_dot = m_dot_g + m_dot_c + 1e-13
    mass_flux = [m_dot_c, m_dot_g]

    # ~~~~
    # FIXME add blowing correction
    conv_coeff = conv_coeff_0*1.0
#    lambda_corr = 0.5
#    phi = 2*lambda_corr*m_dot/conv_coeff
#    blowing_correction = phi/(np.exp(phi) - 1)

#    conv_coeff = conv_coeff_0*blowing_correction

    Bsurf = m_dot_g/conv_coeff

    # ~~~~
    # get the wall enthalpy
    bnds_T = bprime_class._bounds_T[:-1:6]
    bnds_B = bprime_class._bounds_B

    # couldn't make lazy work without this
    import sys
    sys.setrecursionlimit(4000)

    h_w = 0.0
    for j in range(0,24):
        for k in range(0,15):
            h_w = actx.np.where(actx.np.greater_equal(temperature_bc, bnds_T[j]),
                actx.np.where(actx.np.less(temperature_bc, bnds_T[j+1]),
                    actx.np.where(actx.np.greater_equal(Bsurf, bnds_B[k]),
                        actx.np.where(actx.np.less(Bsurf, bnds_B[k+1]),
                              bprime_class._cs_H[k, 0, j]*(temperature_bc-bnds_T[j])**3
                            + bprime_class._cs_H[k, 1, j]*(temperature_bc-bnds_T[j])**2
                            + bprime_class._cs_H[k, 2, j]*(temperature_bc-bnds_T[j])
                            + bprime_class._cs_H[k, 3, j], 0.0), 0.0), 0.0), 0.0) + h_w

    # ~~~~
    h_g = eos.gas_enthalpy(temperature_bc)

    flux = conv_coeff*(h_e - h_w) - m_dot*h_w + m_dot_g*h_g

    # FIXME make emissivity function of tau
    emissivity = 0.8
    radiation = emissivity*5.67e-8*(temperature_bc**4 - 300**4)
    boundary_flux = make_obj_array([flux - radiation])

    energy_boundaries = {
        BoundaryDomainTag("prescribed"):
            PrescribedFluxDiffusionBoundary(boundary_flux),
        BoundaryDomainTag("neumann"):
            NeumannDiffusionBoundary(0.0)
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    energy_viscous_rhs = diffusion_operator(dcoll, kappa=kappa,
        boundaries=energy_boundaries, u=temperature, time=time,
        penalty_amount=penalty_amount)

    viscous_rhs = wall.make_conserved(
        solid_species_mass=wv.solid_species_mass*0.0,
        gas_density=pressure_scaling_factor*pressure_viscous_rhs,
        energy=energy_viscous_rhs)

    # ~~~~~
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

    actx = zeros.array_context
    visc_diss_energy = (
        wdv.gas_viscosity*wdv.void_fraction**2*(1.0/wdv.solid_permeability)*
        np.dot(velocity, velocity)
    )

    source_terms = wall.make_conserved(
        solid_species_mass=resin_pyrolysis,
        gas_density=gas_source_term,
        energy=visc_diss_energy)

    # ~~~~~
    if split:
        return inviscid_rhs, viscous_rhs, source_terms
    return inviscid_rhs + viscous_rhs + source_terms
