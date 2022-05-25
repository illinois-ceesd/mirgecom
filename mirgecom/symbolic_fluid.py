""":mod:`mirgecom.symbolic_fluid` provides symbolic versions of fluid constructs.

Symbolic fluxes
^^^^^^^^^^^^^^^
.. autofunction:: sym_inviscid_flux
.. autofunction:: sym_viscous_flux
.. autofunction:: sym_diffusive_flux
.. autofunction:: sym_heat_flux
.. autofunction:: sym_viscous_stress_tensor

Symbolic fluid operators
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sym_euler
.. autofunction:: sym_ns
"""
import numpy as np
import numpy.random

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import make_conserved

from mirgecom.symbolic import (
    grad as sym_grad,
    div as sym_div)

# from mirgecom.eos import IdealSingleGas
# from mirgecom.transport import SimpleTransport


def sym_inviscid_flux(sym_cv, sym_pressure):
    """Return symbolic expression for inviscid flux."""
    return make_conserved(
        dim=sym_cv.dim,
        mass=sym_cv.momentum,
        momentum=(sym_cv.mass*(np.outer(sym_cv.velocity, sym_cv.velocity))
                  + sym_pressure*np.eye(sym_cv.dim)),
        energy=(sym_cv.energy + sym_pressure)*sym_cv.velocity,
        species_mass=sym_cv.species_mass.reshape(-1, 1)*sym_cv.velocity)


def sym_euler(sym_cv, sym_pressure):
    """Return symbolic expression for the NS operator applied to a fluid state."""
    return -sym_div(sym_cv.dim, sym_inviscid_flux(sym_cv, sym_pressure))


def sym_viscous_stress_tensor(sym_cv, mu=1):
    """Symbolic version of the viscous stress tensor."""
    dvel = sym_grad(sym_cv.dim, sym_cv.velocity)
    return mu*((dvel + dvel.T) - (2./3.)*(dvel.trace() * np.eye(sym_cv.dim)))


def sym_diffusive_flux(sym_cv, species_diffusivities=None):
    """Symbolic diffusive flux calculator."""
    if species_diffusivities is None:
        return 0*sym_cv.velocity*sym_cv.species_mass.reshape(-1, 1)
    return -(sym_cv.mass * species_diffusivities.reshape(-1, 1)
             * sym_grad(sym_cv.dim, sym_cv.species_mass_fractions))


# Diffusive heat flux is neglected atm.  Full multispecies
# support in the symbolic infrastructure is a WIP.
# TODO: Add diffusive heat flux
def sym_heat_flux(dim, sym_temperature, kappa=0):
    """Symbolic heat flux calculator."""
    return -kappa * sym_grad(dim, sym_temperature)


def sym_viscous_flux(sym_cv, sym_temperature, mu=1, kappa=0,
                     species_diffusivities=None):
    """Return symbolic version of viscous flux."""
    dim = sym_cv.dim
    rho = sym_cv.mass
    mom = sym_cv.momentum
    vel = mom/rho

    # viscous stress tensor = momentum flux
    tau = sym_viscous_stress_tensor(sym_cv, mu=mu)

    # energy flux : viscous + heat_flux
    e_flux = np.dot(tau, vel) - sym_heat_flux(dim, sym_temperature, kappa=kappa)

    # species fluxes
    sp_flux = sym_diffusive_flux(sym_cv, species_diffusivities=species_diffusivities)

    return make_conserved(dim=dim, mass=0*mom, energy=e_flux, momentum=tau,
                          species_mass=-sp_flux)


def sym_ns(sym_cv, sym_pressure, sym_temperature, mu=1, kappa=0,
           species_diffusivities=None):
    """Symbolic Navier-Stokes operator."""
    viscous_flux = sym_viscous_flux(sym_cv, sym_temperature, mu=mu, kappa=kappa,
                                    species_diffusivities=species_diffusivities)
    inviscid_flux = sym_inviscid_flux(sym_cv, sym_pressure)
    return sym_div(sym_cv.dim, viscous_flux - inviscid_flux)
