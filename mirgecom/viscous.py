r""":mod:`mirgecom.viscous` provides helper functions for viscous flow.

Flux Calculation
^^^^^^^^^^^^^^^^

.. autofunction:: viscous_flux
.. autofunction:: viscous_stress_tensor
.. autofunction:: diffusive_flux
.. autofunction:: conductive_heat_flux
.. autofunction:: diffusive_heat_flux
.. autofunction:: viscous_facial_flux

Time Step Computation
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_viscous_timestep
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
from pytools.obj_array import make_obj_array
from mirgecom.fluid import (
    split_conserved,
    join_conserved,
    velocity_gradient,
    species_mass_fraction_gradient
)
from meshmode.dof_array import thaw


def viscous_stress_tensor(discr, eos, q, grad_q):
    """Compute the viscous stress tensor."""
    dim = discr.dim
    cv = split_conserved(dim, q)
    grad_cv = split_conserved(dim, grad_q)
    transport = eos.transport_model()

    mu_b = transport.bulk_viscosity(eos, cv)
    mu = transport.viscosity(eos, cv)

    grad_v = velocity_gradient(discr, cv, grad_cv)
    div_v = np.trace(grad_v)

    return mu*(grad_v + grad_v.T) + (mu_b - 2*mu/3)*div_v*np.eye(dim)


def diffusive_flux(discr, eos, q, grad_q):
    r"""Compute the species diffusive flux vector, ($\mathbf{J}_{\alpha}$).

    The species diffussive flux is defined by:

    .. math::

        \mathbf{J}_{\alpha} = -\rho{d}_{(\alpha)}\nabla{Y_{\alpha}}~~
        (\mathtt{no~implied~sum}),

    with species diffusivities ${d}_{\alpha}$, and species mass
    fractions ${Y}_{\alpha}$.
    """
    cv = split_conserved(discr.dim, q)
    grad_cv = split_conserved(discr.dim, grad_q)
    nspecies = len(cv.species_mass)
    transport = eos.transport_model()

    grad_y = species_mass_fraction_gradient(discr, cv, grad_cv)
    d = transport.species_diffusivity(eos, cv)

    return -make_obj_array([cv.mass*d[i]*grad_y[i] for i in range(nspecies)])


def conductive_heat_flux(discr, eos, q, grad_t):
    r"""Compute the conductive heat flux, ($\mathbf{q}_{c}$).

    The conductive heat flux is defined by:

    .. math::

        \mathbf{q}_{c} = \kappa\nabla{T},

    with thermal conductivity $\kappa$, and gas temperature $T$.
    """
    cv = split_conserved(discr.dim, q)
    transport = eos.transport_model()
    return transport.thermal_conductivity(eos, cv)*grad_t


def diffusive_heat_flux(discr, eos, q, j):
    r"""Compute the diffusive heat flux, ($\mathbf{q}_{d}$).

    The diffusive heat flux is defined by:

    .. math::

        \mathbf{q}_{d} = \sum_{\alpha=1}^{\mathtt{Nspecies}}{j}_{\alpha}
        \mathbf{J}_{\alpha},

    with species diffusive flux ($\mathbf{J}_{\alpha}$) defined as:

    .. math::

        \mathbf{J}_{\alpha} = -\rho{d}_{\alpha}\nabla{Y}_{\alpha},

    where ${Y}_{\alpha}$ is the vector of species mass fractions.
    """
    cv = split_conserved(discr.dim, q)
    numspecies = len(cv.species_mass)
    transport = eos.transport_model()
    d = transport.species_diffusivity(eos, cv)
    return sum(d[i]*j[i] for i in range(numspecies))


def viscous_flux(discr, eos, q, grad_q, t, grad_t):
    r"""Compute the viscous flux vectors.

    The viscous fluxes are:

    .. math::

        \mathbf{F}_V = [0,\tau\cdot\mathbf{v} - \mathbf{q},
        \tau_{:i},-\mathbf{j}_\alpha],

    with fluid velocity ($\mathbf{v}$), viscous stress tensor
    ($\mathbf{\tau}$), and diffusive flux for each species
    ($\mathbf{j}_\alpha$).

    .. note::

        The fluxes are returned as a 2D object array with shape:
        ``(num_equations, ndim)``.  Each entry in the
        flux array is a :class:`~meshmode.dof_array.DOFArray`.  This
        form and shape for the flux data is required by the built-in
        state data handling mechanism in :mod:`mirgecom.fluid`. That
        mechanism is used by at least
        :class:`mirgecom.fluid.ConservedVars`, and
        :func:`mirgecom.fluid.join_conserved`, and
        :func:`mirgecom.fluid.split_conserved`.
    """
    dim = discr.dim
    cv = split_conserved(dim, q)

    j = diffusive_flux(discr, eos, q, grad_q)
    heat_flux = conductive_heat_flux(discr, eos, q, grad_t)
    #    heat_flux = (conductive_heat_flux(discr, eos, q, grad_t)
    #                 + diffusive_heat_flux(discr, eos, q, j))
    vel = cv.momentum / cv.mass
    tau = viscous_stress_tensor(discr, eos, q, grad_q)
    viscous_mass_flux = 0 * cv.momentum
    viscous_energy_flux = np.dot(tau, vel) - heat_flux

    # passes the right shape for diffusive flux when no species
    # TODO: fix single gas join_conserved for vectors at each cons eqn
    if len(j) == 0:
        j = cv.momentum * cv.species_mass.reshape(-1, 1)

    return join_conserved(dim,
            mass=viscous_mass_flux,
            energy=viscous_energy_flux,
            momentum=tau, species_mass=-j)


def viscous_facial_flux(discr, eos, q_tpair, grad_q_tpair,
                        t_tpair, grad_t_tpair, local=False):
    """Return the viscous flux across a face given the solution on both sides.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    q_tpair: :class:`grudge.sym.TracePair`
        Trace pair for the face upon which flux calculation is to be performed

    local: bool
        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.
    """
    actx = q_tpair[0].int.array_context
    normal = thaw(actx, discr.normal(q_tpair.dd))

    # todo: user-supplied flux routine
    f_int = viscous_flux(discr, eos, q_tpair.int, grad_q_tpair.int, t_tpair.int,
                         grad_t_tpair.int)
    f_ext = viscous_flux(discr, eos, q_tpair.ext, grad_q_tpair.ext, t_tpair.ext,
                         grad_t_tpair.ext)
    f_avg = 0.5*(f_int + f_ext)
    flux_weak = f_avg @ normal

    if local is False:
        return discr.project(q_tpair.dd, "all_faces", flux_weak)
    return flux_weak


def get_viscous_timestep(discr, eos, transport, cfl, q):
    """Routine (will) return the (local) maximum stable viscous timestep.

    Currently, it's a hack waiting for the geometric_factor helpers port
    from grudge.
    """
    dim = discr.dim
    mesh = discr.mesh
    order = max([grp.order for grp in discr.discr_from_dd("vol").groups])
    nelements = mesh.nelements
    nel_1d = nelements ** (1.0 / (1.0 * dim))

    # This roughly reproduces the timestep AK used in wave toy
    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return cfl * dt
#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = compute_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
