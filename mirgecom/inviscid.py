r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Flux Calculation
^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux

Time Step Computation
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_inviscid_timestep
.. autofunction:: get_inviscid_cfl
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
from mirgecom.fluid import (
    split_conserved,
    join_conserved
)


def inviscid_flux(discr, eos, q):
    r"""Compute the inviscid flux vectors from flow solution *q*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$

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
    p = eos.pressure(cv)

    mom = cv.momentum

    return join_conserved(dim,
            mass=mom,
            energy=mom * (cv.energy + p) / cv.mass,
            momentum=np.outer(mom, mom) / cv.mass + np.eye(dim)*p,
            species_mass=(  # reshaped: (nspecies, dim)
                (mom / cv.mass) * cv.species_mass.reshape(-1, 1)))


def get_inviscid_timestep(discr, eos, q):
    """Routine returns the node-local maximum stable inviscid timestep.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    q: numpy.ndarray
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.

    Returns
    -------
    class:`~meshmode.dof_array.DOFArray`
        The maximum stable timestep at each node.
    """
    from grudge.dt_utils import characteristic_length_scales
    from mirgecom.fluid import compute_wavespeed

    dim = discr.dim
    cv = split_conserved(dim, q)
    return (
        characteristic_length_scales(discr)/compute_wavespeed(discr, eos, cv)
    )


def get_inviscid_cfl(discr, eos, dt, q):
    """Calculate and return node-local CFL based on current state and timestep.

    Parameters
    ----------
    discr: :class:`grudge.eager.EagerDGDiscretization`
        the discretization to use
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    dt: float or :class:`~meshmode.dof_array.DOFArray`
        A constant scalar dt or node-local dt
    q: numpy.ndarray
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.

    Returns
    -------
    :class:`meshmode.dof_array.DOFArray`
        The CFL at each node.
    """
    return dt / get_inviscid_timestep(discr, eos=eos, q=q)
