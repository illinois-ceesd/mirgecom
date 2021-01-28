r""":mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics.

Euler's equations of gas dynamics:

.. math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_{\partial\Omega} + \mathbf{S}

where:

-  state $\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  flux $\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
   (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I}), \rho{Y}_\alpha\vec{V}]$,
-  unit normal $\hat{n}$ to the domain boundary $\partial\Omega$,
-  sources $\mathbf{S} = [{s}_\rho, {s}_e, \mathbf{s}_p, \mathbf{s}_s]$
-  vector of species mass fractions ${Y}_\alpha$,
   with $1\le\alpha\le\mathtt{nspecies}$.

State Vector Handling
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ConservedVars
.. autofunction:: split_conserved
.. autofunction:: join_conserved

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_operator

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

from dataclasses import dataclass

import numpy as np
from meshmode.dof_array import thaw, DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)


@dataclass(frozen=True)
class ConservedVars:  # FIXME: Name?
    r"""Resolve the canonical conserved quantities.

    Get the canonical conserved quantities (mass, energy, momentum,
    and species masses) per unit volume = $(\rho,\rho{E},\rho\vec{V},
    \rho{Y_s})$ from an agglomerated object array.

    .. attribute:: dim

        Integer indicating spatial dimension of the state

    .. attribute:: mass

        :class:`~meshmode.dof_array.DOFArray` for the fluid mass per unit volume

    .. attribute:: energy

        :class:`~meshmode.dof_array.DOFArray` for total energy per unit volume

    .. attribute:: momentum

        Object array (:class:`~numpy.ndarray`) with shape ``(ndim,)``
        of :class:`~meshmode.dof_array.DOFArray` for momentum per unit volume.

    .. attribute:: species_mass

        Object array (:class:`~numpy.ndarray`) with shape ``(nspecies,)``
        of :class:`~meshmode.dof_array.DOFArray` for species mass per unit volume.
        The species mass vector has components, $\rho~Y_\alpha$, where $Y_\alpha$
        is the vector of species mass fractions.

    .. automethod:: join
    .. automethod:: replace
    """

    mass: DOFArray
    energy: DOFArray
    momentum: np.ndarray
    species_mass: np.ndarray = np.empty((0,), dtype=object)

    @property
    def dim(self):
        """Return the number of physical dimensions."""
        return len(self.momentum)

    def join(self):
        """Call :func:`join_conserved` on *self*."""
        return join_conserved(
            dim=self.dim,
            mass=self.mass,
            energy=self.energy,
            momentum=self.momentum,
            species_mass=self.species_mass)

    def replace(self, **kwargs):
        """Return a copy of *self* with the attributes in *kwargs* replaced."""
        from dataclasses import replace
        return replace(self, **kwargs)


def _aux_shape(ary, leading_shape):
    """:arg leading_shape: a tuple with which ``ary.shape`` is expected to begin."""
    from meshmode.dof_array import DOFArray
    if (isinstance(ary, np.ndarray) and ary.dtype == np.object
            and not isinstance(ary, DOFArray)):
        naxes = len(leading_shape)
        if ary.shape[:naxes] != leading_shape:
            raise ValueError("array shape does not start with expected leading "
                    "dimensions")
        return ary.shape[naxes:]
    else:
        if leading_shape != ():
            raise ValueError("array shape does not start with expected leading "
                    "dimensions")
        return ()


def get_num_species(dim, q):
    """Return number of mixture species."""
    return len(q) - (dim + 2)


def split_conserved(dim, q):
    """Get the canonical conserved quantities.

    Return a :class:`ConservedVars` that is the canonical conserved quantities,
    mass, energy, momentum, and any species' masses, from the agglomerated
    object array extracted from the state vector *q*. For single component gases,
    i.e. for those state vectors *q* that do not contain multi-species mixtures, the
    returned dataclass :attr:`ConservedVars.species_mass` will be set to an empty
    array.
    """
    #    assert len(q) == dim + 2 + get_num_species(dim, q)
    nspec = get_num_species(dim, q)
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim],
                         species_mass=q[2+dim:2+dim+nspec])


def join_conserved(dim, mass, energy, momentum, species_mass=np.empty((0,),
        dtype=object)):
    """Create an agglomerated solution array from the conserved quantities."""
    nspec = len(species_mass)
    aux_shapes = [
        _aux_shape(mass, ()),
        _aux_shape(energy, ()),
        _aux_shape(momentum, (dim,)),
        _aux_shape(species_mass, (nspec,))]

    from pytools import single_valued
    aux_shape = single_valued(aux_shapes)

    result = np.empty((2+dim+nspec,) + aux_shape, dtype=object)
    result[0] = mass
    result[1] = energy
    result[2:dim+2] = momentum
    result[dim+2:] = species_mass

    return result


def inviscid_flux(discr, eos, q):
    r"""Compute the inviscid flux vectors from flow solution *q*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$
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


def _get_wavespeed(dim, eos, cv: ConservedVars):
    """Return the maximum wavespeed in for flow solution *q*."""
    actx = cv.mass.array_context

    v = cv.momentum / cv.mass
    return actx.np.sqrt(np.dot(v, v)) + eos.sound_speed(cv)


def _facial_flux(discr, eos, q_tpair, local=False):
    """Return the flux across a face given the solution on both sides *q_tpair*.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    q_tpair: :class:`grudge.symbolic.TracePair`
        Trace pair for the face upon which flux calculation is to be performed

    local: bool
        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.
    """
    dim = discr.dim

    actx = q_tpair[0].int.array_context

    flux_int = inviscid_flux(discr, eos, q_tpair.int)
    flux_ext = inviscid_flux(discr, eos, q_tpair.ext)

    # Lax-Friedrichs/Rusanov after [Hesthaven_2008]_, Section 6.6
    flux_avg = 0.5*(flux_int + flux_ext)

    lam = actx.np.maximum(
        _get_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.int)),
        _get_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.ext))
    )

    normal = thaw(actx, discr.normal(q_tpair.dd))
    flux_weak = (
        flux_avg @ normal
        - 0.5 * lam * (q_tpair.ext - q_tpair.int))

    if local is False:
        return discr.project(q_tpair.dd, "all_faces", flux_weak)
    return flux_weak


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    r"""Compute RHS of the Euler flow equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = \mathbf{S} - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

    Parameters
    ----------
    q
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.

    boundaries
        Dictionary of boundary functions, one for each valid btag

    t
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    vol_flux = inviscid_flux(discr, eos, q)
    dflux = discr.weak_div(vol_flux)

    interior_face_flux = _facial_flux(
        discr, eos=eos, q_tpair=interior_trace_pair(discr, q))

    # Domain boundaries
    domain_boundary_flux = sum(
        _facial_flux(
            discr,
            q_tpair=boundaries[btag].boundary_pair(discr,
                                                   eos=eos,
                                                   btag=btag,
                                                   t=t,
                                                   q=q),
            eos=eos
        )
        for btag in boundaries
    )

    # Flux across partition boundaries
    partition_boundary_flux = sum(
        _facial_flux(discr, eos=eos, q_tpair=part_pair)
        for part_pair in cross_rank_trace_pairs(discr, q)
    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux
                                + partition_boundary_flux)
    )


def get_inviscid_cfl(discr, eos, dt, q):
    """Calculate and return CFL based on current state and timestep."""
    wanted_dt = get_inviscid_timestep(discr, eos=eos, cfl=1.0, q=q)
    return dt / wanted_dt


def get_inviscid_timestep(discr, eos, cfl, q):
    """Routine (will) return the (local) maximum stable inviscid timestep.

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
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
