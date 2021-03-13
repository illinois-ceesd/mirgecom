""":mod:`mirgecom.fluid` provides common utilities for fluid simulation.

State Vector Handling
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ConservedVars
.. autofunction:: split_conserved
.. autofunction:: join_conserved

Helper Functions
^^^^^^^^^^^^^^^^

.. autofunction:: compute_local_velocity_gradient
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
import numpy as np  # noqa
from meshmode.dof_array import DOFArray  # noqa
from dataclasses import dataclass


@dataclass(frozen=True)
class ConservedVars:
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
    species_mass: np.ndarray = np.empty((0,), dtype=object)  # empty = immutable

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
    nspec = get_num_species(dim, q)
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim],
                         species_mass=q[2+dim:2+dim+nspec])


def join_conserved(dim, mass, energy, momentum,
        # empty: immutable
        species_mass=np.empty((0,), dtype=object)):
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


def compute_local_velocity_gradient(discr, cv: ConservedVars):
    r"""
    Compute the cell-local gradient of fluid velocity.

    Computes the cell-local gradient of fluid velocity from:

    .. math::

        \nabla{v_i} = \frac{1}{\rho}(\nabla(\rho{v_i})-v_i\nabla{\rho}),

    where $v_i$ is ith velocity component.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    cv: mirgecom.fluid.ConservedVars
        the fluid conserved variables
    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray`
        representing $\partial_j{v_i}$.
    """
    dim = discr.dim
    velocity = cv.momentum/cv.mass
    dmass = discr.grad(cv.mass)
    dmom = [discr.grad(cv.momentum[i]) for i in range(dim)]
    return [(dmom[i] - velocity[i]*dmass)/cv.mass for i in range(dim)]
