""":mod:`mirgecom.fluid` provides common utilities for fluid simulation.

State Vector Handling
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ConservedVars
.. autofunction:: split_conserved
.. autofunction:: join_conserved

Helper Functions
^^^^^^^^^^^^^^^^

.. autofunction:: compute_wavespeed
.. autofunction:: velocity_gradient
.. autofunction:: species_mass_fraction_gradient
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
import numpy as np  # noqa
from pytools.obj_array import make_obj_array
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
    if (isinstance(ary, np.ndarray) and ary.dtype == object
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


def velocity_gradient(discr, cv, grad_cv):
    r"""
    Compute the gradient of fluid velocity.

    Computes the gradient of fluid velocity from:

    .. math::

        \nabla{v_i} = \frac{1}{\rho}(\nabla(\rho{v_i})-v_i\nabla{\rho}),

    where $v_i$ is ith velocity component.

    .. note::
        The product rule is used to evaluate gradients of the primitive variables
        from the existing data of the gradient of the fluid solution,
        $\nabla\mathbf{Q}$, following [Hesthaven_2008]_, section 7.5.2. If something
        like BR1 ([Bassi_1997]_) is done to treat the viscous terms, then
        $\nabla{\mathbf{Q}}$ should be naturally available.

        Some advantages of doing it this way:

        * avoids an additional DG gradient computation
        * enables the use of a quadrature discretization for computation
        * jibes with the already-applied bcs of $\mathbf{Q}$

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    cv: ConservedVars
        the fluid conserved variables
    grad_cv: ConservedVars
        the gradients of the fluid conserved variables

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray`
        for each row of $\partial_j{v_i}$. e.g. for 2D:
        $\left( \begin{array}{cc}
        \partial_{x}\mathbf{v}_{x}&\partial_{y}\mathbf{v}_{x} \\
        \partial_{x}\mathbf{v}_{y}&\partial_{y}\mathbf{v}_{y} \end{array} \right)$

    """
    velocity = cv.momentum / cv.mass
    return (1/cv.mass)*make_obj_array([grad_cv.momentum[i]
                                       - velocity[i]*grad_cv.mass
                                       for i in range(discr.dim)])


def species_mass_fraction_gradient(discr, cv, grad_cv):
    r"""
    Compute the gradient of species mass fractions.

    Computes the gradient of species mass fractions from:

    .. math::

        \nabla{Y}_{\alpha} =
        \frac{1}{\rho}\left(\nabla(\rho{Y}_{\alpha})-{Y_\alpha}(\nabla{\rho})\right),

    where ${Y}_{\alpha}$ is the mass fraction for species ${\alpha}$.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    cv: ConservedVars
        the fluid conserved variables
    grad_cv: ConservedVars
        the gradients of the fluid conserved variables

    Returns
    -------
    numpy.ndarray
        object array of :class:`~meshmode.dof_array.DOFArray`
        representing $\partial_j{Y}_{\alpha}$.
    """
    nspecies = len(cv.species_mass)
    y = cv.species_mass / cv.mass
    return (1/cv.mass)*make_obj_array([grad_cv.species_mass[i]
                                       - y[i]*grad_cv.mass
                                       for i in range(nspecies)])


def compute_wavespeed(dim, eos, cv: ConservedVars):
    r"""Return the wavespeed in the flow.

    The wavespeed is calculated as:

    .. math::

        s_w = \|\mathbf{v}\| + c,

    where $\mathbf{v}$ is the flow velocity and c is the speed of sound in the fluid.
    """
    actx = cv.mass.array_context

    v = cv.momentum / cv.mass
    return actx.np.sqrt(np.dot(v, v)) + eos.sound_speed(cv)
