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
    r"""Store and resolve quantities according to the fluid conservation equations.

    Store and resolve quantities that correspond to the fluid conservation equations
    for the canonical conserved quantities (mass, energy, momentum,
    and species masses) per unit volume: $(\rho,\rho{E},\rho\vec{V},
    \rho{Y_s})$ from an agglomerated object array.  This data structure is intimately
    related to helper functions :func:`join_conserved` and :func:`split_conserved`,
    which pack and unpack (respectively) quantities to-and-from flat object
    array representations of the data.

    .. note::

        The use of the terms pack and unpack here is misleading. In reality, there
        is no data movement when using this dataclass to view your data. This
        dataclass is entirely about the representation of the data.

    .. attribute:: dim

        Integer indicating spatial dimension of the simulation

    .. attribute:: mass

        :class:`~meshmode.dof_array.DOFArray` for scalars or object array of
        :class:`~meshmode.dof_array.DOFArray` for vector quantities corresponding
        to the mass continuity equation.

    .. attribute:: energy

        :class:`~meshmode.dof_array.DOFArray` for scalars or object array of
        :class:`~meshmode.dof_array.DOFArray` for vector quantities corresponding
        to the energy conservation equation.

    .. attribute:: momentum

        Object array (:class:`~numpy.ndarray`) with shape ``(ndim,)``
        of :class:`~meshmode.dof_array.DOFArray` , or an object array with shape
        ``(ndim, ndim)`` respectively for scalar or vector quantities corresponding
        to the ndim equations of momentum conservation.

    .. attribute:: species_mass

        Object array (:class:`~numpy.ndarray`) with shape ``(nspecies,)``
        of :class:`~meshmode.dof_array.DOFArray`, or an object array with shape
        ``(nspecies, ndim)`` respectively for scalar or vector quantities
        corresponding to the `nspecies` species mass conservation equations.

    :example::

        Use `ConservedVars` to access the fluid conserved variables (CV).

        The vector of fluid CV is commonly denoted as $\mathbf{Q}$, and for a
        fluid mixture with `nspecies` species and in `ndim` spatial dimensions takes
        the form:

        .. math::

            \mathbf{Q} &=
            \begin{bmatrix}\rho\\\rho{E}\\\rho{v}_{i}\\\rho{Y}_{\alpha}\end{bmatrix},

        with the `ndim`-vector components of fluid velocity ($v_i$), and the
        `nspecies`-vector of species mass fractions ($Y_\alpha$). In total, the
        fluid system has $N_\mbox{eq}$ = (`ndim + 2 + nspecies`) equations.

        Internally to `MIRGE-Com`, $\mathbf{Q}$ is stored as an object array
        (:class:`numpy.ndarray`) of :class:`~meshmode.dof_array.DOFArray`, one for
        each component of the fluid $\mathbf{Q}$, i.e. a flat object array of
        $N_\mbox{eq}$ :class:`~meshmode.dof_array.DOFArray`.

        To use this dataclass for a fluid CV-specific view on the content of
        $\mathbf{Q}$, one can call :func:`split_conserved` to get a `ConservedVars`
        dataclass object that resolves the fluid CV associated with each conservation
        equation::

            fluid_cv = split_conserved(ndim, Q),

        after which::

            fluid_mass_density = fluid_cv.mass  # a DOFArray with fluid density
            fluid_momentum_density = fluid_cv.momentum  # ndim-vector obj array
            fluid_species_mass_density = fluid_cv.species_mass  # nspecies-vector

        Examples of using `ConservedVars` as in this example can be found in:

        - :mod:`~mirgecom.boundary`
        - :mod:`~mirgecom.euler`
        - :mod:`~mirgecom.initializers`
        - :mod:`~mirgecom.simutil`

    :example::

        Use `join_conserved` to create an agglomerated $\mathbf{Q}$ array from the
        fluid conserved quantities (CV).

        See the first example for the definition of CV, $\mathbf{Q}$, `ndim`,
        `nspecies`, and $N_\mbox{eq}$.

        Often, a user starts with the fluid conserved quantities like mass and
        energy densities, and it is desired to glom those quantities together into
        a *MIRGE*-compatible $\mathbf{Q}$ data structure.

        For example, a solution initialization routine may set the fluid
        quantities::

            rho = ... # rho is a DOFArray with fluid density
            v = ... # v is an ndim-vector of DOFArray with components of velocity
            e = ... # e is a DOFArray with fluid energy

        An agglomerated array of fluid independent variables can then be
        created with::

            q = join_conserved(ndim, mass=rho, energy=rho*e, momentum=rho*v)

        after which *q* will be an obj array of $N_\mbox{eq}$ DOFArrays containing
        the fluid conserved state data.

        Examples of this sort of use for `join_conserved` can be found in:

        - :mod:`~mirgecom.initializers`

    :example::

        Use `ConservedVars` to access a vector quantity for each fluid equation.

        See the first example for the definition of CV, $\mathbf{Q}$, `ndim`,
        `nspecies`, and $N_\mbox{eq}$.

        Suppose the user wants to access the gradient of the fluid state,
        $\nabla\mathbf{Q}$, in a fluid-specific way. For a fluid $\mathbf{Q}$,
        such an object would be:

        .. math::

            \nabla\mathbf{Q} &=
            \begin{bmatrix}(\nabla\rho)_j\\(\nabla\rho{E})_j\\(\nabla\rho{v}_{i})_j
            \\(\nabla\rho{Y}_{\alpha})_j\end{bmatrix},

        where $1 \le j \le \mbox{ndim}$, such that the first component of
        $\mathbf{Q}$ is an `ndim`-vector corresponding to the gradient of the fluid
        density, i.e. object array of `ndim` `DOFArray`. Similarly for the energy
        term. The momentum part of $\nabla\mathbf{Q}$ is a 2D array with shape
        ``(ndim, ndim)`` with each row corresponding to the gradient of a component
        of the `ndim`-vector of fluid momentum.  The species portion of
        $\nabla\mathbf{Q}$ is a 2D array with shape ``(nspecies, ndim)`` with each
        row being the gradient of a component of the `nspecies`-vector corresponding
        to the species mass.

        Presuming that `grad_q` is the agglomerated *MIRGE* data structure with the
        gradient data, this dataclass can be used to get a fluid CV-specific view on
        the content of $\nabla\mathbf{Q}$. One can call :func:`split_conserved` to
        get a `ConservedVars` dataclass object that resolves the vector quantity
        associated with each conservation equation::

            grad_cv = split_conserved(ndim, grad_q),

        after which::

            grad_mass = grad_cv.mass  # an `ndim`-vector grad(fluid density)
            grad_momentum = grad_cv.momentum  # 2D array shape=(ndim, ndim)
            grad_spec = grad_cv.species_mass  # 2D (nspecies, ndim)

        Examples of this type of use for `ConservedVars` can be found in:

        - :func:`~mirgecom.inviscid.inviscid_flux`

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
    """Get quantities corresponding to fluid conservation equations.

    Return a :class:`ConservedVars` with quantities corresponding to the
    canonical conserved quantities, mass, energy, momentum, and any species'
    masses, from an agglomerated object array, *q*. For single component gases,
    i.e. for those state vectors *q* that do not contain multi-species mixtures, the
    returned dataclass :attr:`ConservedVars.species_mass` will be set to an empty
    array.
    """
    nspec = get_num_species(dim, q)
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim],
                         species_mass=q[2+dim:2+dim+nspec])


def join_conserved(dim, mass, energy, momentum,
        # empty: immutable
        species_mass=None):
    """Create agglomerated array from quantities for each conservation eqn."""
    if species_mass is None:
        species_mass = np.empty((0,), dtype=object)

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
