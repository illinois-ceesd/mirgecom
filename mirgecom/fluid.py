""":mod:`mirgecom.fluid` provides common utilities for fluid simulation.

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
from mirgecom.euler import ConservedVars # noqa


def velocity_gradient(discr, cv, grad_cv):
    r"""
    Compute the gradient of fluid velocity.

    Computes the gradient of fluid velocity from:

    .. math::

        \nabla{v_i} = \frac{1}{\rho}(\nabla(\rho{v_i})-v_i\nabla{\rho}),

    where $v_i$ is ith velocity component.

    .. note::
        We use the product rule to evaluate gradients of the primitive variables
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


    .. note::
        We use the product rule to evaluate gradients of the primitive variables
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
        representing $\partial_j{v_i}$.
    """
    nspecies = len(cv.species_mass)
    y = cv.species_mass / cv.mass
    return (1/cv.mass)*make_obj_array([grad_cv.species_mass[i]
                                       - y[i]*grad_cv.mass
                                       for i in range(nspecies)])
