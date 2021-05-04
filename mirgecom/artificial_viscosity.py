r""":mod:`mirgecom.artificial_viscosity` Artificial viscosity for hyperbolic systems.

Consider the following system of conservation laws of the form:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} = 0,

where $\mathbf{Q}$ is the state vector and $\mathbf{F}$ is the vector of
fluxes. This module applies an artificial viscosity term by augmenting
the governing equations in the following way:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} =
    \nabla\cdot{\varepsilon\nabla\mathbf{Q}},

where $\varepsilon$ is the artificial viscosity coefficient.
To evalutate the second order derivative numerically, the problem
is recast as a set of first order problems:

.. math::

    \partial_t{\mathbf{Q}} + \nabla\cdot\mathbf{F} &= \nabla\cdot\mathbf{R} \\
    \mathbf{R} &= \varepsilon\nabla\mathbf{Q}

where $\mathbf{R}$ is an auxiliary variable, and the artificial viscosity
coefficient, $\varepsilon$, is spatially dependent and calculated using the
smoothness indicator of [Persson_2012]_:

.. math::

    S_e = \frac{\langle u_{N_p} - u_{N_{p-1}}, u_{N_p} -
        u_{N_{p-1}}\rangle_e}{\langle u_{N_p}, u_{N_p} \rangle_e}

where:

- $S_e$ is the smoothness indicator
- $u_{N_p}$ is the solution in modal basis at the current polynomial order
- $u_{N_{p-1}}$ is the truncated modal represention to the polynomial order $p-1$
- The $L_2$ inner product on an element is denoted $\langle \cdot,\cdot \rangle_e$

The elementwise viscosity is then calculated:

.. math::

    \varepsilon_e =
        \begin{cases}
            0, & s_e < s_0 - \kappa \\
            \frac{1}{2}\left( 1 + \sin \frac{\pi(s_e - s_0)}{2 \kappa} \right ),
            & s_0-\kappa \le s_e \le s_0+\kappa \\
            1, & s_e > s_0+\kappa
        \end{cases}

where:

- $\varepsilon_e$ is the element viscosity
- $s_e = \log_{10}{S_e} \sim 1/p^4$ is the smoothness indicator
- $s_0$ is a reference smoothness value
- $\kappa$ controls the width of the transition between 0 to 1

Smoothness Indicator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: smoothness_indicator

AV RHS Evaluation
^^^^^^^^^^^^^^^^^

.. autofunction:: av_operator
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

from pytools import memoize_in, keyed_memoize_in
from meshmode.dof_array import DOFArray
from grudge.dof_desc import DD_VOLUME_MODAL, DD_VOLUME, DISCR_TAG_BASE
from mirgecom.diffusion import diffusion_operator


def av_operator(discr, boundaries, q, alpha, boundary_kwargs=None, **kwargs):
    r"""Compute the artificial viscosity right-hand-side.

    Computes the the right-hand-side term for artificial viscosity.

    .. math::

        \mbox{RHS}_{\mbox{av}} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

    Parameters
    ----------
    q: :class:`~meshmode.dof_array.DOFArray` or :class:`~numpy.ndarray`

        Single :class:`~meshmode.dof_array.DOFArray` for a scalar or an object array
        (:class:`~numpy.ndarray`) for a vector of
        :class:`~meshmode.dof_array.DOFArray` on which to operate.

        When used with fluid solvers, *q* is expected to be the fluid state array
        of the canonical conserved variables (mass, energy, momentum)
        for the fluid along with a vector of species masses for multi-component
        fluids.

    boundaries: float

        Dictionary of boundary functions, one for each valid boundary tag

    alpha: float

        The maximum artificial viscosity coefficient to be applied

    boundary_kwargs: :class:`dict`

        dictionary of extra arguments to pass through to the boundary conditions

    Returns
    -------
    numpy.ndarray

        The artificial viscosity operator applied to *q*.
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    # Get smoothness indicator based on first component
    indicator_field = q[0] if isinstance(q, np.ndarray) else q
    indicator = smoothness_indicator(discr, indicator_field, **kwargs)

    return diffusion_operator(discr, quad_tag=DISCR_TAG_BASE,
        alpha=alpha*indicator, boundaries=boundaries,
        boundary_kwargs=boundary_kwargs, u=q)


def artificial_viscosity(discr, t, eos, boundaries, q, alpha, **kwargs):
    """Interface :function:`av_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call artificial_viscosity; it is now called av_operator. This"
         "function will disappear in 2021", DeprecationWarning, stacklevel=2)
    return av_operator(discr=discr, boundaries=boundaries,
        boundary_kwargs={"t": t, "eos": eos}, q=q, alpha=alpha, **kwargs)


def smoothness_indicator(discr, u, kappa=1.0, s0=-6.0):
    r"""Calculate the smoothness indicator.

    Parameters
    ----------
    u: meshmode.dof_array.DOFArray
        The field that is used to calculate the smoothness indicator.

    kappa
        An optional argument that controls the width of the transition from 0 to 1.
    s0
        An optional argument that sets the smoothness level to limit
        on. Values in the range $(-\infty,0]$ are allowed, where $-\infty$ results in
        all cells being tagged and 0 results in none.

    Returns
    -------
    meshmode.dof_array.DOFArray
        The elementwise constant values between 0 and 1 which indicate the smoothness
        of a given element.
    """
    if not isinstance(u, DOFArray):
        raise ValueError("u argument must be a DOFArray.")

    actx = u.array_context

    @memoize_in(actx, (smoothness_indicator, "smooth_comp_knl"))
    def indicator_prg():
        """Compute the smoothness indicator for all elements."""
        from meshmode.array_context import make_loopy_program
        return make_loopy_program([
            "{[iel]: 0 <= iel < nelements}",
            "{[idof]: 0 <= idof < ndiscr_nodes_in}",
            "{[jdof]: 0 <= jdof < ndiscr_nodes_in}",
            "{[kdof]: 0 <= kdof < ndiscr_nodes_in}"
            ],
            """
                result[iel,idof] = sum(kdof, vec[iel, kdof]     \
                                             * vec[iel, kdof]   \
                                             * modes[kdof]) /   \
                                   sum(jdof, vec[iel, jdof]     \
                                             * vec[iel, jdof]   \
                                             + 1.0e-12 / ndiscr_nodes_in)
            """,
            name="smooth_comp",
        )

    @keyed_memoize_in(actx, (smoothness_indicator,
                             "highest_mode"),
                      lambda grp: grp.discretization_key())
    def highest_mode(grp):
        return actx.from_numpy(
            np.asarray([1 if sum(mode_id) == grp.order
                        else 0
                        for mode_id in grp.mode_ids()])
        )

    # Convert to modal solution representation
    modal_map = discr.connection_from_dds(DD_VOLUME, DD_VOLUME_MODAL)
    uhat = modal_map(u)

    # Compute smoothness indicator value
    indicator = DOFArray(
        actx,
        data=tuple(
            actx.call_loopy(
                indicator_prg(),
                vec=uhat[grp.index],
                modes=highest_mode(grp))["result"]
            for grp in discr.discr_from_dd("vol").groups
        )
    )
    indicator = actx.np.log10(indicator + 1.0e-12)

    # Compute artificial viscosity percentage based on indicator and set parameters
    yesnol = indicator > (s0 - kappa)
    yesnou = indicator > (s0 + kappa)
    sin_indicator = actx.np.where(
        yesnol,
        0.5 * (1.0 + actx.np.sin(np.pi * (indicator - s0) / (2.0 * kappa))),
        0.0 * indicator,
    )
    indicator = actx.np.where(yesnou, 1.0 + 0.0 * indicator, sin_indicator)

    return indicator
