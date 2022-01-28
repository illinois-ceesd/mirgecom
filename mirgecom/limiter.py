""":mod:`mirgecom.limiter` is for limiters and limiter-related constructs.
"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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

from arraycontext import map_array_container

from meshmode.dof_array import DOFArray

from grudge.discretization import DiscretizationCollection

from mirgecom.fluid import ConservedVars

import grudge.op as op


def limiter_liu_osher(dcoll: DiscretizationCollection, state, quadrature_tag=None):
    """Implements the positivity-preserving limiter of Liu and Osher (1996).

    The limiter is summarized in the review paper [Zhang_2011]_, Section 2.3,
    equation 2.9, which uses a linear scaling factor.
    
    .. note:
        This limiter is applied only to mass fields
        (e.g. mass or species masses for multi-component flows)

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        Grudge discretization with boundaries object
    state: :class:`mirgecom.fluid.ConservedVars`
        An array container containing the conserved variables.

    Returns
    -------
    result: :class:`mirgecom.fluid.ConservedVars`
        An array container containing the filtered field(s).
    """
    from grudge.dof_desc import as_dofdesc
    from grudge.geometry import area_element

    actx = state.array_context

    if not actx.supports_nonscalar_broadcasting:
        raise RuntimeError("Can't use the limiter in eager-mode... yeah, I know.")

    dd = as_dofdesc("vol")
    dd_quad = dd.with_discr_tag(quadrature_tag)

    def compute_limited_field(field):
        if not isinstance(field, DOFArray):
            # vecs is not a DOFArray -> treat as array container
            return map_array_container(compute_limited_field, field)

        # Compute nodal and elementwise max/mins of the field
        # on the quadrature grid
        field_quad = op.project(dcoll, dd, dd_quad, field)
        mmax = op.nodal_max(dcoll, dd_quad, field_quad)
        mmin = op.nodal_min(dcoll, dd_quad, field_quad)
        mmax_i = op.elementwise_max(dcoll, dd_quad, field_quad)
        mmin_i = op.elementwise_min(dcoll, dd_quad, field_quad)

        # Compute cell averages of the state
        inv_area_elements = 1./area_element(
            actx, dcoll, dd=dd_quad,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
        field_cell_avgs = \
            inv_area_elements * op.elementwise_integral(dcoll, dd_quad, field_quad)

        # Compute minmod factor (Eq. 2.9)
        theta = actx.np.minimum(
            1.,
            actx.np.minimum(
                abs((mmax - field_cell_avgs)/(mmax_i - field_cell_avgs)),
                abs((mmin - field_cell_avgs)/(mmin_i - field_cell_avgs))
            )
        )

        # import ipdb; ipdb.set_trace()
        return theta*(field - field_cell_avgs) + field_cell_avgs

    return ConservedVars(
        mass=compute_limited_field(state.mass),
        energy=state.energy,
        momentum=state.momentum,
        species_mass=compute_limited_field(state.species_mass)
    )
