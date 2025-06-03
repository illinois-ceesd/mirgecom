""":mod:`mirgecom.float_comparisons` provides comparisons for float-valued arrays.

Comparison Functions
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: componentwise_norm
.. autofunction:: componentwise_err
.. autofunction:: within_tol
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
from mirgecom.fluid import ConservedVars
from arraycontext import map_array_container, get_container_context_recursively
from meshmode.dof_array import DOFArray
import numpy as np


def componentwise_norm(discr, a, order=np.inf, actx=None):
    """Calculate a component-wise norm."""
    if isinstance(a, ConservedVars):
        return componentwise_norm(
            discr, a.join(), order=order, actx=a.array_context
        )
    if actx is None:
        actx = get_container_context_recursively(a)
    return map_array_container(lambda b: discr.norm(DOFArray(actx, (b,)), order), a)


def componentwise_err(discr, lhs, rhs, relative=True, order=np.inf):
    """Calculate the component-wise error."""
    lhs_norm = componentwise_norm(discr, lhs, order=order)
    rhs_norm = componentwise_norm(discr, rhs, order=order)

    err_norm = componentwise_norm(discr, rhs - lhs, order=order)
    return err_norm if not relative else err_norm / np.maximum(lhs_norm, rhs_norm)


def within_tol(discr, lhs, rhs, tol=1e-6, relative=True,
                correct_for_eps_differences_from_zero=True, order=np.inf):
    """Check if the component-wise error is within a tolerance."""
    lhs_norm = componentwise_norm(discr, lhs, order=order)
    rhs_norm = componentwise_norm(discr, rhs, order=order)

    err_norm = componentwise_norm(discr, rhs - lhs, order=order)
    if relative:
        try:
            actx = get_container_context_recursively(err_norm)
            err_norm = err_norm / actx.np.maximum(lhs_norm, rhs_norm)
        except AttributeError:
            err_norm = err_norm / np.maximum(lhs_norm, rhs_norm)
        if correct_for_eps_differences_from_zero:
            return np.all(
                np.logical_or(np.logical_and(
                    np.minimum(lhs_norm, rhs_norm) == 0, lhs_norm - rhs_norm < 1e-30
                    ), err_norm <= tol)
                )

    return np.all(err_norm <= tol)
