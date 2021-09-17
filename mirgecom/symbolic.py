"""Utilities and functions for symbolic code expressions.

.. autofunction:: diff
.. autofunction:: div
.. autofunction:: grad

.. autoclass:: EvaluationMapper
"""

__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

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
import numpy.linalg as la # noqa
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
import pymbolic.mapper.evaluator as ev


def diff(var):
    """Return the symbolic derivative operator with respect to *var*."""
    from pymbolic.mapper.differentiator import DifferentiationMapper

    def func_map(arg_index, func, arg, allowed_nonsmoothness):
        if func == pmbl.var("sin"):
            return pmbl.var("cos")(*arg)
        elif func == pmbl.var("cos"):
            return -pmbl.var("sin")(*arg)
        elif func == pmbl.var("exp"):
            return pmbl.var("exp")(*arg)
        else:
            raise ValueError("Unrecognized function")

    return DifferentiationMapper(var, func_map=func_map)


def div(vector_func):
    """Return the symbolic divergence of *vector_func*."""
    dim = len(vector_func)
    coords = pmbl.make_sym_vector("x", dim)
    return sum([diff(coords[i])(vector_func[i]) for i in range(dim)])


def grad(dim, func):
    """Return the symbolic *dim*-dimensional gradient of *func*."""
    coords = pmbl.make_sym_vector("x", dim)
    return make_obj_array([diff(coords[i])(func) for i in range(dim)])


class EvaluationMapper(ev.EvaluationMapper):
    """Evaluates symbolic expressions given a mapping from variables to values.

    Inherits from :class:`pymbolic.mapper.evaluator.EvaluationMapper`.
    """

    def map_call(self, expr):
        """Map a symbolic code expression to actual function call."""
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        if expr.function.name == "sin":
            par, = expr.parameters
            return self._sin(self.rec(par))
        elif expr.function.name == "cos":
            par, = expr.parameters
            return self._cos(self.rec(par))
        elif expr.function.name == "exp":
            par, = expr.parameters
            return self._exp(self.rec(par))
        else:
            raise ValueError("Unrecognized function '%s'" % expr.function)

    def _sin(self, val):
        from numbers import Number
        if isinstance(val, Number):
            return np.sin(val)
        else:
            return val.array_context.np.sin(val)

    def _cos(self, val):
        from numbers import Number
        if isinstance(val, Number):
            return np.cos(val)
        else:
            return val.array_context.np.cos(val)

    def _exp(self, val):
        from numbers import Number
        if isinstance(val, Number):
            return np.exp(val)
        else:
            return val.array_context.np.exp(val)
