"""Utilities and functions for symbolic code expressions.

.. autofunction:: diff
.. autofunction:: div
.. autofunction:: grad

.. autoclass:: EvaluationMapper
.. autofunction:: evaluate
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

import numpy as np  # noqa
import numpy.linalg as la # noqa
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
from pymbolic.mapper.evaluator import EvaluationMapper as BaseEvaluationMapper
import mirgecom.math as mm


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


class EvaluationMapper(BaseEvaluationMapper):
    """Evaluates symbolic expressions given a mapping from variables to values.

    Inherits from :class:`pymbolic.mapper.evaluator.EvaluationMapper`.
    """

    def __init__(self, context=None, zeros_factory=None):
        super().__init__(context)
        self.zeros_factory = zeros_factory

    def map_constant(self, expr):
        value = super().map_constant(expr)
        if self.zeros_factory is None:
            return value
        else:
            return value + self.zeros_factory()

    def map_call(self, expr):
        """Map a symbolic code expression to actual function call."""
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        par, = expr.parameters
        return getattr(mm, expr.function.name)(self.rec(par))


def evaluate(expr, eval_mapper=None):
    """Evaluate a symbolic expression using a specified mapper."""

    if eval_mapper is None:
        eval_mapper = EvaluationMapper()

    from arraycontext.container import serialize_container, NotAnArrayContainerError
    from arraycontext.container.traversal import map_array_container
    from functools import partial
    try:
        serialize_container(expr)
    except NotAnArrayContainerError:
        pass
    else:
        return map_array_container(
            partial(evaluate, eval_mapper=eval_mapper), expr)

    return eval_mapper(expr)
