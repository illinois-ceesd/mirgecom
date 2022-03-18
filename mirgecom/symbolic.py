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
from pymbolic.mapper.differentiator import (
    DifferentiationMapper as BaseDifferentiationMapper
)
from pymbolic.mapper.evaluator import EvaluationMapper as BaseEvaluationMapper
import mirgecom.math as mm


class DifferentiationMapper(BaseDifferentiationMapper):
    """
    Differentiates a symbolic expression.

    Inherits from :class:`pymbolic.mapper.differentiator.DifferentiationMapper`.
    """

    def __call__(self, expr, *args, **kwargs):
        """Differentiate *expr*."""
        from arraycontext import rec_map_array_container
        return rec_map_array_container(
            lambda f: super(DifferentiationMapper, self).__call__(
                f, *args, **kwargs),
            expr)


def diff(var):
    """Return the symbolic derivative operator with respect to *var*."""
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


def _is_expression_or_number(f):
    from pymbolic.primitives import Expression
    return isinstance(f, Expression) or np.isscalar(f)


def div(ambient_dim, func):
    """Return the symbolic divergence of *func*."""
    coords = pmbl.make_sym_vector("x", ambient_dim)

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        f=lambda f: sum(diff(coords[i])(f[i]) for i in range(ambient_dim)),
        in_shape=(ambient_dim,),
        out_shape=(),
        is_scalar=_is_expression_or_number,
        ary=func)


def grad(ambient_dim, func, nested=False):
    """Return the symbolic *dim*-dimensional gradient of *func*."""
    coords = pmbl.make_sym_vector("x", ambient_dim)

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        f=lambda f: make_obj_array([
            diff(coords[i])(f) for i in range(ambient_dim)]),
        in_shape=(),
        out_shape=(ambient_dim,),
        is_scalar=_is_expression_or_number,
        return_nested=nested,
        ary=func)


class EvaluationMapper(BaseEvaluationMapper):
    """Evaluates symbolic expressions given a mapping from variables to values.

    Inherits from :class:`pymbolic.mapper.evaluator.EvaluationMapper`.
    """

    def __init__(self, context=None, constant_zero=None):
        super().__init__(context)
        self._cache = {}
        self._constant_zero = constant_zero
        self._used_symbols = set()

    def rec(self, expr):
        try:
            return self._cache[expr]
        except KeyError:
            result = super().rec(expr)

            # This stores any-and-everything in its cache.
            # FIXME: Be more selective if this ends up eating too
            # much memory.
            self._cache[expr] = result
            return result

    def map_constant(self, expr):
        if self._constant_zero is None:
            return super().map_constant(expr)
        else:
            return super().map_constant(expr) + self._constant_zero

    def map_call(self, expr):
        """Map a symbolic code expression to actual function call."""
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        par, = expr.parameters
        return getattr(mm, expr.function.name)(self.rec(par))

    def map_variable(self, expr):
        self._used_symbols.add(expr.name)
        return super().map_variable(expr)


def evaluate(expr, mapper_type=EvaluationMapper, **kwargs):
    """Evaluate a symbolic expression using a specified mapper."""
    constant_zero = kwargs.pop("zero", None)
    unused_ok = kwargs.pop("unused_ok", False)
    mapper = mapper_type(kwargs, constant_zero=constant_zero)

    from arraycontext import rec_map_array_container
    result = rec_map_array_container(mapper, expr)

    if not unused_ok and mapper._used_symbols != set(mapper.context):
        raise ValueError("some variables were not used during evaluation: "
                f"{', '.join(set(mapper.context) - mapper._used_symbols)}")

    return result
