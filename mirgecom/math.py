"""Utilities and functions for math.

.. autodata:: math_mapper
"""

__copyright__ = """Copyright (C) 2021 University of Illinois Board of Trustees"""

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
from pymbolic.primitives import Expression
from arraycontext import get_container_context_recursively


class _SymbolicMathContainer:
    def __getattr__(self, name):
        return pmbl.var(name)


class MathMapper:
    """Dispatcher for math operations on symbolic/numeric input."""

    @staticmethod
    def _math_container_for(*args):
        if any(isinstance(arg, Expression) for arg in args):
            return _SymbolicMathContainer()
        else:
            actx = get_container_context_recursively(make_obj_array(args))
            if actx is not None:
                return actx.np
            else:
                return np

    def __getattr__(self, name):
        """Retrieve the function corresponding to *name*."""
        return lambda *args: getattr(self._math_container_for(*args), name)(*args)


math_mapper = MathMapper()
