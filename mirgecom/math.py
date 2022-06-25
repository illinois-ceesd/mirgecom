"""Utilities and functions for math.

Functions not explicitly defined here fall back to the appropriate numpy-like math
function for the provided inputs (if one exists), through :func:`__getattr__`.
The inputs can be :mod:`numpy` arrays, :mod:`arraycontext` arrays, or :mod:`pymbolic`
expressions.

:example::

    With :mod:`numpy` input data::

        import mirgecom.math as mm

        x_np = np.array([0, np.pi/2, np.pi])
        s_np = mm.sin(x_np)  # Calls np.sin

    or :mod:`arraycontext` input data::

        x_device = actx.from_numpy(x_np)
        s_device = mm.sin(x_device)  # Calls actx.np.sin

    or :mod:`pymbolic` input expression::

        x_sym = pmbl.var("x")
        s_sym = mm.sin(x_sym)  # Creates an expression pmbl.var("sin")(x_sym)

.. autofunction:: __getattr__
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


def __getattr__(name):
    """
    Return a function that calls an appropriate math function based on its inputs.

    Returns a function that inspects the types of its input arguments and dispatches
    to the appropriate math function. If any of the arguments are symbolic, the
    function returns a :class:`pymbolic.primitives.Expression` representing the call
    to *name*. If not, it next checks whether any of the arguments have array
    contexts. If so, it calls *name* from the array context's :mod:`numpy` workalike.
    And if none of the arguments have array contexts, it calls :mod:`numpy`'s version
    of *name*.
    """
    # Avoid special/private names, and restrict to functions that exist in numpy
    if name.startswith("_") or name.endswith("_") or not hasattr(np, name):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    def dispatcher_func(*args):
        if any(isinstance(arg, Expression) for arg in args):
            math_func = pmbl.var(name)
        else:
            actx = get_container_context_recursively(make_obj_array(args))
            if actx is not None:
                np_like = actx.np
            else:
                np_like = np
            math_func = getattr(np_like, name)
        return math_func(*args)

    return dispatcher_func
