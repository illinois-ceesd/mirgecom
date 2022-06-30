"""Time-integration module for Mirgecom."""

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

from .explicit_rk import rk4_step                          # noqa: F401
from .lsrk import euler_step, lsrk54_step, lsrk144_step    # noqa: F401

__doc__ = """
.. automodule:: mirgecom.integrators.explicit_rk
.. automodule:: mirgecom.integrators.lsrk
"""


def lsrk4_step(state, t, dt, rhs):
    """Call lsrk54_step with backwards-compatible interface."""
    from warnings import warn
    warn("Do not call lsrk4; it is now callled lsrk54_step. This function will "
         "disappear August 1, 2021", DeprecationWarning, stacklevel=2)
    return lsrk54_step(state, t, dt, rhs)
