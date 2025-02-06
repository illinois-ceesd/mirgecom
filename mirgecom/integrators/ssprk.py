"""Timestepping routines for strong-stability preserving Runge-Kutta methods.

.. autofunction:: ssprk43_step
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


def ssprk43_step(state, t, dt, rhs, context):
    """Take one step using an explicit 4-stage, 3rd-order, SSPRK method.
    Context is a dictionary for any variables which can affect state,
    but maybe not the initial state.
    """

    def rhs_update(t, y, context):
        return y + dt*rhs(t, y, context)

    y1 = 1/2*state + 1/2*rhs_update(t, state, context)
    y2 = 1/2*y1 + 1/2*rhs_update(t + dt/2, y1, context)
    y3 = 2/3*state + 1/6*y2 + 1/6*rhs_update(t + dt, y2, context)
    y4 = 1/2*y3 + 1/2*rhs_update(t + dt/2, y3, context)

    return y4
