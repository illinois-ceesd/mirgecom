"""Provide custom exceptions for use in callback routines.

.. autoexception:: MirgecomException
.. autoexception:: ExactSolutionMismatch
.. autoexception:: SimulationHealthError
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


class MirgecomException(Exception):
    """Exception base class for mirgecom exceptions.

    .. attribute:: step

        A :class:`int` denoting the simulation step when the exception
        was raised.

    .. attribute:: t

        A :class:`float` denoting the simulation time when the
        exception was raised.

    .. attribute:: state

        The simulation state when the exception was raised.

    .. attribute:: message

        A :class:`str` describing the message for the exception.
    """

    def __init__(self, step, t, state, message):
        """Record the simulation state on creation."""
        self.step = step
        self.t = t
        self.state = state
        self.message = message
        super().__init__(self.message)


class ExactSolutionMismatch(MirgecomException):
    """Exception class for solution mismatch."""

    def __init__(self, step, t, state):
        super().__init__(
            step, t, state,
            message="Solution doesn't agree with analytic result."
        )


class SimulationHealthError(MirgecomException):
    """Exception class for an unphysical simulation."""

    def __init__(self, step, t, state, message):
        super().__init__(step, t, state, message)
