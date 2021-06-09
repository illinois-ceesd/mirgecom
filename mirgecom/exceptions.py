"""Provide custom exceptions for use in callback routines.

.. autoexception:: SynchronizedError
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


class SimulationHealthError(Exception):
    """Exception class for an unphysical simulation."""

    def __init__(self, message):
        super().__init__(message)


class SynchronizedException(Exception):
    """Exception class wrapping an exception which has been globally synchronized.

    .. attribute:: exception

        The wrapped exception.
    """

    def __init__(self, exception):
        super().__init__(
            f"({exception.__class__.__module__}.{exception.__class__.__name__}) "
            + exception.__str__())
        self.exception = exception
