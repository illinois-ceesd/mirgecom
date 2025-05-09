"""Test built-in logging functionality."""

__copyright__ = """
Copyright (C) 2025 University of Illinois Board of Trustees
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

import pytest

from logpyle import (
    LogManager,
    add_general_quantities,
)


@pytest.fixture
def basic_logmgr():
    import os

    # setup
    filename = "THIS_LOG_SHOULD_BE_DELETED.sqlite"
    logmgr = LogManager(filename, "wo")

    # give obj to test
    yield logmgr

    # clean up object
    logmgr.close()
    os.remove(filename)


def test_logmgr_verify_steptime(basic_logmgr) -> None:
    from mirgecom.logging_quantities import logmgr_verify_steptime
    from time import sleep

    add_general_quantities(basic_logmgr)

    basic_logmgr.tick_before()
    sleep(0.1)
    basic_logmgr.tick_after()

    sleep(0.1)  # Do "work" in between ticks

    basic_logmgr.tick_before()
    sleep(0.1)
    basic_logmgr.tick_after()

    with pytest.warns(UserWarning):
        logmgr_verify_steptime(basic_logmgr)

    import warnings
    with warnings.catch_warnings():
        # make sure this does not raise a warning
        warnings.simplefilter("error")
        logmgr_verify_steptime(basic_logmgr, 3.0)
