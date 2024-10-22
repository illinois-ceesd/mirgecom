"""Common functions for all tests."""

__copyright__ = """
Copyright (C) 2024 University of Illinois Board of Trustees
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


@pytest.fixture(autouse=True)
def mem_usage(capsys, pytestconfig):
    # Code that will run before your test goes here

    yield

    # Code that will run after your test goes here

    # {{{ Memory usage reporting

    if not pytestconfig.option.verbose:
        return

    # Copied from logpyle.MemoryHwm
    import os
    if os.uname().sysname == "Linux":
        fac = 1024
    elif os.uname().sysname == "Darwin":
        fac = 1024*1024

    from resource import RUSAGE_SELF, getrusage
    res = getrusage(RUSAGE_SELF)

    with capsys.disabled():
        print(f" HWM={res.ru_maxrss / fac}")

    # }}}
