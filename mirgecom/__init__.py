__copyright__ = "Copyright (C) 2020 CEESD Developers"

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

import os
import getpass
import sys
from warnings import warn

try:
    from mpi4py import MPI

    # This code avoids slow startups due to file system locking when running with
    # large numbers of ranks. See https://github.com/illinois-ceesd/planning/issues/27
    # for details.

    size = MPI.COMM_WORLD.Get_size()

    if size >= 128:
        for mod in ["pyopencl", "pytools", "loopy"]:
            if mod in sys.modules:
                warn("{} already loaded, not adjusting XDG_CACHE_HOME.".format(mod))

        if "XDG_CACHE_HOME" in os.environ:
            warn("XDG_CACHE_HOME already set, not adjusting it.")
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            username = getpass.getuser()

            os.environ["XDG_CACHE_HOME"] = "/tmp/{}/xdg-cache-r{}".format(username, rank)
except:
    pass
