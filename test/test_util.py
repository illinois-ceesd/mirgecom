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

import logging
from mirgecom.simutil import MIRGEComParameters

logger = logging.getLogger(__name__)


def test_mirgecom_parameters():
    """Quick test of MIRGE-Com parameters container."""
    test_params = MIRGEComParameters(dim=2, order=3, casename="hello")
    my_params = test_params.parameters
    print(f"{test_params.parameters}")
    assert len(my_params) == 3
    assert my_params["dim"] == 2
    assert my_params["order"] == 3
    assert my_params["casename"] == "hello"

    test_params.update(order=4, casename="goodbye", hello="hello")
    my_params = test_params.parameters
    assert len(my_params) == 4
    assert my_params["order"] == 4
    assert my_params["dim"] == 2
    assert my_params["casename"] == "goodbye"
    assert my_params["hello"] == "hello"

    params_string = (
        f"from mirgecom.simutil import MIRGEComParameters"
        f"\nmirgecom_parameters = MIRGEComParameters("
        f"\ndim=5, newparam=\"string\")"
    )

    file1 = open("test_params_fjsfjksd.py", "a")
    file1.write(params_string)
    file1.close()
    test_params.read("test_params_fjsfjksd.py")
    my_params = test_params.parameters
    assert len(my_params) == 5
    assert my_params["dim"] == 5
    assert my_params["newparam"] == "string"
    import os
    os.remove("test_params_fjsfjksd.py")


