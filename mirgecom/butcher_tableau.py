"""Butcher tableau for Runge-Kutta timestepping methods."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__author__ = """
Center for Exascale-Enabled Scramjet Design
University of Illinois, Urbana, IL 61801
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

import numpy as np

# Butcher table for the Kennedy-Carpenter explicit 5-stage, 4th order,
# low-storage Runge-Kutta method
_LSRK4_A = np.array([
    0.,
    -567301805773/1357537059087,
    -2404267990393/2016746695238,
    -3550918686646/2091501179385,
    -1275806237668/842570457699])

_LSRK4_B = np.array([
    1432997174477/9575080441755,
    5161836677717/13612068292357,
    1720146321549/2090206949498,
    3134564353537/4481467310338,
    2277821191437/14882151754819])

_LSRK4_C = np.array([
    0.,
    1432997174477/9575080441755,
    2526269341429/6820363962896,
    2006345519317/3224310063776,
    2802321613138/2924317926251])

# Butcher table for the explicit 14-stage, 4th order,
# low-storage Runge-Kutta method of Niegemann, Diehl, and Busch (2012)
# with an optimized stability region
_LSRK144_A = np.array([
    0.,
    -0.7188012108672410,
    -0.7785331173421570,
    -0.0053282796654044,
    -0.8552979934029281,
    -3.9564138245774565,
    -1.5780575380587385,
    -2.0837094552574054,
    -0.7483334182761610,
    -0.7032861106563359,
    0.0013917096117681,
    -0.0932075369637460,
    -0.9514200470875948,
    -7.1151571693922548])

_LSRK144_B = np.array([
    0.0367762454319673,
    0.3136296607553959,
    0.1531848691869027,
    0.0030097086818182,
    0.3326293790646110,
    0.2440251405350864,
    0.3718879239592277,
    0.6204126221582444,
    0.1524043173028741,
    0.0760894927419266,
    0.0077604214040978,
    0.0024647284755382,
    0.0780348340049386,
    5.5059777270269628])

_LSRK144_C = np.array([
    0.,
    0.0367762454319673,
    0.1249685262725025,
    0.2446177702277698,
    0.2476149531070420,
    0.2969311120382472,
    0.3978149645802642,
    0.5270854589440328,
    0.6981269994175695,
    0.8190890835352128,
    0.8527059887098624,
    0.8604711817462826,
    0.8627060376969976,
    0.8734213127600976])
