from mirgecom.timesteppers.base import TimestepperBase
import numpy as np


__all__ = ("LSRKEuler", "LSRK54CarpenterKennedy",
           "LSRK144NiegemannDiehlBusch")


class LowStorageRungeKutta(TimestepperBase):
    """An amazingly descriptive docstring."""

    def __init__(self, A, B, C):
        """An amazingly descriptive docstring."""

        super(LowStorageRungeKutta, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.nstages = len(B)

    def step(self, state, t, dt, rhs):
        """Take one step using a low-storage Runge Kutta method."""

        k = 0.0
        for i in range(self.nstages):
            k = self.A[i]*k + dt*rhs(t + self.C[i]*dt, state)
            state += self.B[i]*k

        return state


class LSRKEuler(LowStorageRungeKutta):
    """An amazingly descriptive docstring."""

    def __init__(self):
        """An amazingly descriptive docstring."""

        A = np.array([0.])

        B = np.array([1.])

        C = np.array([0.])

        super(LSRKEuler, self).__init__(A, B, C)
    

class LSRK54CarpenterKennedy(LowStorageRungeKutta):
    """An amazingly descriptive docstring."""

    def __init__(self):
        """An amazingly descriptive docstring."""

        A = np.array([
            0.,
            -567301805773/1357537059087,
            -2404267990393/2016746695238,
            -3550918686646/2091501179385,
            -1275806237668/842570457699])

        B = np.array([
            1432997174477/9575080441755,
            5161836677717/13612068292357,
            1720146321549/2090206949498,
            3134564353537/4481467310338,
            2277821191437/14882151754819])

        C = np.array([
            0.,
            1432997174477/9575080441755,
            2526269341429/6820363962896,
            2006345519317/3224310063776,
            2802321613138/2924317926251])

        super(LSRK54CarpenterKennedy, self).__init__(A, B, C)


class LSRK144NiegemannDiehlBusch(LowStorageRungeKutta):
    """An amazingly descriptive docstring."""

    def __init__(self):
        """An amazingly descriptive docstring."""

        A = np.array([
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

        B = np.array([
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

        C = np.array([
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

        super(LSRK144NiegemannDiehlBusch, self).__init__(A, B, C)
