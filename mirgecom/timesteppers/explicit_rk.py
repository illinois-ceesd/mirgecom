from mirgecom.timesteppers.base import TimestepperBase


__all__ = ("RK4Classical", )


class RK4Classical(TimestepperBase):
    """An amazingly descriptive docstring."""

    def step(self, state, t, dt, rhs):
        """Take one step using 4th order Runge-Kutta."""

        k1 = rhs(t, state)
        k2 = rhs(t+dt/2, state + dt/2*k1)
        k3 = rhs(t+dt/2, state + dt/2*k2)
        k4 = rhs(t+dt, state + dt*k3)
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)
