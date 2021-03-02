import abc


class TimestepperBase(object, metaclass=abc.ABCMeta):

    def __init__(self):
        """An amazingly descriptive docstring."""

        super(TimestepperBase, self).__init__()


    @abc.abstractmethod
    def step(self, state, t, dt, rhs):
        """An amazingly descriptive docstring."""
        pass
