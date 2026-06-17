"""
Classes for diminishing returns or other transformations of the raw fitness
"""
import math
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray


T = TypeVar('T', bound=float | NDArray[float])

@runtime_checkable  # So it can be used in Pydantic fields
class FitnessTransform(Protocol):
    """Protocol for fitness transformation objects.

    Objects implementing this protocol can transform raw clone fitness values and
    calculate inverse transformations.
    """

    def fitness(self, x: T) -> T:
        ...

    def inverse(self, y: T) -> T:
        ...


class UnboundedFitness:
    """Identity fitness transform.

    This transform returns the input fitness unchanged.
    """

    def __str__(self) -> str:
        return 'UnboundedFitness'

    def fitness(self, x: T) -> T:
        return x

    def inverse(self, y: T) -> T:
        return y


class BoundedLogisticFitness:
    """Bounded logistic fitness transform.

    Applies a logistic transformation to raw fitness values so that
    additional beneficial mutations have diminishing returns and the output
    fitness is bounded by ``a``.
    """
    def __init__(self, a: float, b: float=math.exp(1)):
        """Initialize the logistic fitness transform.

        Parameters
        ----------
        a : float
            Maximum output fitness of a clone. Must be greater than 1.
        b : float, default math.exp(1)
            Controls the slope of the transformation. Must be greater than 1.
        """
        assert (a > 1)
        assert (b > 1)
        self.a = a
        self.b = b
        self.c = (a - 1) * self.b

    def __str__(self) -> str:
        return 'Bounded Logistic: a {0}, b {1}, c {2}'.format(self.a, self.b, self.c)

    def fitness(self, x: T) -> T:
        return self.a / (1 + self.c * (self.b ** (-x)))

    def inverse(self, y: T) -> T:
        return np.emath.logn(self.b, (self.c / (self.a / y - 1)))
