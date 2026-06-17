"""
Probability distributions for drawing the fitness of new mutations.
Set up so the distributions can be called like functions without argument, but can print the attributes
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable  # So it can be used in Pydantic fields
class DistributionProtocol(Protocol):
    """Protocol for distribution-like objects.

    Objects implementing this protocol can be used as mutation fitness
    distributions in the simulation.
    """

    def __call__(self) -> float:
        ...

    def get_mean(self) -> float:
        ...


class NormalDist:
    """Normal mutation fitness distribution.

    Draws from a normal distribution and retries if the sample is negative.

    Parameters
    ----------
    var : float
        Variance of the normal distribution.
    mean : float, default 1.
        Mean of the normal distribution.
    """
    def __init__(self, var: float, mean: float=1.):
        self.var = var
        self.mean = mean

    def __str__(self) -> str:
        return 'Normal distribution(mean {0}, variance {1})'.format(self.mean, self.var)

    def __call__(self) -> float:
        g = np.random.normal(self.mean, self.var)
        if g < 0:
            return self()
        return g

    def get_mean(self) -> float:
        return self.mean


class FixedValue:
    """Fixed mutation fitness distribution.

    Wraps a constant value so it behaves like a stochastic fitness distribution.

    Parameters
    ----------
    value : float
        Fixed fitness value returned for every draw.
    """
    def __init__(self, value: float):
        self.mean = value

    def __str__(self) -> str:
        return 'Fixed value {0}'.format(self.mean)

    def __call__(self) -> float:
        return self.mean

    def get_mean(self) -> float:
        return self.mean


class ExponentialDist:
    """Exponential mutation fitness distribution.

    Wraps ``numpy.random.exponential`` and applies an offset so that the
    drawn fitness value is always greater than or equal to ``offset``.

    Parameters
    ----------
    mean : float
        Target mean of the distribution after the offset is applied. Must be
        greater than ``offset``.
    offset : float, default 1
        Minimum fitness value.
    """
    def __init__(self, mean: float, offset: float=1):
        if mean <= offset:
            raise ValueError('mean must be greater than offset')
        self.mean = mean
        self.offset = offset  # Offset of 1 means the mutations will start from neutral.

    def __str__(self) -> str:
        return 'Exponential distribution(mean {0}, offset {1})'.format(self.mean, self.offset)

    def __call__(self) -> float:
        g = self.offset + np.random.exponential(self.mean - self.offset)
        return g

    def get_mean(self) -> float:
        return self.mean


class UniformDist:
    """Uniform mutation fitness distribution.

    Wraps ``numpy.random.uniform`` to draw a fitness value between a lower
    and upper bound.

    Parameters
    ----------
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.
    """
    def __init__(self, low: float, high: float):
        if low >= high:
            raise ValueError('high bound must be higher than the low bound')
        self.low = low
        self.high = high

    def __str__(self) -> str:
        return 'Uniform distribution(low {0}, high {1})'.format(self.low, self.high)

    def __call__(self) -> float:
        g = np.random.uniform(self.low, self.high)
        return g

    def get_mean(self) -> float:
        return (self.high + self.low)/2


