from typing import Self
from enum import Enum, auto

class AlgorithmClass(str, Enum):
    """The main classes of algorithms (ignoring 2D vs non-2D variants)"""
    WF = auto()
    MORAN = auto()
    BRANCHING = auto()


class Algorithm(Enum):
    """Enumeration of supported algorithms, with metadata about whether they are 2D and their main class."""
    WF = "WF", False, AlgorithmClass.WF
    WF2D = "WF2D", True, AlgorithmClass.WF
    MORAN = "Moran", False, AlgorithmClass.MORAN
    MORAN2D = "Moran2D", True, AlgorithmClass.MORAN
    BRANCHING = "Branching", False, AlgorithmClass.BRANCHING

    def __new__(cls, value, two_dimensional: bool, algorithm_class: AlgorithmClass) -> Self:
        obj = object.__new__(cls)
        obj._value_ = value
        obj.two_dimensional = two_dimensional
        obj.algorithm_class = algorithm_class
        return obj
