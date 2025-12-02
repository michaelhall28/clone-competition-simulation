from enum import Enum, auto
from pydantic import ValidationError, BaseModel


class Algorithm(str, Enum):
    WF = auto()
    WF2D = auto()
    MORAN = auto()
    MORAN2D = auto()
    BRANCHING = auto()


class AlgorithmClass(str, Enum):
    WF = auto()
    MORAN = auto()
    BRANCHING = auto()


class ValidationCategories(BaseModel):
    two_dimensional: bool
    algorithm_class: AlgorithmClass


ALGORITHMS = {
    Algorithm.WF: ValidationCategories(two_dimensional=False, algorithm_class=AlgorithmClass.WF),
    Algorithm.WF2D: ValidationCategories(two_dimensional=True, algorithm_class=AlgorithmClass.WF),
    Algorithm.MORAN: ValidationCategories(two_dimensional=False, algorithm_class=AlgorithmClass.MORAN),
    Algorithm.MORAN2D: ValidationCategories(two_dimensional=True, algorithm_class=AlgorithmClass.MORAN),
    Algorithm.BRANCHING: ValidationCategories(two_dimensional=False, algorithm_class=AlgorithmClass.BRANCHING)
}
