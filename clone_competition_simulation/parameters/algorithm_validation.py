from enum import Enum, auto

class AlgorithmClass(str, Enum):
    WF = auto()
    MORAN = auto()
    BRANCHING = auto()


class Algorithm(Enum):
    WF = "WF", False, AlgorithmClass.WF
    WF2D = "WF2D", True, AlgorithmClass.WF
    MORAN = "Moran", False, AlgorithmClass.MORAN
    MORAN2D = "Moran2D", True, AlgorithmClass.MORAN
    BRANCHING = "Branching", False, AlgorithmClass.BRANCHING

    def __new__(cls, value, two_dimensional: bool, algorithm_class: AlgorithmClass):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.two_dimensional = two_dimensional
        obj.algorithm_class = algorithm_class
        return obj


if __name__ == '__main__':
    print(Algorithm.MORAN.value)