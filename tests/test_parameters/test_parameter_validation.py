import os

import pytest
import numpy as np
from pydantic import ValidationError

from src.clone_competition_simulation.parameters import (
    Parameters, 
    Algorithm, 
    FitnessParameters, 
    TimeParameters, 
    PopulationParameters
)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_validation_failure1():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters()

    assert "algorithm\n  Input should be 'WF', 'WF2D', 'Moran', 'Moran2D' or 'Branching'" in str(exc_info)
    assert 'tag' not in str(exc_info)


def test_validation_no_config():
    p = Parameters(algorithm=Algorithm.BRANCHING, population={'initial_cells': 100},
                   times={"division_rate": 1, "max_time": 10})
    assert p.algorithm == Algorithm.BRANCHING
    assert p.times.division_rate == 1
    assert p.times.max_time == 10


def test_validation_with_typo_1():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(algorithm=Algorithm.BRANCHING, 
                   population={'initial_cells': 100},
                   times={"division_rate": 1, "max_time": 10}, 
                   fitness={"mutation_rate": 0.1}
                   )
        
    print(str(exc_info))
        
    assert "Extra inputs are not permitted" in str(exc_info)


def test_validation_with_typo_2():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(algorithm=Algorithm.BRANCHING, 
                   population=PopulationParameters(initial_cells=100),
                   times=TimeParameters(division_rate=1, max_time=10), 
                   fitness=FitnessParameters(mutation_rate=0.1)
                   )
        
    assert "Extra inputs are not permitted" in str(exc_info)

