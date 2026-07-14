import os

import pytest
import numpy as np
from copy import deepcopy
from pydantic import ValidationError

from src.clone_competition_simulation.parameters import (
    Parameters, 
    Algorithm, 
    FitnessParameters, 
    TimeParameters, 
    PopulationParameters, 
    DifferentiatedCellsParameters
)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_validation_failure1():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters()

    assert "algorithm\n  Input should be 'WF', 'WF2D', 'Moran', 'Moran2D' or 'Branching'" in str(exc_info)
    assert 'tag' not in str(exc_info)


def test_validation_no_config():
    p = Parameters(algorithm=Algorithm.BRANCHING, population={'initial_cells': 100},
                   times={"division_rate": 1, "max_time": 10}, 
                   show_progress=False)
    assert p.algorithm == Algorithm.BRANCHING
    assert p.times.division_rate == 1
    assert p.times.max_time == 10
    assert p.show_progress is False


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


def test_factory_reset():
    p = Parameters(algorithm=Algorithm.WF, 
                   population=PopulationParameters(
                       initial_size_array=[100, 200, 300]),
                   times=TimeParameters(division_rate=1, max_time=10)
                   )
    
    # Check that the population array is started afresh
    np.random.seed(0)
    sim1 = p.get_simulator()
    sim1.run_sim()
    assert sim1.population_array.size > 3

    sim2 = p.get_simulator()
    assert sim2.population_array.size == 3


@pytest.mark.parametrize("alg", [
    Algorithm.MORAN2D,Algorithm.WF2D
])
def test_factory_reset2(alg):
    # Check the initial grid isn't overwritten
    initial_grid = np.arange(16).reshape((4,4))
    p = Parameters(algorithm=alg, 
                   population=PopulationParameters(
                       initial_grid=deepcopy(initial_grid), 
                       cell_in_own_neighbourhood=True),
                   times=TimeParameters(division_rate=1, max_time=10)
                   )
    np.random.seed(0)
    sim1 = p.get_simulator()
    sim1.run_sim()

    sim2 = p.get_simulator()
    np.testing.assert_array_equal(sim2.parameters.population.initial_grid, 
                                  initial_grid)
    

def test_factory_reset3():
    # Check the initial grid isn't overwritten
    initial_grid = np.arange(16).reshape((4,4))
    p = Parameters(algorithm=Algorithm.MORAN2D, 
                   population=PopulationParameters(
                       initial_grid=deepcopy(initial_grid), 
                       cell_in_own_neighbourhood=True),
                   times=TimeParameters(division_rate=1, max_time=10), 
                   differentiated_cells=DifferentiatedCellsParameters(
                       r=0.1, gamma=0.7, 
                       stratification_sim_proportion=0.7
                   )
                   )
    np.random.seed(0)
    sim1 = p.get_simulator()
    sim1.run_sim()

    sim2 = p.get_simulator()
    np.testing.assert_array_equal(sim2.parameters.population.initial_grid, 
                                  initial_grid)
