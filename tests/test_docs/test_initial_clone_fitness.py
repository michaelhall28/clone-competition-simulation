"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters
from clone_competition_simulation import FitnessParameters
import numpy as np


def test_initial_clone_fitness():
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.ones(16))
    )
    s = p.get_simulator()
    s.run_sim()
    assert s.view_clone_info()['fitness'].unique() == 1


def test_initial_clone_fitness1():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.ones(16)),
        fitness=FitnessParameters(initial_fitness_array=np.random.uniform(0.5, 1.5, size=16))
    )
    s = p.get_simulator()
    s.run_sim()
    assert s.view_clone_info()['fitness'].max() <= 1.5
    assert s.view_clone_info()['fitness'].min() >= 0.5
    assert len(s.view_clone_info()['fitness'].unique()) > 1


def test_initial_clone_fitness2():
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_grid=np.arange(16).reshape(4, 4), cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(initial_fitness_array=np.random.uniform(0.5, 1.5, size=16))
    )
    s = p.get_simulator()
    s.run_sim()
    assert s.view_clone_info()['fitness'].max() <= 1.5
    assert s.view_clone_info()['fitness'].min() >= 0.5
    assert len(s.view_clone_info()['fitness'].unique()) > 1
