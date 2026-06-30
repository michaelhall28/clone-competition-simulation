"""
Check that the code from the documentation runs.  

Not checking simulation results (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from clone_competition_simulation import (
    Parameters, 
    PopulationParameters, 
    TimeParameters
)
from clone_competition_simulation import FitnessParameters


def test_algorithm_example1():
    # Branching simulation starting from 1000 single-cell clones, all with fitness 1. 
    p = Parameters(
        algorithm="Branching", 
        population=PopulationParameters(initial_size_array=np.ones(1000)), 
        times=TimeParameters(max_time=10, division_rate=1)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)

    s.plot_overall_population()


def test_algorithm_example2():
    p = Parameters(
        algorithm="Branching", 
        population=PopulationParameters(initial_size_array=np.ones(100)),
        times=TimeParameters(max_time=10, division_rate=1),
        fitness=FitnessParameters(initial_fitness_array=np.full(100, 1.1)), 
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example3():
    # A fitness of 2 leads to rapid growth
    np.random.seed(0)
    p = Parameters(
        algorithm="Branching",
        population=PopulationParameters(initial_cells=100),
        times=TimeParameters(max_time=3, division_rate=1),
        fitness=FitnessParameters(initial_fitness_array=[2]),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example4():
    # Increasing the fitness further has no additional effect 
    np.random.seed(0)
    p = Parameters(
        algorithm="Branching",
        population=PopulationParameters(initial_cells=100),
        times=TimeParameters(max_time=3, division_rate=1),
        fitness=FitnessParameters(initial_fitness_array=[100]),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)

def test_algorithm_example5():
    # Increasing the fitness further has no additional effect 
    np.random.seed(0)
    p = Parameters(
        algorithm="Branching",
        population=PopulationParameters(initial_cells=100),
        times=TimeParameters(max_time=3, division_rate=1),
        fitness=FitnessParameters(initial_fitness_array=[100]),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example6():
    p = Parameters(
        algorithm="Branching",
        population=PopulationParameters(
            initial_size_array=np.ones(100), 
            population_limit=30000  # Set the population limit here 
        ),
        times=TimeParameters(max_time=3, division_rate=1),
        fitness=FitnessParameters(initial_fitness_array=np.full(100, 2))
    )
    s = p.get_simulator()
    from clone_competition_simulation.simulation_algorithms.branching_process import OverPopulationError

    with pytest.raises(OverPopulationError) as exc_info:
        s.run_sim()

    assert 'Ending early as population limit exceeded' in str(exc_info)


def test_algorithm_example7():
    p = Parameters(
        algorithm="Moran",
        population=PopulationParameters(initial_size_array=np.ones(1000)),
        times=TimeParameters(max_time=10, division_rate=1),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example8():
    p = Parameters(
        algorithm="WF", 
        population=PopulationParameters(initial_size_array=np.ones(1000)),
        times=TimeParameters(max_time=10, division_rate=1)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example9():
    p = Parameters(
        algorithm='Moran2D', 
        population=PopulationParameters(initial_cells=36, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=1, division_rate=1)
    )
    s = p.get_simulator()


    # Will colour a cell and its neighbourhood to show the hexagonal neighbours. 
    from clone_competition_simulation import get_neighbour_coords_2D
    row, col = 2, 3
    # Get a grid from the sim
    grid = s.grid_results[0]
    grid[row, col] = 1
    neighbours = get_neighbour_coords_2D(s, row, col)
    grid[neighbours[:, 0], neighbours[:, 1]] = 2
    s.plot_grid(grid=grid)

def test_algorithm_example10():
    p = Parameters(
        algorithm="Moran2D", 
        population=PopulationParameters(initial_grid=np.arange(30**2).reshape(30, 30), 
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=10, division_rate=1)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example11():
    p = Parameters(
        algorithm="WF2D", 
        population=PopulationParameters(initial_grid=np.arange(30**2).reshape(30, 30), 
                                        cell_in_own_neighbourhood=True),
        times=TimeParameters(max_time=10, division_rate=1)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), allow_y_extension=True)


def test_algorithm_example12():
    p = Parameters(
        algorithm="WF2D", 
        population=PopulationParameters(
            initial_grid=np.repeat(np.arange(3), 20).reshape(10, 6), 
            cell_in_own_neighbourhood=True),
        times=TimeParameters(max_time=10, division_rate=2, samples=21)
    )
    s = p.get_simulator()
    s.run_sim()
    s.plot_grid() # Plot the final grid of the simulation

    s.plot_grid(t=0)
    s.plot_grid(t=0, index_given=True) 
    s.plot_grid(grid=s.grid_results[0])

    s.plot_grid(t=5)
    s.plot_grid(t=10, index_given=True) 
    s.plot_grid(grid=s.grid_results[10])

    s.plot_grid(figsize=(3, 3))
    s.plot_grid(figxsize=3)
    
