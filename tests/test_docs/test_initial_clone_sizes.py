"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters
from clone_competition_simulation import FitnessParameters, Gene, FitnessCalculator, NormalDist
import pytest
import numpy as np


def test_initial_clone_sizes():
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(samples=10, max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=1000)
    )
    s = p.get_simulator()
    s.run_sim()

    # There is only one clone in the simulation. 
    assert len(s.view_clone_info()) == 1
    assert np.unique(s.population_array.toarray()) == 1000


def test_initial_clone_sizes1():
    np.random.seed(0)
    p = Parameters(
        algorithm='Branching', 
        times=TimeParameters(samples=10, max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=1000)
    )
    s = p.get_simulator()
    s.run_sim()
    np.testing.assert_array_equal(
        s.population_array.toarray(), 
        [[1000., 1046., 1070., 1117., 1164., 1077., 1155., 
          1098., 1100., 1035., 1043.]]
    )


def test_initial_clone_sizes2():
    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )

    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(samples=10, max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=1000), 
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fitness_calculator)
    )
    s = p.get_simulator()
    s.run_sim()

    s.muller_plot(figsize=(6, 6), show_mutations_with_x=False)


def test_initial_clone_sizes3():
    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=30**2, cell_in_own_neighbourhood=False) 
    )


def test_initial_clone_sizes4():
    with pytest.raises(ValueError) as exc_info:
        p = Parameters(
            algorithm='Moran2D',
            times=TimeParameters(max_time=10, division_rate=1),
            population=PopulationParameters(initial_cells=1000, cell_in_own_neighbourhood=False)
        )

    assert "Square grid not compatible" in str(exc_info)


def test_initial_clone_sizes4():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.arange(10, 110, 10))
    )
    s = p.get_simulator()
    s.run_sim()

    # There are multiple clones in the simulation
    assert len(s.view_clone_info()) > 1

    s.muller_plot(figsize=(5, 5))

    
def test_initial_clone_sizes5():

    p = Parameters(
        algorithm="Moran2D",
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(grid_shape=(60, 10), cell_in_own_neighbourhood=False) 
    )
    s = p.get_simulator()
    s.run_sim()
    s.plot_grid(figsize=(6, 1))

    
def test_initial_clone_sizes6():

    initial_grid = np.repeat([0, 1], 50).reshape(10, 10)

    p = Parameters(
        algorithm="Moran2D",
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=False) 
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_initial_clone_sizes7():
    p = Parameters(
        algorithm="Moran2D",
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(800).reshape(40, 20), cell_in_own_neighbourhood=False) 
    )
    s = p.get_simulator()
    s.run_sim()
    s.plot_grid(figsize=(8, 4))


def test_initial_clone_sizes8():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(16))
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_initial_clone_sizes9():
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(16).reshape(4, 4), cell_in_own_neighbourhood=False) 
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_initial_clone_sizes10():
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.arange(16))
    )
    assert p.population.initial_cells == 120
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))

    
