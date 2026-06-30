"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from clone_competition_simulation import (DifferentiatedCellsParameters,
                                          FitnessCalculator, FitnessParameters,
                                          Gene, NormalDist, Parameters,
                                          PopulationParameters, TimeParameters)


def test_diff_cells():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(5000)),
        differentiated_cells=DifferentiatedCellsParameters(r=0.15, gamma=3)
    )
    s = p.get_simulator()
    s.run_sim()

    s.population_array
    s.diff_cell_population

    a_cells = s.population_array[4].toarray().astype(int)[0]
    b_cells = s.diff_cell_population[4].toarray().astype(int)[0]
    total_cells = a_cells + b_cells

    a_cells[:8]
    b_cells[:8]
    total_cells[:8]

    csd_without_diff_cells = s.get_clone_size_distribution_for_non_mutation(
        include_diff_cells=False, t=2)
    csd_with_diff_cells = s.get_clone_size_distribution_for_non_mutation(
        include_diff_cells=True, t=2)

    s.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=False, show_spm_fit=False, 
                                                legend_label='Progenitor only')
    s.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, show_spm_fit=False, ax=plt.gca(), 
                                                legend_label='Progenitor + differentiated')
    plt.legend()

    s.plot_clone_size_distribution_for_non_mutation(include_diff_cells=True, t=5)


def test_diff_cells2():


    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=5000),
        fitness=FitnessParameters(mutation_rates=0.01, fitness_calculator=fitness_calculator),
        differentiated_cells=DifferentiatedCellsParameters(r=0.15, gamma=3)
    )
    s = p.get_simulator()
    s.run_sim()

    s.get_mutant_clone_sizes(include_diff_cells=True, t=3, non_zero_only=True)
    s.get_mutant_clone_size_distribution(include_diff_cells=True, t=3)


def test_diff_cells3():
    np.random.seed(0)
    start_time = time.time()
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(times=[0, 20, 40], division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(10000).reshape(100, 100), cell_in_own_neighbourhood=False),
    )
    s = p.get_simulator()
    s.run_sim()
    end_time = time.time()
    end_time - start_time


def test_diff_cells4():
    np.random.seed(0)
    start_time = time.time()
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(times=[0, 20, 40], division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(10000).reshape(100, 100), cell_in_own_neighbourhood=False),
        differentiated_cells=DifferentiatedCellsParameters(r=0.15, gamma=3)
    )
    s = p.get_simulator()
    s.run_sim()
    end_time = time.time()

    end_time - start_time

    np.random.seed(0)
    start_time = time.time()
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(times=[0, 20, 40], division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(10000).reshape(100, 100), cell_in_own_neighbourhood=False),
        differentiated_cells=DifferentiatedCellsParameters(
            r=0.15, gamma=3, 
            stratification_sim_proportion=0.999   # This means simulating diff cells with a 99.9% chance of being observed
        )
    )
    s2 = p.get_simulator()
    s2.run_sim()
    end_time = time.time()

    end_time - start_time

    s.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, show_spm_fit=False, 
                                                legend_label='Full simulation', 
                                                plot_kwargs={'marker': 'x'})
    s2.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, ax=plt.gca(), show_spm_fit=False, 
                                                legend_label='Faster partial simulation', 
                                                plot_kwargs={'marker': 's', 'alpha': 0.5})
    plt.legend()

    
