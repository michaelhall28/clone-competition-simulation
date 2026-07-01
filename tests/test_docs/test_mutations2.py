"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""

import matplotlib.pyplot as plt
import numpy as np
from clone_competition_simulation import (
    Parameters,
    TimeParameters, 
    PopulationParameters,
    FitnessParameters,
    Gene, 
    FitnessCalculator, 
    FixedValue
)


def test_mutations():
    gene = Gene(name='Gene1', mutation_distribution=FixedValue(2), synonymous_proportion=0)
    fit_calc = FitnessCalculator(genes=[gene])

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=50),
        fitness=FitnessParameters(
            mutation_rates=0.01, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))

    np.testing.assert_array_equal(
        s.view_clone_info()['parent clone id'], 
        [-1, 0, 0, 1, 0, 3]
    )

    s.population_array.toarray()[:, -10:]

    s.get_mutant_clone_sizes()

    s.get_mutant_clone_sizes(t=9, non_zero_only=True, gene_mutated='Gene1', selection='ns')

    s.get_mutant_clone_size_distribution()

    s.get_clone_ancestors(5)

    s.get_clone_descendants(1)

    gene = Gene(name='Gene1', mutation_distribution=FixedValue(1.5), synonymous_proportion=0.5)
    fit_calc = FitnessCalculator(genes=[gene])

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=10000),
        fitness=FitnessParameters(
            mutation_rates=0.01, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))

    s.muller_plot(figsize=(5, 5), show_mutations_with_x=False)


    s.muller_plot(figsize=(5, 5), show_mutations_with_x=False, min_size=30)

    fig, ax = plt.subplots(figsize=(5, 5))
    s.plot_incomplete_moment(ax=ax)


def test_mutations2():
    gene = Gene(name='Gene1', mutation_distribution=FixedValue(1), synonymous_proportion=0.5)
    fit_calc = FitnessCalculator(genes=[gene])

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=10000),
        fitness=FitnessParameters(
            mutation_rates=0.01, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    fig, ax = plt.subplots(figsize=(5, 5))
    s.plot_incomplete_moment(ax=ax)
    
