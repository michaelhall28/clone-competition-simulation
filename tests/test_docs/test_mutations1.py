"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
import pytest

from clone_competition_simulation import (ExponentialDist, FitnessCalculator,
                                          FitnessParameters, FixedValue, Gene,
                                          NormalDist, Parameters,
                                          PopulationParameters, TimeParameters,
                                          UniformDist)


@pytest.fixture
def fitness_calculator():
    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )
    return fitness_calculator


def test_mutations():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=1000)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_mutations1(fitness_calculator):

    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=1000),
        fitness=FitnessParameters(
            mutation_rates=0.01, 
            fitness_calculator=fitness_calculator
        ),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_mutations2(fitness_calculator):
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=1000),
        fitness=FitnessParameters(
            mutation_rates=[
                [1, 0.1],  # At time 1, start a mutation rate of 0.1 per cell division
                [4, 0.01]  # At time 4, start a mutation rate of 0.01 per cell division. This continues until the end.
            ], 
            fitness_calculator=fitness_calculator
        ),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_mutations3():
    gene1 = Gene(
        name='Gene1',  # This is the name for the gene. 
        mutation_distribution=UniformDist(0.5, 1.5),  # This defines the fitness effects of non-synonymous mutations
        synonymous_proportion=0.4   # This means 40% of the mutations will be synonymous
    )

    fit_calc = FitnessCalculator(genes=[gene1])
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=100, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            mutation_rates=0.01,
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()
    assert s.view_clone_info().loc[1:]['last gene mutated'].unique() == "Gene1"


def test_mutations4():
    gene1 = Gene(
        name='Gene1',  # This is the name for the gene. 
        mutation_distribution=UniformDist(0.5, 1.5),  # This defines the fitness effects of non-synonymous mutations
        synonymous_proportion=0.4   # This means 40% of the mutations will be synonymous
    )

    # One with a normal distribution
    gene_norm = Gene(name='GeneNorm', mutation_distribution=NormalDist(mean=0.6, var=0.1), synonymous_proportion=0.5)

    # One with an exponential distribution
    gene_exp = Gene(name='GeneExp', mutation_distribution=ExponentialDist(mean=1.05, offset=1), synonymous_proportion=0.4)

    # One where every non-synonymous mutation has the same fixed value of fitness
    gene_fix = Gene(name='GeneFix', mutation_distribution=FixedValue(value=1.01), synonymous_proportion=0.3)

    fit_calc = FitnessCalculator(genes=[gene1, gene_norm, gene_exp, gene_fix])
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=100, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            mutation_rates=0.03,
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.view_clone_info()


def test_mutations5():
    # Two genes, where gene1 is 3 times more likely to be mutated than gene2
    gene1 = Gene(name='Gene1', mutation_distribution=UniformDist(0.5, 1.1), synonymous_proportion=0.4, weight=3)
    gene2 = Gene(name='Gene2', mutation_distribution=UniformDist(1.1, 1.5), synonymous_proportion=0.4, weight=1)

    fit_calc = FitnessCalculator(genes=[gene1, gene2])
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=100, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            mutation_rates=0.1,
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.view_clone_info()['last gene mutated'].value_counts()


def test_mutations6():
    gene1 = Gene(name='Gene1', mutation_distribution=UniformDist(0.5, 1.1), synonymous_proportion=0.4, weight=3)
    gene2 = Gene(name='Gene2', mutation_distribution=UniformDist(1.1, 1.5), synonymous_proportion=0.5, weight=1)

    fit_calc = FitnessCalculator(genes=[gene1, gene2])
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=62500, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            mutation_rates=0.01,
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()
    assert isinstance(s.get_dnds(), float)
    assert isinstance(s.get_dnds(t=5), float)
    assert isinstance(s.get_dnds(gene='Gene1'), float)
    assert isinstance(s.get_dnds(gene='Gene2'), float)
    assert isinstance(s.get_dnds(gene='Gene2', min_size=5), float)

    s.plot_dnds()
    s.plot_dnds(gene='Gene1')
    s.plot_dnds(gene='Gene2', clear_previous=False)

    s.plot_dnds(gene='Gene2', min_size=5)
