"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import matplotlib.pyplot as plt
import numpy as np

from clone_competition_simulation import (BoundedLogisticFitness,
                                          EpistaticEffect, FitnessCalculator,
                                          FitnessParameters, FixedValue, Gene,
                                          Parameters, PopulationParameters,
                                          TimeParameters, UniformDist,
                                          add_array_fitness, add_fitness,
                                          max_array_fitness, max_fitness,
                                          min_array_fitness, min_fitness,
                                          multiply_array_fitness,
                                          multiply_fitness,
                                          priority_array_fitness,
                                          replace_fitness)


def test_mutations():
    fit_calc = FitnessCalculator(
        genes=[Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0)], 
        combine_mutations=multiply_fitness  # This is also the default option
    )

    np.random.seed(1)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(4)),
        fitness=FitnessParameters(
            initial_fitness_array=np.linspace(1, 1.3, 4),  # Giving the initial clones some different fitness values
            mutation_rates=0.1, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info()

    s.raw_fitness_array


def test_mutations2():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0)
        ], 
        combine_mutations=add_fitness  # Using add. Can also use combine_mutations="add"
    )

    # The simulation is otherwise the same as before
    np.random.seed(1)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(4)),
        fitness=FitnessParameters(
            initial_fitness_array=np.linspace(1, 1.3, 4),  # Giving the initial clones some different fitness values
            mutation_rates=0.1, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info()


def test_mutations3():
    def custom_combine(old, new):
        return old + new + 1

    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0)
        ], 
        combine_mutations=custom_combine  # Using the custom function
    )

def test_mutations4():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0)
        ], 
        multi_gene_array=True,
        combine_mutations=multiply_fitness, # This is the default option
        combine_array=multiply_array_fitness,  # This is also the default
    )

    np.random.seed(1)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(4)),
        fitness=FitnessParameters(
            initial_fitness_array=np.linspace(1, 1.3, 4),  # Giving the initial clones some different fitness values
            mutation_rates=0.1, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info()

    s.raw_fitness_array

    s.view_clone_info(include_raw_fitness=True)


def test_mutations5():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0)
        ], 
        multi_gene_array=True,
        combine_mutations=replace_fitness,   # With FixedValue this means further mutations will do nothing
        combine_array=add_array_fitness,  # Add the effects from the two genes
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=6),
        fitness=FitnessParameters(
            mutation_rates=0.15, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info(include_raw_fitness=True)


def test_mutations6():

    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0)
        ],                 
        # Add an epistatic effect. 
        # If both Gene1 and Gene2 are mutated, the clone has a fitness of 3
        # This replaces the individual effects of both genes
        epistatics=[
            EpistaticEffect(
                name='Epi1', 
                gene_names=['Gene1', 'Gene2'], 
                fitness_distribution=FixedValue(3)
            )
        ],
        multi_gene_array=True,
        combine_mutations=replace_fitness,   # With FixedValue this means further mutations will do nothing
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=6),
        fitness=FitnessParameters(
            mutation_rates=0.15, 
            fitness_calculator=fit_calc,     
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info(include_raw_fitness=True)


    fitness_combinations = fit_calc.plot_fitness_combinations()


def test_mutations7():

    fit_calc2 = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0),
            Gene(name='Gene3', mutation_distribution=UniformDist(1, 2), synonymous_proportion=0)],                            
        epistatics=[
            EpistaticEffect(
                name='Epi1', 
                gene_names=['Gene1', 'Gene2'], 
                fitness_distribution=FixedValue(3)
            )
        ],
        multi_gene_array=True,
        combine_array=multiply_array_fitness
    )
    fitness_combinations = fit_calc2.plot_fitness_combinations()


def test_mutations8():
    fit_calc2 = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0),
            Gene(name='Gene3', mutation_distribution=UniformDist(1, 2), synonymous_proportion=0)],                            
        epistatics=[
            EpistaticEffect(
                name='Epi1', 
                gene_names=['Gene1', 'Gene2'], 
                fitness_distribution=FixedValue(3)
            )
        ],
        multi_gene_array=True,
        combine_array=max_array_fitness
    )
    fitness_combinations = fit_calc2.plot_fitness_combinations()


def test_mutations9():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0, weight=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0)
        ],
        epistatics=[
            EpistaticEffect(
                name='Epi1', 
                gene_names=['Gene1', 'Gene2'], 
                fitness_distribution=FixedValue(2)
            )
        ],
        multi_gene_array=True,
        combine_mutations=replace_fitness,   # With FixedValue this means further mutations will do nothing
    )

    np.random.seed(1)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(
            initial_size_array=[6, 1, 1, 1, 1],   # Set up 4 1-cell clones and 6 WT cells
        ),
        fitness=FitnessParameters(
            mutation_rates=0.08, 
            fitness_calculator=fit_calc,     
            
            # The initial_mutant_gene_array defines the genes to be associated with the gene
            # The value is the name of the Gene
            # Use None for the clones with no mutation/associated gene
            initial_mutant_gene_array=[None, "Gene1", "Gene1", "Gene1", "Gene1"],  
            
            # Also need to define the fitness array or all initial clones will have a fitness=1
            # These fitness values do not have to equal the fitness values from the Gene assigned. 
            initial_fitness_array=[1, 1.1, 1.1, 1.1, 1.1],
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info(include_raw_fitness=True)


def test_mutations10():
    # Run a simulation with loads of mutations that multiply in fitness
    fit_calc = FitnessCalculator(
        genes=[Gene(name='Gene1', mutation_distribution=FixedValue(1.4), synonymous_proportion=0)], 
        combine_mutations=multiply_fitness 
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(
            initial_cells=1000
        ),
        fitness=FitnessParameters(
            mutation_rates=0.1, 
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info()[-10:]

    s.get_clone_ancestors(1009)


def test_mutations11():
    b = BoundedLogisticFitness(3)
    plt.plot(np.linspace(0, 10, 100), b.fitness(np.linspace(0, 10, 100)))
    plt.xlabel('Raw fitness')
    plt.ylabel('Transformed fitness')

def test_mutations12():
    b = BoundedLogisticFitness(3)
    plt.plot(np.linspace(0, 10, 100), b.fitness(np.linspace(0, 10, 100)))
    b2 = BoundedLogisticFitness(3, 10)
    plt.plot(np.linspace(0, 10, 100), b2.fitness(np.linspace(0, 10, 100)))
    plt.xlabel('Raw fitness')
    plt.ylabel('Transformed fitness')


def test_mutations13():
    fit_calc = FitnessCalculator(
        genes=[Gene(name='Gene1', mutation_distribution=FixedValue(1.4), synonymous_proportion=0)], 
        combine_mutations=multiply_fitness, 
        mutation_combination_class=BoundedLogisticFitness(3)   # This limits fitness to 3
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(
            initial_cells=1000
        ),
        fitness=FitnessParameters(
            mutation_rates=0.1, 
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.view_clone_info(include_raw_fitness=True)[-10:]



def test_mutations14():

    genes = [
        Gene(name="Gene1", mutation_distribution=FixedValue(1), synonymous_proportion=0),
        Gene(name="Gene2", mutation_distribution=FixedValue(1), synonymous_proportion=0),
        Gene(name="Gene3", mutation_distribution=FixedValue(1), synonymous_proportion=0),
        Gene(name="Gene4", mutation_distribution=FixedValue(1), synonymous_proportion=0),
    ]
    epistatic_effects = [
        EpistaticEffect(
            name="1+2", 
            gene_names=["Gene1", "Gene2"], 
            fitness_distribution=FixedValue(2)
        ), 
        EpistaticEffect(
            name="1+2+3", 
            gene_names=["Gene1", "Gene2", "Gene3"],
            fitness_distribution=FixedValue(3)
        ),
        EpistaticEffect(
            name="1+4", 
            gene_names=["Gene1", "Gene4"], 
            fitness_distribution=FixedValue(4)
        )
    ]

    fit_calc = FitnessCalculator(
        genes=genes,
        epistatics=epistatic_effects,
        combine_mutations=multiply_fitness
    )

    fitness_arrays = np.array([
        [1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan], # First
        [1, 1, 1, 1     , np.nan, np.nan, np.nan, np.nan], # Second
        [1, 1, 1, 1     , 1     , np.nan, np.nan, np.nan], # Second and third
    ])
    new_fitness_array, full_fitness_arrays = fit_calc.combine_vectors(fitness_arrays)
    np.testing.assert_almost_equal(
        full_fitness_arrays,
        np.array([
            [1, 1, 1, np.nan, np.nan, 2,     np.nan, np.nan], # First
            [1, 1, 1, 1     , np.nan, np.nan,     3, np.nan], # Second
            [1, 1, 1, 1     , 1     , np.nan,     3,      4], # Second and third
        ])
    )
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([2, 3, 12])
    )
