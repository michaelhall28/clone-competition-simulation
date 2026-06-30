"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from clone_competition_simulation import (CloneFeature, ColourRule,
                                          FeatureValue, FitnessCalculator,
                                          FitnessParameters, FixedValue, Gene,
                                          Parameters, PlotColourMaps,
                                          PlottingParameters,
                                          PopulationParameters, TimeParameters,
                                          TreatmentParameters)


def test_treatments():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[500, 500]),  
        fitness=FitnessParameters(initial_fitness_array=[1, 1.5]) 
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments2():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[500, 500]),  
        fitness=FitnessParameters(initial_fitness_array=[1, 1.5]),  
        treatment=TreatmentParameters(
            treatment_timings=[4],  
            treatment_effects=[
                [1, 0.5]  
            ],  
            treatment_replace_fitness=False   
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments3():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[500, 500]),  
        fitness=FitnessParameters(initial_fitness_array=[1, 1.5]),  
        treatment=TreatmentParameters(
            treatment_timings=[3, 5, 8],  
            treatment_effects=[
                [1, 0.5],  
                [3, 0.5], 
                [1, 1]  
            ],  
            treatment_replace_fitness=False   
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments4():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[500, 500]),  
        fitness=FitnessParameters(initial_fitness_array=[1, 1.5]),  
        treatment=TreatmentParameters(
            treatment_timings=[3, 6],  
            treatment_effects=[
                [3, 0.5], 
                [1, 1]   
            ],  
            treatment_replace_fitness=True  
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments5():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(2), synonymous_proportion=0), 
            Gene(name='Gene3', mutation_distribution=FixedValue(3), synonymous_proportion=0)
        ], 
        multi_gene_array=True,  
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[300, 300, 300]), 
        fitness=FitnessParameters(
            initial_fitness_array=[1, 2, 3],  
            initial_mutant_gene_array=["Gene1", "Gene2", "Gene3"],   
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments6():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1), synonymous_proportion=0), 
            Gene(name='Gene2', mutation_distribution=FixedValue(2), synonymous_proportion=0), 
            Gene(name='Gene3', mutation_distribution=FixedValue(3), synonymous_proportion=0)
        ], 
        multi_gene_array=True,  
    )
    
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[300, 300, 300]),  
        fitness=FitnessParameters(
            initial_fitness_array=[1, 2, 3], 
            initial_mutant_gene_array=["Gene1", "Gene2", "Gene3"],   
            fitness_calculator=fit_calc
        ), 
        treatment=TreatmentParameters(
            # Define the treatment. 
            treatment_timings=[3, 6], 
            treatment_effects=[
                [1, 3, 1, 1/3],  
                [1, 1, 2, 1]   
            ],  
            treatment_replace_fitness=False   # The `treatment_effects` will multiply fitness
        )
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_treatments7():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1), synonymous_proportion=0),
            Gene(name='Gene2', mutation_distribution=FixedValue(3), synonymous_proportion=0)
        ], 
        multi_gene_array=True, 
        combine_mutations='replace', 
        combine_array='add'   
    )

    cm2 = PlotColourMaps(
        colour_rules=[
            ColourRule(  
                rule_filter=[ 
                        FeatureValue(  
                            clone_feature=CloneFeature.GENES_MUTATED,   
                            value=set()  # Empty set for "no genes mutated"
                    )
                ], 
                colourmap=cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.Blues).to_rgba
            ), 
            ColourRule( 
                rule_filter=[ 
                        FeatureValue(  
                            clone_feature=CloneFeature.GENES_MUTATED,   
                            value="Gene1"   
                    )
                ], 
                colourmap=cm.ScalarMappable(norm=Normalize(vmin=-2, vmax=1), cmap=cm.Reds).to_rgba,
            ),
            ColourRule(  
                rule_filter=[ 
                        FeatureValue(  
                            clone_feature=CloneFeature.GENES_MUTATED,   
                            value="Gene2"
                    )
                ], 
                colourmap=cm.ScalarMappable(norm=Normalize(vmin=-20, vmax=2), cmap=cm.inferno).to_rgba,
            ), 
            ColourRule(  # Both genes mutated, purple
                rule_filter=[ 
                        FeatureValue(  
                            clone_feature=CloneFeature.GENES_MUTATED,   
                            value={"Gene1", "Gene2"}  
                    )
                ], 
                colourmap=cm.ScalarMappable(norm=Normalize(vmin=-5, vmax=1), cmap=cm.Purples).to_rgba
            )
        ]
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=1000), 
        fitness=FitnessParameters(
            fitness_calculator=fit_calc, 
            mutation_rates=0.01
        ), 
        treatment=TreatmentParameters(
            # Define the treatment. 
            treatment_timings=[5], 
            treatment_effects=[
                [1, 3, 1],  
            ],  
            treatment_replace_fitness=True   
        ), 
        plotting=PlottingParameters(plot_colour_maps=cm2)
    )
    s = p.get_simulator()
    s.run_sim()

    s.muller_plot(figsize=(10, 10)) 
