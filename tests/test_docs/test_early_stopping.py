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
                                          EndConditionError, FeatureValue,
                                          FitnessCalculator, FitnessParameters,
                                          Gene, NormalDist, Parameters,
                                          PlotColourMaps, PlottingParameters,
                                          PopulationParameters, TimeParameters)


def test_early_stopping():
    np.random.seed(1)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=20, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(100)),
    )
    s = p.get_simulator()
    s.run_sim()
    s.plot_surviving_clones_for_non_mutation()

def test_early_stopping1():

    def stop_when_10_clones_left(sim):
        current_clones = sim.population_array[:, sim.plot_idx-1]
        
        num_surviving = (current_clones > 0).sum()
        
        if num_surviving <= 10: 
            sim.stop_condition_result = num_surviving
            raise EndConditionError()
            
    np.random.seed(1)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=20, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(100)),
        end_condition_function=stop_when_10_clones_left  # Pass the stop function in
    )
    s = p.get_simulator()
    s.run_sim()

    s.plot_surviving_clones_for_non_mutation()

    s.stop_time

    s.stop_condition_result


def test_early_stopping2():
    cm1 = PlotColourMaps(
        colour_rules=[
            ColourRule(  # Initial clones are blue
                rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.INITIAL,   
                        value=True
                    )
                ], 
                colourmap=cm.Blues 
            ), 
            ColourRule(  # Mutant clones are red
                rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.INITIAL,   
                        value=False
                    )
                ], 
                colourmap=cm.Reds
            ), 
        ], 
        use_fitness=False
    )

    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(
            max_time=50,  # Run up to time 50.
            division_rate=1
        ),
        population=PopulationParameters(initial_cells=10000, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            fitness_calculator=fitness_calculator, 
            mutation_rates=0.2
        ),
        plotting=PlottingParameters(plot_colour_maps=cm1)
    )
    s = p.get_simulator()
    s.run_sim()

    s.plot_grid(figsize=(5, 5))

    s.plot_grid(figsize=(5, 5), t=40)

    def stop_when_fully_mutant(sim):
        WT_pop = sim.population_array[0, sim.plot_idx-1]
        if WT_pop == 0:
            raise EndConditionError()
        
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(
            max_time=50,  # Run up to time 50.
            division_rate=1
        ),
        population=PopulationParameters(initial_cells=10000, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            fitness_calculator=fitness_calculator, 
            mutation_rates=0.2
        ),
        plotting=PlottingParameters(plot_colour_maps=cm1),
        end_condition_function=stop_when_fully_mutant
    )
    s = p.get_simulator()
    s.run_sim()

    s.stop_time

    s.plot_grid(figsize=(5, 5))

    s.plot_grid(figsize=(5, 5), t=-2, index_given=True)


def test_early_stopping3():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=NormalDist(mean=1.1, var=0.1), synonymous_proportion=0),
            Gene(name='Gene2', mutation_distribution=NormalDist(mean=1, var=0.01), synonymous_proportion=0)
        ], 
        multi_gene_array=True
    )

    def stop_when_both_genes_mutated(sim):
        if np.any(~np.isnan(sim.raw_fitness_array).any(axis=1)):
            
            # Save the clone id(s) of the double mutant(s)
            sim.stop_condition_result = np.where(~np.isnan(sim.raw_fitness_array).any(axis=1))[0]
            raise EndConditionError()
        
    rules = [
        ColourRule(  # No genes mutated, light blue colour
            rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.GENES_MUTATED,   
                        value=set()  # Empty set for "no genes mutated"
                )
            ], 
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.Blues).to_rgba
        ), 
        ColourRule(  # First gene mutated, dark Red
            rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.GENES_MUTATED,   
                        value="Gene1"   # This will not match clones if other genes are mutated too! 
                )
            ], 
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=-2, vmax=1), cmap=cm.Reds).to_rgba,
        ),
        ColourRule(  # Second gene mutated, yellow
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
                        value={"Gene1", "Gene2"}  # This will match clones with both Gene1 and Gene2 mutated
                )
            ], 
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=-5, vmax=1), cmap=cm.Purples).to_rgba
        )
    ]


    cm2 = PlotColourMaps(
        colour_rules=rules
    )

    np.random.seed(1)
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=50, division_rate=1),
        population=PopulationParameters(initial_cells=10000, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(
            fitness_calculator=fit_calc, 
            mutation_rates=0.001
        ),
        plotting=PlottingParameters(plot_colour_maps=cm2),
        end_condition_function=stop_when_both_genes_mutated,
    )
    s = p.get_simulator()
    s.run_sim()
    # Print the time the simulation stopped
    s.stop_time

    s.plot_grid(figsize=(5, 5))

    clone_info = s.view_clone_info(include_raw_fitness=True)

    clone_info[clone_info['clone id'].isin(s.stop_condition_result)][
          ['clone id', 'fitness', 'Gene1', 'Gene2', 
           'generation born', 'parent clone id']
    ]
