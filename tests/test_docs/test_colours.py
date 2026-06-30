"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

from clone_competition_simulation import (CloneFeature, ColourRule,
                                          FeatureValue, FitnessCalculator,
                                          FitnessParameters, FixedValue, Gene,
                                          LabelParameters, NormalDist,
                                          Parameters, PlotColourMaps,
                                          PlottingParameters,
                                          PopulationParameters, TimeParameters)


@pytest.fixture
def colour_rule():
    return ColourRule(
        colourmap=cm.Reds
    )

@pytest.fixture
def fit_calc():
    fit_calc = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(0.1), synonymous_proportion=0.5)],
    )
    return fit_calc


def plot_with_colour_map(plot_colour_maps, fit_calc):
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=100),
        fitness=FitnessParameters(mutation_rates=0.05, 
                                  fitness_calculator=fit_calc), 
        plotting=PlottingParameters(plot_colour_maps=plot_colour_maps)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_colours():
    fit_calc = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(0.1), synonymous_proportion=0.5)],
    )

    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=100),
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fit_calc)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))


def test_colours1(colour_rule, fit_calc):
    colour_rule = ColourRule(
        colourmap=cm.viridis
    )
    # And create a PlotColourMaps object using the colour rule
    plot_colour_maps = PlotColourMaps(colour_rules=[colour_rule])

    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours2(colour_rule, fit_calc):
    # And create a PlotColourMaps object using the colour rule
    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule], 
        use_fitness=True
    )

    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours3(colour_rule, fit_calc):
    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule], 
        use_fitness=True, 
        all_clones_noisy=True
    )
    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours4(colour_rule, fit_calc):
    def my_noise():
        return np.random.normal(scale=0.3)

    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule], 
        use_fitness=True, 
        all_clones_noisy=True, 
        random_noise_fn=my_noise
    )
    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours5(fit_calc):
    colour_rule = ColourRule(
        rule_filter=[  # Define the rule(s) for this clone subset
            FeatureValue(  # This defines a feature and a value
                clone_feature=CloneFeature.INITIAL,   # Whether the clones are initial or not
                value=False  # Here we select the clones that are not initial, i.e. are born from mutations
            )
        ], 
        colourmap=cm.Blues  # Assign a colour map for this clone subset
    )

    # Then define the Plot colours. We are not setting use_fitess=True, so clones will # be given a random colour from the assigned colourmap
    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule], 
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.full(10, 10)),   # 10 initial clones
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fit_calc), 
        plotting=PlottingParameters(plot_colour_maps=plot_colour_maps)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5), show_mutations_with_x=False)


def test_colours6(fit_calc):
    colour_rule1 = ColourRule(
        rule_filter=[  # Define the rule(s) for this clone subset
            FeatureValue(  # This defines a feature and a value
                clone_feature=CloneFeature.INITIAL,   # Whether the clones are initial or not
                value=False  # Here we select the clones that are not initial, i.e. are born from mutations
            )
        ], 
        colourmap=cm.Blues  # Assign a colour map for this clone subset
    )
    colour_rule2 = ColourRule(
        rule_filter=[  # Define the rule(s) for this clone subset
            FeatureValue(  # This defines a feature and a value
                clone_feature=CloneFeature.INITIAL,   # Whether the clones are initial or not
                value=True  # Here we select the clones that *are* initial
            )
        ], 
        colourmap=cm.Reds  # Assign a colour map for this clone subset
    )

    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule1, colour_rule2], 
    )

    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours7(fit_calc):
    colour_rule1 = ColourRule(
        rule_filter=[  # Define the rule(s) for this clone subset
            FeatureValue(  # This defines a feature and a value
                clone_feature=CloneFeature.INITIAL,   # Whether the clones are initial or not
                value=False  # Here we select the clones that are not initial, i.e. are born from mutations
            )
        ], 
        colourmap=cm.Blues  # Assign a colour map for this clone subset
    )

    colour_rule2 = ColourRule(
        # Do not assign a rule_filter to include all clones 
        colourmap=cm.Reds  # Assign a colour map for this clone subset
    )

    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule1, colour_rule2], 
    )
    plot_with_colour_map(plot_colour_maps, fit_calc)


def test_colours8(fit_calc):
    colour_rule1 = ColourRule(
        rule_filter=[  # Define the rule(s) for this clone subset
            FeatureValue(  # This defines a feature and a value
                clone_feature=CloneFeature.INITIAL,   # Whether the clones are initial or not
                value=False  # Here we select the clones that are not initial, i.e. are born from mutations
            )
        ], 
        colourmap=cm.Blues  # Assign a colour map for this clone subset
    )

    colour_rule2 = ColourRule(
        # Do not assign a rule_filter to include all clones 
        colourmap=cm.Reds  # Assign a colour map for this clone subset
    )
    plot_colour_maps = PlotColourMaps(
        colour_rules=[colour_rule1, colour_rule2], 
        use_fitness=True
    )
    plot_with_colour_map(plot_colour_maps, fit_calc)



def test_colours9():
    rules = [
        ColourRule(  # Initial clones are blue
            rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.INITIAL,   
                        value=True  
                )
            ], 
            colourmap=cm.Blues 
        ), 
        ColourRule(  # Label 1 clones are red
            rule_filter=[ 
                    FeatureValue(  
                        clone_feature=CloneFeature.LABEL,   
                        value=1
                )
            ], 
            # This scales the map so the values used (between 0 and 1) are all brighter reds
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cm.Reds).to_rgba
        ), 
        ColourRule(  # Label 2 clones are yellow/green
            rule_filter=[
                    FeatureValue(  
                        clone_feature=CloneFeature.LABEL,   
                        value=2
                )
            ], 
            # Scaling the viridis colourmap to use yellow/green colours
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=-5, vmax=1), cmap=cm.viridis).to_rgba
        )
    ]

    plot_colour_maps = PlotColourMaps(
        colour_rules=rules
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.full(10, 100)), 
        labels=LabelParameters(
            label_times=[2, 5], 
            label_frequencies=[0.05, 0.1],
            label_values=[1, 2], 
            label_fitness=[1.5, 1.5],
        ),
        # We are combining label fitness, so need to supply a FitnessCalculator which defines how the fitnesses will be combined.
        fitness=FitnessParameters(fitness_calculator=FitnessCalculator(genes=[])),   
        plotting=PlottingParameters(plot_colour_maps=plot_colour_maps)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(6, 6), show_mutations_with_x=False)


def test_colours10():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0),
            Gene(name='Gene2', mutation_distribution=FixedValue(1.2), synonymous_proportion=0)
        ], 
        multi_gene_array=True   # This must be set to True
    )

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


    plot_colour_maps = PlotColourMaps(
        colour_rules=rules
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.full(10, 20)), 
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fit_calc),
        plotting=PlottingParameters(plot_colour_maps=plot_colour_maps)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(6, 6), show_mutations_with_x=False)



def test_colours11():

    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0),
            Gene(name='Gene2', mutation_distribution=FixedValue(1.2), synonymous_proportion=0)
        ], 
        multi_gene_array=True   # This must be set to True
    )

    # Define the list of rules
    rules = [
        ColourRule(  # Clones with label 0 and last mutated with Gene1 are Blue
            rule_filter=[ 
                FeatureValue(  
                    clone_feature=CloneFeature.LAST_MUTATED_GENE,   
                    value="Gene1"  
                ), 
                FeatureValue(  
                    clone_feature=CloneFeature.LABEL,   
                    value=0
                )
            ], 
            colourmap=cm.Blues 
        ), 
        ColourRule(  # Clones with label 1 and last mutated with Gene2 are Red
            rule_filter=[ 
                FeatureValue(  
                    clone_feature=CloneFeature.LAST_MUTATED_GENE,   
                    value="Gene2"    
                ),
                FeatureValue(  
                    clone_feature=CloneFeature.LABEL,   
                    value=1
                )
            ], 
            colourmap=cm.Reds,
        ),
        ColourRule(  # Other clones with label 0 are white
            rule_filter=[
                FeatureValue(  
                    clone_feature=CloneFeature.LABEL,   
                    value=0
                )
            ], 
            # Any function that returns a valid colour can be used for the colourmap:
            colourmap=lambda x: "#FFFFFF"  
        ), 
        ColourRule(  # Other clones with label 1 are grey
            rule_filter=[
                FeatureValue(  
                    clone_feature=CloneFeature.LABEL,   
                    value=1
                )
            ], 
            colourmap=cm.Greys
        )
    ]

    plot_colour_maps = PlotColourMaps(
        colour_rules=rules
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[500, 500]), 
        labels=LabelParameters(
            initial_label_array=[0, 1]  # Apply some labels
        ),
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fit_calc),
        plotting=PlottingParameters(plot_colour_maps=plot_colour_maps)
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(6, 6), show_mutations_with_x=False)



def test_colours12():

    fit_calc = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(0.1), synonymous_proportion=0.5)],
    )

    np.random.seed(4)
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=100), 
        fitness=FitnessParameters(mutation_rates=0.05, fitness_calculator=fit_calc),
    )
    s = p.get_simulator()
    s.run_sim()
    s.muller_plot(figsize=(5, 5))

    new_colour_maps = PlotColourMaps(
        colour_rules=[
            ColourRule(
                colourmap=cm.magma,
            )
        ]
    )

    s.set_colour_maps(new_colour_maps)
    s.muller_plot(figsize=(5, 5))
