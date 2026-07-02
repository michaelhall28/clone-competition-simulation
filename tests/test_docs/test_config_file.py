
import os
import numpy as np

from clone_competition_simulation import (Algorithm, FitnessCalculator,
                                          FitnessParameters, Parameters,
                                          PlotColourMaps, PlottingParameters, 
                                          Gene, FixedValue, add_fitness)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_CONFIG_FILE = os.path.join(DIR_PATH, "..", "..", "example_run_config.yml")


def test_config():
    
    p = Parameters(run_config_file=LOCAL_CONFIG_FILE)


def test_config1():
    os.environ["CCS_RUN_CONFIG"] = LOCAL_CONFIG_FILE
    p =  Parameters()
    del os.environ["CCS_RUN_CONFIG"]


def test_config2():

    p = Parameters(run_config_file=LOCAL_CONFIG_FILE)
    assert p.algorithm == Algorithm.MORAN


def test_config3():

    p = Parameters(run_config_file=LOCAL_CONFIG_FILE, algorithm="Branching")

    assert p.algorithm == Algorithm.BRANCHING


def test_config4():

    p = Parameters(
        run_config_file=LOCAL_CONFIG_FILE,
        fitness=FitnessParameters(
            fitness_calculator=FitnessCalculator(
                multi_gene_array=True,
                genes=[
                    Gene(
                        name="Gene1", mutation_distribution=FixedValue(4), 
                        synonymous_proportion=0.5, weight=1
                    ), 
                    Gene(
                        name="Gene2", mutation_distribution=FixedValue(3), 
                        synonymous_proportion=0.5, weight=1
                    )
                ], 
                combine_mutations="add"
            )
        ), 
        plotting=PlottingParameters(
            plot_colour_maps=PlotColourMaps(
            )
        )
    )

    # Check the combine mutations function has been replaced
    assert p.fitness.fitness_calculator.combine_mutations == add_fitness

    # Check another fitness parameter is from the config
    np.testing.assert_array_equal(p.fitness.initial_mutant_gene_array, 
                                  [0, 1])

    # Check the colour rules are just the default ones
    assert len(p.plotting.plot_colour_maps.colour_rules) == 1

