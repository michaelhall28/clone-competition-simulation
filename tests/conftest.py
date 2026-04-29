from collections import namedtuple

import pytest

from src.clone_competition_simulation.fitness import FitnessCalculator, Gene, UnboundedFitness, NormalDist
from src.clone_competition_simulation.parameters.algorithm_validation import Algorithm
from src.clone_competition_simulation.plotting.plot_colours import PlotColourMaps, ColourRule, CloneFeature, FeatureValue



def pytest_addoption(parser):
    for alg in Algorithm:
        parser.addoption(f"--{alg.name}", action="store_true", help=f"include {alg.name} in tests")


def pytest_generate_tests(metafunc):
    if "algorithm" in metafunc.fixturenames:
        algorithms = []
        for alg in Algorithm:
            if metafunc.config.getoption(alg.name):
                algorithms.append(alg)
        if len(algorithms) == 0:
            # None in particular requested. Run all
            algorithms = [a for a in Algorithm]
        metafunc.parametrize("algorithm", algorithms)



# Must define the namedtuples globally for the pickling to work
KEY1 = namedtuple('KEY1', ['label'])


@pytest.fixture()
def cs_label():
    # matplotlib colourmaps do not seem to work the same the first time they are called.
    # Use deterministic functions
    def init_colour(rate):
        return "#FEEBAD"

    def green(rate):
        return "#158710"

    def blue(rate):
        return "#1A1EA2"

    def red(rate):
        return "#C62619"

    cs_label = PlotColourMaps(
        all_clones_noisy=False,
        colour_rules=[
            ColourRule(
                rule_filter=[
                    FeatureValue(
                        clone_feature=CloneFeature.LABEL,
                        value=0
                    )
                ], 
                colourmap=init_colour
            ), 
            ColourRule(
                rule_filter=[
                    FeatureValue(
                        clone_feature=CloneFeature.LABEL,
                        value=1
                    )
                ], 
                colourmap=green
            ), 
            ColourRule(
                rule_filter=[
                    FeatureValue(
                        clone_feature=CloneFeature.LABEL,
                        value=2
                    )
                ], 
                colourmap=blue
            ), 
            ColourRule(
                rule_filter=[
                    FeatureValue(
                        clone_feature=CloneFeature.LABEL,
                        value=3
                    )
                ], 
                colourmap=red
            )
        ],
        use_fitness=True
    )
    return cs_label


@pytest.fixture()
def mutation_generator():
    return FitnessCalculator(
        combine_mutations='multiply',
        multi_gene_array=False,
        genes=(Gene(name='all', mutation_distribution=NormalDist(0.1),
                    synonymous_proportion=0.5, weight=1),),
        mutation_combination_class=UnboundedFitness()
    )
