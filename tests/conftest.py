from collections import namedtuple
import pytest
from clone_competition_simulation.parameters.parameters import Parameters
from plotting.colourscales import ColourScale


def pytest_addoption(parser):
    all_algorithms = Parameters.algorithm_options
    for alg in all_algorithms:
        parser.addoption("--{}".format(alg), action="store_true", help="include {} in tests".format(alg))


def pytest_generate_tests(metafunc):
    if "algorithm" in metafunc.fixturenames:
        algorithms = []
        all_algorithms = Parameters.algorithm_options
        for alg in all_algorithms:
            if metafunc.config.getoption(alg):
                algorithms.append(alg)
        if len(algorithms) == 0:
            # None in particular requested. Run all
            algorithms = all_algorithms
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

    cs_label = ColourScale(
            name='4 labels',
            all_clones_noisy=False,
            colourmaps={KEY1(label=0): init_colour,
                        KEY1(label=1): green,
                        KEY1(label=2): blue,
                        KEY1(label=3): red
                 },
            use_fitness=True
        )
    return cs_label