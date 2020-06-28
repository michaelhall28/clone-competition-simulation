from clone_competition_simulation.parameters import Parameters


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
