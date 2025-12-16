import pytest
import numpy as np
from clone_competition_simulation.parameters import (
    Parameters,
    TimeParameters,
    PopulationParameters,
    FitnessParameters,
    DifferentiatedCellsParameters,
    PlottingParameters,
    LabelParameters,
    Algorithm
)
import matplotlib.pyplot as plt
from clone_competition_simulation.fitness.fitness_classes import FixedValue, NormalDist, ExponentialDist, UniformDist, Gene, MutationGenerator, \
    BoundedLogisticFitness
from clone_competition_simulation.simulation_algorithms.general_differentiated_cell_class import set_gsl_random_seed
from matplotlib.ticker import NullFormatter
from clone_competition_simulation.plotting.colourscales import ColourScale
from collections import namedtuple
import os
import random
import warnings
from scipy.sparse import SparseEfficiencyWarning

from clone_competition_simulation.parameters.algorithm_validation import AlgorithmClass
from parameters import TreatmentParameters
from tests.utilities import (
    compare_to_old_results,
    next_ax,
    INITIAL_CELLS,
    MAX_TIME,
    DIVISION_RATE,
    PLOT_DIR,
    get_plots
)
warnings.simplefilter('ignore',SparseEfficiencyWarning)


@pytest.fixture(scope="module")
def monkeymodule():
    mpatch = pytest.MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def mock_random(monkeymodule):
    """
    np.random.RandomState should be consistent across numpy versions,
    whereas the normal np.random functions might not be.

    Args:
        monkeypatch:

    Returns:

    """
    rng = np.random.RandomState()
    monkeymodule.setattr('numpy.random', rng)


@pytest.fixture(scope='session')
def axes():
    return get_plots()


def test_simple(mock_random, axes, algorithm, overwrite_results=False):
    # Does it run with simplest settings
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm,
                   population=PopulationParameters(initial_cells=INITIAL_CELLS,
                                                              cell_in_own_neighbourhood=False),
                   times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE)
                   )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='simple_run', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Basic run')


def test_multiple_clones(mock_random, axes, algorithm, overwrite_results=False):
    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    #  Multiple clones
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm,
                   population=PopulationParameters(
                                  initial_size_array=initial_size_array,
                                  initial_grid=initial_grid,
                                  cell_in_own_neighbourhood=False
                              ),
                   fitness=FitnessParameters(fitness_array=[1, 1.1, 0.9]),
                   times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE)
                   )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiple_clones', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multiple clones')


def test_mutations(mock_random, axes, algorithm, overwrite_results=False):
    # All the ways of adding mutations
    # Too many tests to do all combinations
    # Make sure every individual option is checked at least.
    # Any changes to the random process will make a large change here.
    # Not good for comparing algorithms, just for checking it runs.
    mutation_rate = 0.1

    # Need lower fitnesses for the branching process - numbers are not equivalent in the different algorithms
    if algorithm == Algorithm.BRANCHING:
        norm_mean = 1.02
        exp_mean = 1.02
        uniform_high = 1.05
    else:
        norm_mean = 4.5
        exp_mean = 1.3
        uniform_high = 1.2
    genes = [Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene(name='mild_driver', mutation_distribution=FixedValue(1.1), synonymous_proportion=0.5),
             Gene(name='random_driver', mutation_distribution=NormalDist(mean=norm_mean, var=0.05),
                  synonymous_proportion=0.8),
             Gene(name='uniform_driver', mutation_distribution=UniformDist(low=0.7, high=uniform_high),
                  synonymous_proportion=0.2),
             Gene(name='exp_driver', mutation_distribution=ExponentialDist(mean=exp_mean, offset=0.8),
                  synonymous_proportion=0.2)]
    mut_gen1 = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='add')

    #  Simple
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen1)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='simple_mutations', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Adding mutations')

    # High mutation rate
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=3, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=0.8, mutation_generator=mut_gen1)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='high_mutation_rate', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('High mutation rate')

    # Variable mutation rate
    mutation_rates = np.array([[0, 0.01], [3.5, 0.4]])
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rates, mutation_generator=mut_gen1)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='variable_mutation_rate', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Variable mutation rate')

    #  Multi-gene array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_mutations='add')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multi_gene_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multi-gene')

    #  Multiply fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='multiply')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiply_fitness', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multiply fitness')

    #  Replace fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='replace')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='replace_fitness', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Replace fitness')

    #  max fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='max')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='max', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('max')

    # min fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='min')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='min', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('min')

    #  Add array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_array='add',
                                combine_mutations='multiply')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='add_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Add array')

    #  Multiply array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_array='multiply',
                                combine_mutations='replace')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiply_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multiply array')

    #  Max array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_array='max',
                                combine_mutations='add')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='max_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Max array')

    #  Min array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_array='min',
                                combine_mutations='add')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=mutation_rate, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='min_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Min array')

    #  Logistic
    if algorithm == Algorithm.BRANCHING:
        h = 1.5
    else:
        h = 5
    genes = [Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene(name='uniform_driver', mutation_distribution=UniformDist(low=0.95, high=h),
                  synonymous_proportion=0.2)]

    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_array='multiply',
                                mutation_combination_class=BoundedLogisticFitness(1.1, 10),
                                combine_mutations='multiply')
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=20, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=0.5, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='logistic', overwrite_results=overwrite_results)
    ax = next_ax(axes, algorithm)
    sim.plot_average_fitness_over_time(ax=ax)
    ax.set_title('Logistic fitness')

    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_array='multiply',
                                mutation_combination_class=BoundedLogisticFitness(1.1, 10),
                                combine_mutations='multiply')
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_cells=INITIAL_CELLS, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=20, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_rates=0.5, mutation_generator=mut_gen)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='logistic_multi', overwrite_results=overwrite_results)
    ax = next_ax(axes, algorithm)
    sim.plot_average_fitness_over_time(ax=ax)
    ax.set_title('Logistic - multi gene')


def test_neutral_hallmarks(mock_random, axes, algorithm, overwrite_results=False):
    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.arange(100 ** 2, dtype=int).reshape(100, 100)
    else:
        initial_size_array = np.ones(100 ** 2)
        initial_grid = None

    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array,
                   initial_grid=initial_grid, cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='neutral_run', overwrite_results=overwrite_results)

    ax = next_ax(axes, algorithm)
    sim.plot_mean_clone_size_graph_for_non_mutation(ax=ax)
    ax.set_title('Mean clone size')

    ax = next_ax(axes, algorithm)
    sim.plot_surviving_clones_for_non_mutation(ax=ax)
    ax.set_title('Surviving clones')

    ax = next_ax(axes, algorithm)
    sim.plot_clone_size_scaling_for_non_mutation(times=[MAX_TIME / 2, MAX_TIME], ax=ax)
    ax.set_title('Clone size scaling')

    ax = next_ax(axes, algorithm)
    sim.plot_clone_size_distribution_for_non_mutation(ax=ax)
    ax.set_title('Clone size dist')


def test_imbalance(mock_random, axes, algorithm, overwrite_results=False):
    # Want large population so almost deterministic
    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.zeros((250, 250), dtype=int)
        initial_grid[range(250), range(250)] = 1
        initial_grid[range(250), range(249, -1, -1)] = 1
        initial_grid[125, range(250)] = 1
        initial_grid[range(250), 125] = 1
    else:
        initial_size_array = [250 ** 2 - 1000, 1000]
        initial_grid = None

    fitness_array = [1, 1.2]
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(fitness_array=fitness_array)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='non_neutral_run', overwrite_results=overwrite_results)

    ax = next_ax(axes, algorithm)
    ax.plot(sim.times, np.squeeze(sim.population_array[1].toarray()))
    # Add some extra lines to make it easier to compare plots with new versions
    for m in np.linspace(0.1, 1, 5):
        ax.plot(sim.times, 1000 + sim.times * m * 1000)
    ax.set_title('Mean - imbalance')


def test_b_cells(mock_random, axes, algorithm, mutation_generator, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if not algorithm.algorithm_class == AlgorithmClass.WF:
        if algorithm.two_dimensional:
            initial_size_array = None
            initial_grid = np.arange(4 ** 2).reshape((4, 4))
        else:
            initial_size_array = np.ones(30 ** 2)
            initial_grid = None
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(
                initial_size_array=initial_size_array,
                initial_grid=initial_grid,
                cell_in_own_neighbourhood=False
            ),
            times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
            differentiated_cells=DifferentiatedCellsParameters(r=0.2, gamma=2.1)
        )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells', overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='Full clone', ax=ax)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=False, legend_label='A cells', ax=ax)
        ax.set_title('B cells')

        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(max_time=MAX_TIME, division_rate=DIVISION_RATE),
            fitness=FitnessParameters(mutation_rates=0.1, mutation_generator=mutation_generator),
            differentiated_cells=DifferentiatedCellsParameters(r=0.2, gamma=2.1)
        )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells_with_mutation', overwrite_results=overwrite_results)
    else:
        ax.set_title('Not implemented')


def test_treatment_with_fixed_clones(mock_random, axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 0.5, 1.3], [0.4, 1.5, 0.7]]

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(fitness_array=[1.05, 1, 0.9]),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays,
                                      treatment_replace_fitness=False),
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_clones', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: clones')


def test_treatment_replace_with_fixed_clones(mock_random, axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 1, 1], [0.4, 1.5, 0.7]]

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)

    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(fitness_array=[1.05, 1, 0.9]),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays,
                                      treatment_replace_fitness=True),
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_clones_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: clones')


def test_treatment_with_multiple_genes(mock_random, axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 0.5, 1.3], [0.4, 1.25, 0.7]]

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1], 50 * 25).reshape(50, 50)
    else:
        initial_size_array = [400, 500]
        initial_grid = None

    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene(name='driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                                  fitness_array=[1.05, 1]),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays),
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: genes')

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                                  fitness_array=[1.05, 1], mutation_rates=0.005),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays),
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_muts', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: genes+muts')


def test_treatment_replace_with_multiple_genes(mock_random, axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 1, 1], [0.4, 1.25, 0.7]]

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1], 50 * 25).reshape(50, 50)
    else:
        initial_size_array = [400, 500]
        initial_grid = None

    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene(name='driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen, fitness_array=[1.05, 1]),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays,
                                      treatment_replace_fitness=True),
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: genes')

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=12, division_rate=DIVISION_RATE),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen, mutation_rates=0.005,
                                  fitness_array=[1.05, 1]),
        treatment=TreatmentParameters(treatment_timings=timings, treatment_effects=treatment_arrays,
                                      treatment_replace_fitness=False)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_muts_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: genes+muts')


def test_labels(mock_random, cs_label, axes, algorithm, overwrite_results=False):
    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene(name='driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2, 3], 40 * 10).reshape(40, 40)
    else:
        initial_size_array = [400, 500, 600, 700]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=MAX_TIME, division_rate=1),
        plotting=PlottingParameters(colourscales=cs_label),
        fitness=FitnessParameters(mutation_generator=mut_gen, fitness_array=[1.05, 1, 0.9, 1.02], mutation_rates=0.01),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        labels=LabelParameters(label_array=[0, 1, 2, 1])
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='initial_labels', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Initial labels')

    label_times = [2, 3]
    label_frequencies = [0.2, 0.3]
    label_values = [2, 3]
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=MAX_TIME, division_rate=1),
        plotting=PlottingParameters(colourscales=cs_label),
        fitness=FitnessParameters(mutation_generator=mut_gen, fitness_array=[1.05, 1, 0.9, 1.02]),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        labels=LabelParameters(label_array=[0, 1, 0, 1], label_times=label_times,
                               label_frequencies=label_frequencies, label_values=label_values)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='late_labels', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Late labels')


def test_irregular_sampling(mock_random, axes, algorithm, overwrite_results=False):
    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.arange(100 ** 2, dtype=int).reshape(100, 100)
    else:
        initial_size_array = np.ones(100 ** 2)
        initial_grid = None
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(times=[1.1, 2, 3.4, 12], division_rate=DIVISION_RATE)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='irregular_samples', overwrite_results=overwrite_results)
    sim.plot_mean_clone_size_graph_for_non_mutation(ax=ax)
    ax.set_title('Irregular samples')


def test_partially_simulating_B_cells(mock_random, axes, algorithm, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if algorithm.algorithm_class != AlgorithmClass.WF:
        if algorithm.two_dimensional:
            initial_size_array = None
            initial_grid = np.arange(30 ** 2).reshape((30, 30))
        else:
            initial_size_array = np.ones(30 ** 2)
            initial_grid = None
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(times=[1.1, 2, 3.4, 12], division_rate=DIVISION_RATE),
            differentiated_cells=DifferentiatedCellsParameters(r=0.2, gamma=2.1)
        )
        sim = p.get_simulator()
        sim.run_sim()
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=False, legend_label='A cells', ax=ax)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='Full clone', ax=ax)
        ax.set_title('B cells - partial sim')
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(times=[1.1, 2, 3.4, 12], division_rate=DIVISION_RATE),
            differentiated_cells=DifferentiatedCellsParameters(r=0.2, gamma=2.1, stratification_sim_percentile=0.9)
        )
        sim = p.get_simulator()
        sim.run_sim()
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='0.9', ax=ax)

        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(times=[1.1, 2, 3.4, 12], division_rate=DIVISION_RATE),
            differentiated_cells=DifferentiatedCellsParameters(r=0.2, gamma=2.1, stratification_sim_percentile=0.5)
        )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells_partial', overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='0.5', ax=ax)
        ax.legend()
    else:
        ax.set_title('Not implemented')


def test_too_many_sample_points(mock_random, axes, algorithm, overwrite_results=False):
    # If there are more sample points than divisions.
    # Need to reduce number of points.
    ax = next_ax(axes, algorithm)
    ax.set_title('Reduce samples')
    if algorithm.algorithm_class == AlgorithmClass.WF:
        div_rate = 0.1
    else:
        div_rate = 0.001

    TIMES = [1.5, 3.0, 6.0, 12.0, 24.0, 52.0, 78.0]
    grid_edge = 20
    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.arange(grid_edge ** 2).reshape((grid_edge, grid_edge))
    else:
        initial_size_array = np.ones(grid_edge ** 2)
        initial_grid = None

    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(times=TIMES, division_rate=div_rate),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='reduce_sample_points', overwrite_results=overwrite_results)
    sim.plot_mean_clone_size_graph_for_non_mutation(ax=ax)

    ax = next_ax(axes, algorithm)
    if algorithm in {'Moran', 'Moran2D', 'Branching'}:
        ax.set_title('Reduce samples - B')
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(
            algorithm=algorithm,
            population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(times=TIMES, division_rate=div_rate),
            differentiated_cells=DifferentiatedCellsParameters(r=0.1, gamma=0.2)
        )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='reduce_sample_points_B_cells',
                               overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, ax=ax)
    else:
        ax.set_title('Not implemented')


def test_induction_of_label_and_mutant(mock_random, axes, algorithm, overwrite_results=False):
    genes = [Gene(name='Neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene(name='Notch1', mutation_distribution=FixedValue(4), synonymous_proportion=0)]
    mutation_generator = MutationGenerator(genes=genes, combine_mutations='replace', multi_gene_array=True)
    Key1 = namedtuple('Key1', ['label'])
    green_clones = ColourScale(
        name='Single Green Mutant',
        all_clones_noisy=False,
        colourmaps={Key1(label=0): lambda x: (0, 0, 0, 1),
                    Key1(label=1): lambda x: (0.05, 0.75, 0.05, 1),
                    Key1(label=2): lambda x: (0.75, 0.05, 0.05, 1)
                    },
        use_fitness=True
    )

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.arange(30 ** 2).reshape((30, 30))
    else:
        initial_size_array = np.ones(30 ** 2)
        initial_grid = None

    # Single label event
    ax = next_ax(axes, algorithm)
    label_time = 3
    label_value = 1
    label_freq = 0.05
    if algorithm == Algorithm.BRANCHING:
        label_fitness = 1.01
    else:
        label_fitness = 4
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        times=TimeParameters(max_time=10, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_generator=mutation_generator),
        labels=LabelParameters(label_times=label_time, label_values=label_value, label_frequencies=label_freq,
                               label_fitness=label_fitness),
        plotting=PlottingParameters(colourscales=green_clones)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='single_label_with_mutant', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)

    # Multiple labelling events
    ax = next_ax(axes, algorithm)
    label_time = [3, 6]
    label_value = [1, 2]
    label_freq = [0.05, 0.1]
    if algorithm == Algorithm.BRANCHING:
        label_fitness = [1.01, 1.02]
    else:
        label_fitness = [4, 8]
    label_genes = [-1, 1]  # Apply the mutants to different genes
    np.random.seed(0)
    p = Parameters(
        algorithm=algorithm,
        population=PopulationParameters(
            initial_size_array=initial_size_array, initial_grid=initial_grid, cell_in_own_neighbourhood=False
        ),
        times=TimeParameters(max_time=10, division_rate=DIVISION_RATE),
        fitness=FitnessParameters(mutation_generator=mutation_generator),
        labels=LabelParameters(
            label_times=label_time, label_values=label_value, label_frequencies=label_freq,
            label_fitness=label_fitness, label_genes=label_genes),
        plotting=PlottingParameters(colourscales=green_clones)
    )
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiple_label_with_mutant', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)


def test_seven_cell_neighbourhood(mock_random, cs_label, axes, algorithm, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if algorithm.two_dimensional:
        np.random.seed(0)
        initial_grid = np.arange(30 ** 2).reshape((30, 30))
        p = Parameters(
            algorithm=algorithm,
            times=TimeParameters(max_time=MAX_TIME, division_rate=1),
            population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=True),
            plotting=PlottingParameters(colourscales=cs_label),
        )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='7_cell_neighbourhood', overwrite_results=overwrite_results)
        sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
        ax.set_title('7_cell_neighbourhood')


def test_plots(axes, algorithm):
    fig, axs = axes[algorithm]
    plt.figure(fig.number)
    fig.suptitle(algorithm, fontsize=14, y=0.93)
    for row in axs:
        for ax in row:
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())

    plt.savefig(os.path.join(PLOT_DIR, 'test_plots_{}.pdf'.format(algorithm)))


def test_animation(mock_random, algorithm):
    # Make animation
    #  just checking it runs. Can be compared visually or using the output files
    # Have to do after all other plots as this will reset the figures
    non_neut_genes = [Gene(name='mild_driver', mutation_distribution=NormalDist(mean=1.1, var=0.1),
                           synonymous_proportion=0.85)]
    mut_gen = MutationGenerator(multi_gene_array=False, genes=non_neut_genes, combine_mutations='add')

    if algorithm.two_dimensional:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2, 3], 10).reshape(4, 10)
        grid_size = None
    else:
        initial_size_array = [10, 15, 20, 25]
        initial_grid = None
        grid_size = 10

    mutation_rate = 0.05
    np.random.seed(0)
    random.seed(0)  # Animation currently uses random numbers from random module, not numpy
    p = Parameters(
        algorithm=algorithm,
        times=TimeParameters(max_time=MAX_TIME, division_rate=1),
        population=PopulationParameters(initial_size_array=initial_size_array, initial_grid=initial_grid,
                                        cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(mutation_generator=mut_gen, mutation_rates=mutation_rate)
    )
    sim = p.get_simulator()
    sim.run_sim()
    sim.animate(os.path.join(PLOT_DIR, 'test_ani_{}.mp4'.format(algorithm)), grid_size=grid_size)


