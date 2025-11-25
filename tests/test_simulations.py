import pytest
import numpy as np
from clone_competition_simulation.parameters.parameters import Parameters
import pandas as pd
import matplotlib.pyplot as plt
from fitness.fitness_classes import FixedValue, NormalDist, ExponentialDist, UniformDist, Gene, MutationGenerator, \
    BoundedLogisticFitness
from simulation_algorithms.general_differentiated_cell_class import set_gsl_random_seed
from matplotlib.ticker import NullFormatter
from plotting.colourscales import ColourScale
from plotting.animator import HexAnimator
from tissue_sampling.sim_sampling import get_vafs_for_all_biopsies, biopsy_sample
from analysis.analysis import incomplete_moment_vaf_fixed_intervals
from collections import namedtuple
import os
import random
import seaborn as sns
import warnings
from scipy.sparse import SparseEfficiencyWarning
from .utilities import (
    compare_to_old_results,
    next_ax,
    INITIAL_CELLS,
    MAX_TIME,
    DIVISION_RATE,
    PLOT_DIR,
    SPATIAL_ALGS,
    WF_ALGS,
    get_plots
)
warnings.simplefilter('ignore',SparseEfficiencyWarning)


@pytest.fixture(scope='session')
def axes():
    return get_plots()


def test_simple(axes, algorithm, overwrite_results=False):
    # Does it run with simplest settings
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='simple_run', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Basic run')


def test_multiple_clones(axes, algorithm, overwrite_results=False):
    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    #  Multiple clones
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, fitness_array=[1, 1.1, 0.9],
                   max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiple_clones', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multiple clones')


def test_mutations(axes, algorithm, overwrite_results=False):
    # All the ways of adding mutations
    # Too many tests to do all combinations
    # Make sure every individual option is checked at least.
    # Any changes to the random process will make a large change here.
    # Not good for comparing algorithms, just for checking it runs.
    mutation_rate = 0.1

    # Need lower fitnesses for the branching process - numbers are not equivalent in the different algorithms
    if algorithm == 'Branching':
        norm_mean = 1.1
    else:
        norm_mean = 4.5
    genes = [Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene('mild_driver', mutation_distribution=FixedValue(1.1), synonymous_proportion=0.5),
             Gene('random_driver', mutation_distribution=NormalDist(mean=norm_mean, var=0.05),
                  synonymous_proportion=0.8),
             Gene('uniform_driver', mutation_distribution=UniformDist(low=0.7, high=1.2),
                  synonymous_proportion=0.2),
             Gene('exp_driver', mutation_distribution=ExponentialDist(mean=1.3, offset=0.8),
                  synonymous_proportion=0.2)]
    mut_gen1 = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='add')

    #  Simple
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen1)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='simple_mutations', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Adding mutations')

    # High mutation rate
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=3,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=0.8,
                   mutation_generator=mut_gen1)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='high_mutation_rate', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('High mutation rate')

    # Variable mutation rate
    mutation_rates = np.array([[0, 0.01], [3.5, 0.4]])
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rates,
                   mutation_generator=mut_gen1)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='variable_mutation_rate', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Variable mutation rate')

    #  Multi-gene array
    mut_gen = MutationGenerator(multi_gene_array=True, genes=genes, combine_mutations='add')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multi_gene_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multi-gene')

    #  Multiply fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='multiply')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiply_fitness', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Multiply fitness')

    #  Replace fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='replace')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='replace_fitness', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Replace fitness')

    #  max fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='max')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='max', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('max')

    # min fitness
    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='min')
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
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
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
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
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
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
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
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
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='min_array', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Min array')

    #  Logistic
    if algorithm == 'Branching':
        h = 1.5
    else:
        h = 5
    genes = [Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene('uniform_driver', mutation_distribution=UniformDist(low=0.95, high=h),
                  synonymous_proportion=0.2)]

    mut_gen = MutationGenerator(multi_gene_array=False, genes=genes, combine_array='multiply',
                                mutation_combination_class=BoundedLogisticFitness(1.1, 10),
                                combine_mutations='multiply')
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=20,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=0.5,
                   mutation_generator=mut_gen)
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
    p = Parameters(algorithm=algorithm, initial_cells=INITIAL_CELLS, max_time=20,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=0.5,
                   mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='logistic_multi', overwrite_results=overwrite_results)
    ax = next_ax(axes, algorithm)
    sim.plot_average_fitness_over_time(ax=ax)
    ax.set_title('Logistic - multi gene')


def test_neutral_hallmarks(axes, algorithm, overwrite_results=False):
    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.arange(100 ** 2, dtype=int).reshape(100, 100)
    else:
        initial_size_array = np.ones(100 ** 2)
        initial_grid = None

    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, max_time=MAX_TIME,
                   print_warnings=False, division_rate=DIVISION_RATE)
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


def test_imbalance(axes, algorithm, overwrite_results=False):
    # Want large population so almost deterministic
    if algorithm in SPATIAL_ALGS:
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
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, max_time=MAX_TIME, fitness_array=fitness_array,
                   print_warnings=False, division_rate=DIVISION_RATE)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='non_neutral_run', overwrite_results=overwrite_results)

    ax = next_ax(axes, algorithm)
    ax.plot(sim.times, np.squeeze(sim.population_array[1].toarray()))
    # Add some extra lines to make it easier to compare plots with new versions
    for m in np.linspace(0.1, 1, 5):
        ax.plot(sim.times, 1000 + sim.times * m * 1000)
    ax.set_title('Mean - imbalance')


def test_b_cells(axes, algorithm, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if algorithm in {'Moran', 'Moran2D', 'Branching'}:
        if algorithm in SPATIAL_ALGS:
            initial_size_array = None
            initial_grid = np.arange(30 ** 2).reshape((30, 30))
        else:
            initial_size_array = np.ones(30 ** 2)
            initial_grid = None
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, max_time=MAX_TIME,
                       division_rate=DIVISION_RATE,
                       print_warnings=False, r=0.2, gamma=2.1)
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells', overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='Full clone', ax=ax)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=False, legend_label='A cells', ax=ax)
        ax.set_title('B cells')

        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, max_time=MAX_TIME,
                       division_rate=DIVISION_RATE, mutation_rates=0.1,
                       print_warnings=False, r=0.2, gamma=2.1)
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells_with_mutation', overwrite_results=overwrite_results)
    else:
        ax.set_title('Not implemented')


def test_treatment_with_fixed_clones(axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 0.5, 1.3], [0.4, 1.5, 0.7]]

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=12, initial_size_array=initial_size_array,
                   initial_grid=initial_grid,
                   fitness_array=[1.05, 1, 0.9], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays, print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_clones', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: clones')


def test_treatment_replace_with_fixed_clones(axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 1, 1], [0.4, 1.5, 0.7]]

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2], 60 * 20).reshape(60, 60)
    else:
        initial_size_array = [400, 500, 600]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)

    p = Parameters(algorithm=algorithm, max_time=12, initial_size_array=initial_size_array,
                   initial_grid=initial_grid,
                   fitness_array=[1.05, 1, 0.9], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays,
                   treatment_replace_fitness=True,
                   print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_clones_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: clones')


def test_treatment_with_multiple_genes(axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 0.5, 1.3], [0.4, 1.25, 0.7]]

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1], 50 * 25).reshape(50, 50)
    else:
        initial_size_array = [400, 500]
        initial_grid = None

    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene('driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=12, initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays, print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: genes')

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=12, initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays, mutation_rates=0.005,
                   print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_muts', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment: genes+muts')


def test_treatment_replace_with_multiple_genes(axes, algorithm, overwrite_results=False):
    timings = [3, 7]
    treatment_arrays = [[1, 1, 1], [0.4, 1.25, 0.7]]

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1], 50 * 25).reshape(50, 50)
    else:
        initial_size_array = [400, 500]
        initial_grid = None

    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene('driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=12, initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays,
                   treatment_replace_fitness=True, print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: genes')

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=12, initial_mutant_gene_array=[0, 1], mutation_generator=mut_gen,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1], division_rate=DIVISION_RATE,
                   treatment_timings=timings, treatment_effects=treatment_arrays, mutation_rates=0.005,
                   print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='treatment_genes_muts_replace', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Treatment replace: genes+muts')


def test_labels(cs_label, axes, algorithm, overwrite_results=False):
    mut_gen = MutationGenerator(multi_gene_array=True,
                                genes=[
                                    Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
                                    Gene('driver', mutation_distribution=FixedValue(1.1),
                                         synonymous_proportion=0.5)],
                                )

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 2, 3], 40 * 10).reshape(40, 40)
    else:
        initial_size_array = [400, 500, 600, 700]
        initial_grid = None

    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, max_time=MAX_TIME, colourscales=cs_label,
                   mutation_generator=mut_gen, print_warnings=False,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1, 0.9, 1.02], label_array=[0, 1, 2, 1], mutation_rates=0.01)
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
    p = Parameters(algorithm=algorithm, max_time=MAX_TIME, colourscales=cs_label,
                   mutation_generator=mut_gen, print_warnings=False,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   fitness_array=[1.05, 1, 0.9, 1.02], label_array=[0, 1, 0, 1],
                   label_times=label_times, label_frequencies=label_frequencies, label_values=label_values)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='late_labels', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)
    ax.set_title('Late labels')


def test_incomplete_moments(axes, algorithm, overwrite_results=False):
    initial_cells = 10000
    max_time = 10

    neut_genes = [Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5)]
    non_neut_genes = [
        Gene('neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
        Gene('mild_driver', mutation_distribution=NormalDist(mean=1.2, var=0.02),
             synonymous_proportion=0.6)]
    mut_gen_neut = MutationGenerator(multi_gene_array=False, genes=neut_genes, combine_mutations='add')
    mut_gen_non_neut = MutationGenerator(multi_gene_array=False, genes=non_neut_genes, combine_mutations='add')

    mutation_rate = 0.05

    # Neutral
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=initial_cells, max_time=max_time,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen_neut)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='neutral_incom', overwrite_results=overwrite_results)
    sim.plot_incomplete_moment(ax=ax)
    ax.set_title('Neutral')
    ax = next_ax(axes, algorithm)
    sim.plot_dnds(ax=ax, gene='neutral', legend_label='neutral')
    ax.legend()
    ax.set_title('Neutral dN/dS')

    # Non-Neutral
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_cells=initial_cells, max_time=max_time,
                   print_warnings=False, division_rate=DIVISION_RATE, mutation_rates=mutation_rate,
                   mutation_generator=mut_gen_non_neut)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='non_neutral_incom', overwrite_results=overwrite_results)
    sim.plot_incomplete_moment(ax=ax)
    ax.set_title('Non-Neutral')
    ax = next_ax(axes, algorithm)
    sim.plot_dnds(ax=ax, gene='neutral', legend_label='neutral')
    sim.plot_dnds(ax=ax, gene='mild_driver', legend_label='driver')
    ax.legend()
    ax.set_title('Non-Neutral dN/dS')


def test_neighbours(axes, algorithm):
    # If 2D, plot some neighbourhoods. Include some edge cases.
    if algorithm in SPATIAL_ALGS:
        for neigh in [True, False]:
            if neigh:
                s = ' 7'
            else:
                s = ''
            ax = next_ax(axes, algorithm)
            initial_grid = np.tile([0, 1, 2, 3, 4, 5], 10).reshape(6, 10)
            np.random.seed(0)
            p = Parameters(algorithm=algorithm, max_time=1, print_warnings=False,
                           initial_grid=initial_grid, cell_in_own_neighbourhood=neigh)
            sim = p.get_simulator()
            sim.run_sim()
            sim.change_sparse_to_csr()
            np.random.seed(0)
            h = HexAnimator(sim, equal_aspect=True)
            grid = np.zeros(60)
            coords = np.array([0, 3, 17, 35, 39])
            for i, c in enumerate(coords):
                grid[sim.neighbour_map[c]] = i + 1
            h.plot_grid(grid.reshape((6, 10)), ax=ax)
            ax.set_title('Grid neighbours' + s)
            ax = next_ax(axes, algorithm)
            grid = np.zeros(60)
            coords = coords + 1
            for i, c in enumerate(coords):
                grid[sim.neighbour_map[c]] = i + 1
            h.plot_grid(grid.reshape((6, 10)), ax=ax)
            ax.set_title('Grid neighbours2' + s)
            ax = next_ax(axes, algorithm)
            grid = np.zeros(60)
            coords = coords + 9
            for i, c in enumerate(coords):
                grid[sim.neighbour_map[c]] = i + 1
            h.plot_grid(grid.reshape((6, 10)), ax=ax)
            ax.set_title('Grid neighbours3' + s)
            ax = next_ax(axes, algorithm)
            grid = np.zeros(60)
            coords = coords + 10
            for i, c in enumerate(coords):
                grid[sim.neighbour_map[c]] = i + 1
            h.plot_grid(grid.reshape((6, 10)), ax=ax)
            ax.set_title('Grid neighbours4' + s)
    else:
        ax = next_ax(axes, algorithm)
        ax.set_title('Not implemented')
        ax = next_ax(axes, algorithm)
        ax.set_title('Not implemented')
        ax = next_ax(axes, algorithm)
        ax.set_title('Not implemented')
        ax = next_ax(axes, algorithm)
        ax.set_title('Not implemented')


def test_random_sampling(axes, algorithm, overwrite_results=False):
    # Biopsies and sequencing for the 2D algorithms
    ax = next_ax(axes, algorithm)
    if algorithm in SPATIAL_ALGS:
        non_neut_genes = [Gene('driver1', mutation_distribution=NormalDist(mean=1.3, var=0.1),
                               synonymous_proportion=0.8),
                          Gene('driver2', mutation_distribution=NormalDist(mean=2, var=0.1),
                               synonymous_proportion=0.8)]
        mut_gen = MutationGenerator(multi_gene_array=False, genes=non_neut_genes,
                                    combine_mutations='max')
        biopsies = [
            {'biopsy_origin': (0, 0), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (0, 10), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (0, 20), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (0, 30), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (0, 40), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (0, 50), 'biopsy_shape': (50, 10)},
            {'biopsy_origin': (30, 32), 'biopsy_shape': (10, 15)},
            {'biopsy_origin': (72, 64), 'biopsy_shape': (1, 20)}
        ]
        coverage = 300
        detection_limit = 20
        np.random.seed(0)
        p = Parameters(algorithm=algorithm, initial_cells=10000, max_time=70,
                       print_warnings=False, division_rate=DIVISION_RATE,
                       mutation_rates=0.05, mutation_generator=mut_gen)
        sim = p.get_simulator()
        sim.run_sim()

        raw_vafs = biopsy_sample(sim.grid_results[-1], sim, biopsies[0])
        compare_to_old_results(algorithm, raw_vafs, test_name='biopsy_sample', result_type='object',
                               overwrite_results=overwrite_results)

        vafs = get_vafs_for_all_biopsies(sim, biopsies, coverage=coverage, detection_limit=detection_limit,
                                         merge_clones=False,
                                         binom=False, binom_params=(None, None))

        compare_to_old_results(algorithm, vafs, test_name='random_sampling', result_type='object',
                               overwrite_results=overwrite_results)

        vafs_merge = get_vafs_for_all_biopsies(sim, biopsies, coverage=coverage, detection_limit=detection_limit,
                                               merge_clones=True,
                                               binom=True, binom_params=(33, 0.1))
        compare_to_old_results(algorithm, vafs_merge, test_name='random_sampling_merge', result_type='object',
                               overwrite_results=overwrite_results)

        vafs_full = get_vafs_for_all_biopsies(sim, biopsies, coverage=10000, detection_limit=0,
                                              merge_clones=True,
                                              binom=False, binom_params=(None, None))

        x1, incom_1 = incomplete_moment_vaf_fixed_intervals(vafs['vaf'], interval=0.02)
        x2, incom_2 = incomplete_moment_vaf_fixed_intervals(vafs_merge['vaf'], interval=0.02)
        x3, incom_3 = incomplete_moment_vaf_fixed_intervals(vafs_full['vaf'], interval=0.02)

        ax.plot(x1, incom_1)
        ax.plot(x2, incom_2)
        ax.plot(x3, incom_3)
        ax.set_xlim(left=0)
        ax.set_yscale('log')
        ax.set_title('LFIM sampling')

        ax = next_ax(axes, algorithm)
        vafs['a'] = 1
        vafs_merge['a'] = 2

        df = pd.concat([vafs, vafs_merge], sort=True)
        sns.swarmplot(data=df, x='a', y='vaf', hue='gene', ax=ax, order=[1, 2],
                      hue_order=['driver1', 'driver2'])
        ax.get_legend().set_visible(False)
        ax.set_title('Sampling')
    else:
        ax.set_title('Not implemented')
        ax = next_ax(axes, algorithm)
        ax.set_title('Not implemented')


def test_post_processing(axes, algorithm, overwrite_results=False):
    # Generate as many of the post processing results as possible, like the mutant clone array etc.
    # Run all plotting functions
    # Not big enough simulation to compare results visually. Just checks things can run.
    non_neut_genes = [Gene('driver1', mutation_distribution=FixedValue(1.3),
                           synonymous_proportion=0.8),
                      Gene('driver2', mutation_distribution=FixedValue(1.1),
                           synonymous_proportion=0.8)]
    mut_gen = MutationGenerator(multi_gene_array=True, genes=non_neut_genes, combine_array='max',
                                combine_mutations='max')

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], 100).reshape(20, 50)
    else:
        initial_size_array = [100, 200, 300, 400]
        initial_grid = None

    label_array = [0, 1, 2, 3]

    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, label_array=label_array, max_time=5,
                   print_warnings=False, division_rate=DIVISION_RATE,
                   mutation_rates=0.05, mutation_generator=mut_gen)
    sim = p.get_simulator()
    sim.run_sim()
    ax = next_ax(axes, algorithm)
    sim.muller_plot(allow_y_extension=True, min_size=15, ax=ax)
    ax.set_title('Plot min size')

    # All mutant clone size distributions
    ax = next_ax(axes, algorithm)
    csd = sim.get_mutant_clone_size_distribution()
    csd2 = sim.get_mutant_clone_size_distribution(t=2)
    csd_ns = sim.get_mutant_clone_size_distribution(selection='ns')
    csd_s = sim.get_mutant_clone_size_distribution(selection='s')
    csd_gene = sim.get_mutant_clone_size_distribution(gene_mutated='driver1')
    ax.scatter(range(1, len(csd)), csd[1:], s=5, label='Full')
    ax.scatter(range(1, len(csd2)), csd2[1:], s=5, label='t=2')
    ax.scatter(range(1, len(csd_ns)), csd_ns[1:], s=5, label='NS')
    ax.scatter(range(1, len(csd_s)), csd_s[1:], s=5, label='S')
    ax.scatter(range(1, len(csd_gene)), csd_gene[1:], s=5, label='gene')
    ax.set_title('Clone size dists')
    ax.set_xlim([1, 10])
    ax.legend()

    ax = next_ax(axes, algorithm)
    sim.plot_overall_population(legend_label='all', ax=ax)
    sim.plot_overall_population(label=0, legend_label='0', ax=ax)
    sim.plot_overall_population(label=1, legend_label='1', ax=ax)
    sim.plot_overall_population(label=2, legend_label='2', ax=ax)
    sim.plot_overall_population(label=3, legend_label='3', ax=ax)
    ax.legend()
    ax.set_title('Populations')

    compare_to_old_results(algorithm, sim, test_name='post_process', overwrite_results=overwrite_results)

    res_dict = {}
    res_dict['labeled_pop1'] = sim.get_labeled_population(label=1)
    res_dict['labeled_pop2'] = sim.get_labeled_population(label=2)
    res_dict['mean_mutant_clone_size'] = sim.get_mean_clone_size()
    res_dict['mean_driver1_clone_size'] = sim.get_mean_clone_size(gene_mutated="driver1")
    res_dict['mean_clone_sizes_syn_nonsyn'] = sim.get_mean_clone_sizes_syn_and_non_syn()
    res_dict['get_average_fitness'] = sim.get_average_fitness()
    compare_to_old_results(algorithm, res_dict, test_name='post_process_dict', result_type='dict',
                           overwrite_results=overwrite_results)


def test_post_processing_non_mutation(axes, algorithm, overwrite_results=False):
    # Generate as many of the post processing results as possible, like the mutant clone array etc.
    # Run all plotting functions

    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.tile([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], 100).reshape(20, 50)
    else:
        initial_size_array = [100, 200, 300, 400]
        initial_grid = None

    label_array = [0, 1, 1, 2]
    fitness_array = [0.9, 1, 1.1, 1.2]

    res_dict = {}

    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, fitness_array=fitness_array,
                   label_array=label_array, max_time=5,
                   print_warnings=False, division_rate=DIVISION_RATE)
    sim = p.get_simulator()
    sim.run_sim()
    res_dict['clone_sizes_array1'] = sim.get_clone_sizes_array_for_non_mutation()
    res_dict['clone_sizes_array2'] = sim.get_clone_sizes_array_for_non_mutation(t=3, index_given=False,
                                                                                exclude_zeros=False)
    res_dict['clone_sizes_array3'] = sim.get_clone_sizes_array_for_non_mutation(t=3, index_given=True,
                                                                                exclude_zeros=True, label=1)

    res_dict['clone_size_dist1'] = sim.get_clone_size_distribution_for_non_mutation()
    res_dict['clone_size_dist2'] = sim.get_clone_size_distribution_for_non_mutation(t=3, index_given=False,
                                                                                    exclude_zeros=True)
    res_dict['clone_size_dist3'] = sim.get_clone_size_distribution_for_non_mutation(t=3, index_given=True,
                                                                                    exclude_zeros=False, label=1)

    res_dict['surviving_clones1'] = sim.get_surviving_clones_for_non_mutation()
    res_dict['surviving_clones2'] = sim.get_surviving_clones_for_non_mutation(label=1)
    res_dict['surviving_clones3'] = sim.get_surviving_clones_for_non_mutation(times=[1, 2, 3, 3.1, 3.2, 4], label=1)

    res_dict['labeled_pop1'] = sim.get_labeled_population(label=1)
    res_dict['labeled_pop2'] = sim.get_labeled_population(label=2)

    res_dict['get_average_fitness'] = sim.get_average_fitness()

    compare_to_old_results(algorithm, res_dict, test_name='post_process_non_mutation_dict', result_type='dict',
                           overwrite_results=overwrite_results)


def test_irregular_sampling(axes, algorithm, overwrite_results=False):
    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.arange(100 ** 2, dtype=int).reshape(100, 100)
    else:
        initial_size_array = np.ones(100 ** 2)
        initial_grid = None
    ax = next_ax(axes, algorithm)
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, times=[1.1, 2, 3.4, 12],
                   print_warnings=False, division_rate=DIVISION_RATE)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='irregular_samples', overwrite_results=overwrite_results)
    sim.plot_mean_clone_size_graph_for_non_mutation(ax=ax)
    ax.set_title('Irregular samples')


def test_partially_simulating_B_cells(axes, algorithm, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if algorithm in {'Moran', 'Moran2D', 'Branching'}:
        if algorithm in SPATIAL_ALGS:
            initial_size_array = None
            initial_grid = np.arange(30 ** 2).reshape((30, 30))
        else:
            initial_size_array = np.ones(30 ** 2)
            initial_grid = None
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, times=[1.1, 2, 3.4, 12],
                       division_rate=DIVISION_RATE,
                       print_warnings=False, r=0.2, gamma=2.1)
        sim = p.get_simulator()
        sim.run_sim()
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=False, legend_label='A cells', ax=ax)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='Full clone', ax=ax)
        ax.set_title('B cells - partial sim')
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, times=[1.1, 2, 3.4, 12],
                       division_rate=DIVISION_RATE,
                       print_warnings=False, r=0.2, gamma=2.1,
                       stratification_sim_percentile=0.9
                       )
        sim = p.get_simulator()
        sim.run_sim()
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='0.9', ax=ax)

        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, times=[1.1, 2, 3.4, 12],
                       division_rate=DIVISION_RATE,
                       print_warnings=False, r=0.2, gamma=2.1,
                       stratification_sim_percentile=0.5
                       )
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='B_cells_partial', overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, legend_label='0.5', ax=ax)
        ax.legend()
    else:
        ax.set_title('Not implemented')


def test_too_many_sample_points(axes, algorithm, overwrite_results=False):
    # If there are more sample points than divisions.
    # Need to reduce number of points.
    ax = next_ax(axes, algorithm)
    ax.set_title('Reduce samples')
    if algorithm in WF_ALGS:
        div_rate = 0.1
    else:
        div_rate = 0.001

    TIMES = [1.5, 3.0, 6.0, 12.0, 24.0, 52.0, 78.0]
    grid_edge = 20
    if algorithm in SPATIAL_ALGS:
        initial_size_array = None
        initial_grid = np.arange(grid_edge ** 2).reshape((grid_edge, grid_edge))
    else:
        initial_size_array = np.ones(grid_edge ** 2)
        initial_grid = None

    np.random.seed(0)
    p = Parameters(algorithm=algorithm, times=TIMES, division_rate=div_rate,
                   initial_size_array=initial_size_array,
                   initial_grid=initial_grid, print_warnings=False,)

    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='reduce_sample_points', overwrite_results=overwrite_results)
    sim.plot_mean_clone_size_graph_for_non_mutation(ax=ax)

    ax = next_ax(axes, algorithm)
    if algorithm in {'Moran', 'Moran2D', 'Branching'}:
        ax.set_title('Reduce samples - B')
        np.random.seed(0)
        set_gsl_random_seed(0)
        p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                       initial_grid=initial_grid, times=TIMES,
                       division_rate=div_rate,
                       print_warnings=False, r=0.1, gamma=0.2)
        sim = p.get_simulator()
        sim.run_sim()
        compare_to_old_results(algorithm, sim, test_name='reduce_sample_points_B_cells',
                               overwrite_results=overwrite_results)
        sim.plot_mean_clone_size_graph_for_non_mutation(include_diff_cells=True, ax=ax)
    else:
        ax.set_title('Not implemented')


def _induction_of_label_and_mutant(axes, algorithm, overwrite_results=False):
    genes = [Gene('Neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene('Notch1', mutation_distribution=FixedValue(4), synonymous_proportion=0)]
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

    if algorithm in SPATIAL_ALGS:
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
    if algorithm == "Branching":
        label_fitness = 1.2
    else:
        label_fitness = 4
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, max_time=10,
                   division_rate=DIVISION_RATE,
                   mutation_generator=mutation_generator,
                   label_times=label_time, label_values=label_value, label_frequencies=label_freq,
                   label_fitness=label_fitness, colourscales=green_clones,
                   print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='single_label_with_mutant', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)

    # Multiple labelling events
    ax = next_ax(axes, algorithm)
    label_time = [3, 6]
    label_value = [1, 2]
    label_freq = [0.05, 0.1]
    if algorithm == "Branching":
        label_fitness = [1.15, 1.25]
    else:
        label_fitness = [4, 8]
    label_genes = [-1, 1]  # Apply the mutants to different genes
    np.random.seed(0)
    p = Parameters(algorithm=algorithm, initial_size_array=initial_size_array,
                   initial_grid=initial_grid, max_time=10,
                   division_rate=DIVISION_RATE,
                   mutation_generator=mutation_generator,
                   label_times=label_time, label_values=label_value, label_frequencies=label_freq,
                   label_fitness=label_fitness, label_genes=label_genes, colourscales=green_clones,
                   print_warnings=False)
    sim = p.get_simulator()
    sim.run_sim()
    compare_to_old_results(algorithm, sim, test_name='multiple_label_with_mutant', overwrite_results=overwrite_results)
    sim.muller_plot(quick=True, allow_y_extension=True, ax=ax)


def test_seven_cell_neighbourhood(cs_label, axes, algorithm, overwrite_results=False):
    ax = next_ax(axes, algorithm)
    if algorithm in SPATIAL_ALGS:
        np.random.seed(0)
        initial_grid = np.arange(30 ** 2).reshape((30, 30))
        p = Parameters(algorithm=algorithm, max_time=MAX_TIME, colourscales=cs_label,
                       print_warnings=False, initial_grid=initial_grid, cell_in_own_neighbourhood=True)
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


def test_animation(algorithm):
    # Make animation
    #  just checking it runs. Can be compared visually or using the output files
    # Have to do after all other plots as this will reset the figures
    non_neut_genes = [Gene('mild_driver', mutation_distribution=NormalDist(mean=1.1, var=0.1),
                           synonymous_proportion=0.85)]
    mut_gen = MutationGenerator(multi_gene_array=False, genes=non_neut_genes, combine_mutations='add')

    if algorithm in SPATIAL_ALGS:
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
    p = Parameters(algorithm=algorithm, max_time=MAX_TIME,
                   mutation_generator=mut_gen, print_warnings=False,
                   initial_size_array=initial_size_array, initial_grid=initial_grid,
                   mutation_rates=mutation_rate)
    sim = p.get_simulator()
    sim.run_sim()
    sim.animate(os.path.join(PLOT_DIR, 'test_ani_{}.mp4'.format(algorithm)), grid_size=grid_size)


