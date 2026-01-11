import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(TEST_DIR))
import numpy as np
from src.clone_competition_simulation.parameters.algorithm_validation import Algorithm
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import os
import warnings
from scipy.sparse import lil_matrix, csr_matrix, SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)
import gzip



INITIAL_CELLS = 12 ** 2
MAX_TIME = 5
DIVISION_RATE = 1.3
SAMPLES = 100
PLOT_DIR = TEST_DIR
FIGSIZE = (17, 15)
RESULTS_STORE=os.path.join(TEST_DIR, "stored_results")
FIGURE_DIMENSIONS = (7, 8)  # Number of rows and columns in output fig


def get_plots():
    axes_dict = dict()
    for algorithm in Algorithm:
        fig, axs = plt.subplots(*FIGURE_DIMENSIONS, figsize=FIGSIZE, squeeze=False)
        axes_dict[algorithm] = (fig, axs)
    return axes_dict

AX_COUNT_DICT = defaultdict(int)
def get_ax_count(algorithm):
    global AX_COUNT_DICT
    AX_COUNT_DICT[algorithm] += 1
    return AX_COUNT_DICT[algorithm] - 1  # So returns 0 the first time


def next_ax(axes, algorithm):
    ax_count = get_ax_count(algorithm)
    ax = axes[algorithm][1][ax_count // FIGURE_DIMENSIONS[1]][ax_count % FIGURE_DIMENSIONS[1]]
    return ax


def convert_sim_to_standard_form(sim, algorithm):
    # Don't want to store the hash of the pickled sim itself, as any change means it won't match.
    # Take the outputs population array, clones array, ns_clones, s_clones, clone tree
    standard_results = dict()
    standard_results['pop'] = sim.population_array
    standard_results['clones'] = sim.clones_array
    standard_results['muts'] = sim.mutant_clone_array
    standard_results['ns_muts'] = sim.ns_muts
    standard_results['s_muts'] = sim.s_muts
    standard_results['label_muts'] = sim.label_muts
    standard_results['raw_fitness_array'] = sim.raw_fitness_array
    standard_results['clone_tree'] = sim.tree.to_dict()
    standard_results['times'] = sim.times
    standard_results['sample_points'] = sim.sample_points
    if algorithm.two_dimensional:
        standard_results['grid_results'] = sim.grid_results

    if hasattr(sim, "diff_cell_population"):
        standard_results['diff_cell_population'] = sim.diff_cell_population
    else:
        standard_results['diff_cell_population'] = None

    return standard_results


def store_new_results(algorithm, name, results):
    # Overwrite the old stored results
    with gzip.open(os.path.join(RESULTS_STORE, algorithm.value, name + '.pickle.gz'), 'wb') as f:
        pickle.dump(results, f, protocol=4)


def get_stored_results(algorithm, name):
    with gzip.open(os.path.join(RESULTS_STORE, algorithm.value, name + '.pickle.gz'), 'rb') as f:
        old_results = pickle.load(f)
    return old_results


def compare_single_res(old, new):
    # Some cases can be compared directly
    # Others cannot
    try:
        if isinstance(old, (dict, set, int, float, np.integer)):
            # This assumes the contents of the dict can be simply compared
            return old == new
        elif isinstance(old, (np.ndarray,)):
            # Need to use the np.testing function as the normal np.array_equal says nan != nan
            # It raises an assertion error rather than returning True/False
            if isinstance(new, (lil_matrix, csr_matrix)):
                new = new.toarray()
            try:
                np.testing.assert_array_almost_equal(old, new)
            except AssertionError:
                return False
            return True
        elif isinstance(old, (lil_matrix, csr_matrix)):
            if isinstance((old != new), bool):
                return (old != new)
            else:
                return (old != new).nnz == 0

        elif old is None:
            return new is None
        elif isinstance(old, (list, tuple)):
            if len(old) != len(new):
                return False
            res = [compare_single_res(old[i], new[i]) for i in range(len(old))]
            return all(res)
        elif isinstance(old, pd.DataFrame):
            return old.equals(new)
        else:
            print('Cannot compare', type(old))
            print(old)
            print(new)
            return False
    except Exception as e:
        print('FAILED COMPARISON')
        print(old)
        print(new)
        print(type(old))
        print(type(new))
        raise e


def compare_to_old_results(algorithm, new_results, test_name, result_type='sim', overwrite_results=False):
    if result_type == 'sim':
        new_results = convert_sim_to_standard_form(new_results, algorithm)
    if overwrite_results:
        store_new_results(algorithm, test_name, new_results)
    old_results = get_stored_results(algorithm, test_name)

    mismatching_results = []
    last_error = None
    if not isinstance(new_results, dict):
        assert compare_single_res(old_results, new_results)
    else:
        for attr_key, attr_result in new_results.items():
            # print('attr', attr_key)
            try:
                assert compare_single_res(attr_result, old_results[attr_key])
            except KeyError as e:
                print('Missing {} for test {}:{}'.format(attr_key, algorithm, test_name))
                raise e
            except AssertionError as e:
                mismatching_results.append(attr_key)
                last_error = e
            except Exception as e:
                print('Failed to compare {} for test {}:{}'.format(attr_key, algorithm, test_name))
                # print(attr_result.toarray())
                # print(old_results[attr_key].toarray())
                raise e

        old_keys = set(old_results.keys())
        new_keys = new_results.keys()
        if old_keys.difference(new_keys):
            assert False, "Missing keys: {}".format(old_keys.difference(new_keys))

    if len(mismatching_results) > 0:
        print('Mismatching keys:', mismatching_results)
        raise last_error  # Just pick one to raise
