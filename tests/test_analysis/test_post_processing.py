"""
Tests for functions which extract or process data from simulations.
Some of these are used to create plots (plots themselves tested elsewhere)
"""
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix

from src.clone_competition_simulation.parameters import (
    Parameters,
    Algorithm,
    PopulationParameters,
    TimeParameters,
    FitnessParameters,
    LabelParameters,
)
from src.clone_competition_simulation.fitness import MutationGenerator, Gene, NormalDist, FixedValue, EpistaticEffect


@pytest.fixture
def non_mutating_sim(monkeypatch):
    with monkeypatch.context() as m:
        rng = np.random.RandomState()
        monkeypatch.setattr('numpy.random', rng)
        np.random.seed(0)
        p = Parameters(
            algorithm=Algorithm.WF2D,
            population=PopulationParameters(initial_grid=np.arange(100).reshape((10, 10)),
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(max_time=10, division_rate=1, samples=7),
            labels=LabelParameters(label_array=np.arange(100)),
        )
        sim = p.get_simulator()
        sim.run_sim()
    return sim


@pytest.fixture
def mutating_sim(monkeypatch):
    with monkeypatch.context() as m:
        rng = np.random.RandomState()
        monkeypatch.setattr('numpy.random', rng)
        np.random.seed(0)
        p = Parameters(
            algorithm=Algorithm.WF2D,
            population=PopulationParameters(initial_cells=16,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(max_time=10, division_rate=1),
            fitness=FitnessParameters(
                mutation_generator=MutationGenerator(
                    genes=[Gene(name="A",
                                mutation_distribution=NormalDist(mean=1, var=0.1),
                                synonymous_proportion=0.5)],
                ),
                mutation_rates=0.1
            )
        )
        sim = p.get_simulator()
        sim.run_sim()
    return sim


@pytest.fixture
def mutating_sim2(monkeypatch):
    """With multi_gene_array=True"""
    with monkeypatch.context() as m:
        rng = np.random.RandomState()
        monkeypatch.setattr('numpy.random', rng)
        np.random.seed(0)
        p = Parameters(
            algorithm=Algorithm.WF2D,
            population=PopulationParameters(initial_cells=16,
                                            cell_in_own_neighbourhood=False),
            times=TimeParameters(max_time=10, division_rate=1),
            fitness=FitnessParameters(
                mutation_generator=MutationGenerator(
                    genes=[Gene(name="A",
                                mutation_distribution=NormalDist(mean=1, var=0.1),
                                synonymous_proportion=0.5)],
                    multi_gene_array=True
                ),
                mutation_rates=0.1,
            )
        )
        sim = p.get_simulator()
        sim.run_sim()
    return sim


def test_view_clone_info(non_mutating_sim):
    expected = pd.DataFrame(
        {
            "clone id": np.arange(100),
            "label": np.arange(100),
            "fitness": np.ones(100, dtype=float),
            "generation born": np.zeros(100, dtype=int),
            "parent clone id": np.full(100, -1)
        }
    )
    df = non_mutating_sim.view_clone_info()
    pd.testing.assert_frame_equal(df, expected)


def test_view_clone_info2(mutating_sim):
    expected = pd.DataFrame(
        {
            "clone id": np.arange(22),
            "label": np.zeros(22, dtype=int),
            "fitness": np.array([1., 0.98872011, 1.09073459, 1.08152699, 0.98872011, 1.02290979, 1.,
                                 1., 1., 1.04283036, 1.09962778, 1., 1.2625391, 1.16592766, 1.00001396,
                                 1.15058384, 1.09962778, 1., 1., 1.2625391, 1.23700854, 1.25441366]),
            "generation born": np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 9, 10, 10, 10]),
            "parent clone id": np.array([-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6, 0, 10, 0, 0, 8, 10, 0, 0, 12, 13, 13]),
            "last gene mutated": [None] + ["A"] * 21
        }
    )
    df = mutating_sim.view_clone_info(include_raw_fitness=False)
    pd.testing.assert_frame_equal(df, expected)

    df2 = mutating_sim.view_clone_info(include_raw_fitness=True)
    expected['A'] = expected['fitness']
    pd.testing.assert_frame_equal(df2, expected)


def test_view_clone_info3(mutating_sim2):
    expected = pd.DataFrame(
        {
            "clone id": np.arange(22),
            "label": np.zeros(22, dtype=int),
            "fitness": np.array([1., 0.98872011, 1.09073459, 1.08152699, 0.98872011, 1.02290979, 1.,
                                 1., 1., 1.04283036, 1.09962778, 1., 1.2625391, 1.16592766, 1.00001396,
                                 1.15058384, 1.09962778, 1., 1., 1.2625391, 1.23700854, 1.25441366]),
            "generation born": np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 9, 10, 10, 10]),
            "parent clone id": np.array([-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6, 0, 10, 0, 0, 8, 10, 0, 0, 12, 13, 13]),
            "last gene mutated": [None] + ["A"] * 21
        }
    )
    df = mutating_sim2.view_clone_info(include_raw_fitness=False)
    pd.testing.assert_frame_equal(df, expected)

    df2 = mutating_sim2.view_clone_info(include_raw_fitness=True)
    expected['Initial clone fitness'] = np.ones(22)
    expected['A'] = expected['fitness']
    expected.loc[expected['A'] == 1, 'A'] = np.nan
    pd.testing.assert_frame_equal(df2, expected)


def test_view_clone_info4(monkeypatch):
    expected = pd.DataFrame(
        {
            "clone id": np.arange(7),
            "label": np.zeros(7, dtype=int),
            "fitness": [1, 1.1, 1.1, 1.05, 3, 3, 3],
            "generation born": [0, 8, 13, 18, 25, 35, 48],
            "parent clone id": [-1, 0, 0, 0, 2, 4, 4],
            "last gene mutated": [None, "Gene1", "Gene1", "Gene2", "Gene2", "Gene2", "Gene1"],
            "Initial clone fitness": [1.] * 7,
            "Gene1": [np.nan, 1.1, 1.1, np.nan, 1.1, 1.1, 1.1],
            "Gene2": [np.nan, np.nan, np.nan, 1.05, 1.05, 1.05, 1.05],
            "Epi1": [np.nan, np.nan, np.nan, np.nan, 3., 3., 3.]
        }
    )

    with monkeypatch.context() as m:
        rng = np.random.RandomState()
        monkeypatch.setattr('numpy.random', rng)
        np.random.seed(0)
        mut_gen = MutationGenerator(
            genes=[
                Gene(name='Gene1', mutation_distribution=FixedValue(1.1), synonymous_proportion=0),
                Gene(name='Gene2', mutation_distribution=FixedValue(1.05), synonymous_proportion=0)
            ],
            epistatics=[
                EpistaticEffect(
                    name='Epi1',
                    gene_names=['Gene1', 'Gene2'],
                    fitness_distribution=FixedValue(3)
                )
            ],
            multi_gene_array=True,
            combine_mutations='replace',  # With FixedValue this means further mutations will do nothing
        )
        p = Parameters(
            algorithm='Moran',
            times=TimeParameters(max_time=10, division_rate=1),
            population=PopulationParameters(initial_cells=6),
            fitness=FitnessParameters(
                mutation_rates=0.15,
                mutation_generator=mut_gen,
            )
        )
        s = p.get_simulator()
        s.run_sim()

    df = s.view_clone_info(include_raw_fitness=True)
    pd.testing.assert_frame_equal(df, expected)


def test_change_sparse_to_csr(non_mutating_sim):
    assert non_mutating_sim.is_lil
    assert isinstance(non_mutating_sim.population_array, lil_matrix)
    non_mutating_sim.change_sparse_to_csr()
    assert not non_mutating_sim.is_lil
    assert isinstance(non_mutating_sim.population_array, csr_matrix)


def test_convert_time_to_index(monkeypatch, non_mutating_sim):
    times = np.linspace(0, 10, 27)
    with monkeypatch.context() as m:
        m.setattr(non_mutating_sim, "times", times)
        m.setattr(non_mutating_sim, "_search_times", times)
        assert non_mutating_sim._convert_time_to_index(4.5) == 12
        assert non_mutating_sim._convert_time_to_index(2) == 5
        assert non_mutating_sim._convert_time_to_index(7.00123) == 18
        assert non_mutating_sim._convert_time_to_index(6.9) == 18

        assert non_mutating_sim._convert_time_to_index(4.5, nearest=False) == 11
        assert non_mutating_sim._convert_time_to_index(2, nearest=False) == 5
        assert non_mutating_sim._convert_time_to_index(7.00123, nearest=False) == 18
        assert non_mutating_sim._convert_time_to_index(6.9, nearest=False) == 17


def test_find_nearest(monkeypatch, non_mutating_sim):
    times = np.linspace(0, 10, 27)
    with monkeypatch.context() as m:
        m.setattr(non_mutating_sim, "times", times)
        assert non_mutating_sim._find_nearest(4.5) == 12
        assert non_mutating_sim._find_nearest(2) == 5
        assert non_mutating_sim._find_nearest(7.00123) == 18


@pytest.mark.parametrize("t,index_given,label,exclude_zeros,expected", [
    (None,False,None,False,np.array([
        0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0,
        0, 0, 0, 2, 11, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 3, 2, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 23,
        0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0])),
    (5,True,None,True, np.array([5, 3, 6, 5, 8, 1, 5, 7, 2, 1, 5, 1, 5, 1, 8, 5, 21, 2, 2, 7])),
    (5,False,None,True, np.array([3, 3, 1, 6, 6, 5, 1, 8, 5, 1, 1, 1, 3, 3, 5, 2, 5, 9, 22,
                                  1, 2, 5, 2])),
    (None, False, 80, False, [23])
])
def test_get_clone_sizes_array_for_non_mutation(non_mutating_sim, t, index_given,
                                                label, exclude_zeros, expected):
    clone_size_array = non_mutating_sim.get_clone_sizes_array_for_non_mutation(
        t=t, index_given=index_given, label=label, exclude_zeros=exclude_zeros
    )
    np.testing.assert_array_equal(clone_size_array, expected)


@pytest.mark.parametrize("t,index_given,label,exclude_zeros,expected", [
    (None,False,None,False,np.array([81, 2, 4, 5, 1, 1, 0, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1])),
    (5,True,None,True, np.array([0, 4, 3, 1, 0, 6, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
    (5,False,None,True, np.array([0, 6, 3, 4, 0, 5, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
    (None, False, 80, False, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1]))
])
def test_get_clone_size_distribution_for_non_mutation(non_mutating_sim, t, index_given,
                                                      label, exclude_zeros, expected):
    clone_size_array = non_mutating_sim.get_clone_size_distribution_for_non_mutation(
        t=t, index_given=index_given, label=label, exclude_zeros=exclude_zeros
    )
    np.testing.assert_array_equal(clone_size_array, expected)


@pytest.mark.parametrize("times,label,expected_clones,expected_times", [
    (None, None, np.array([100, 67, 39, 32, 23, 20, 19, 19]), np.array([0., 1., 3., 4., 6., 7., 9., 10.])),
    ([1, 4], None, np.array([67, 32]), np.array([1., 4.])),
    ([1, 4], 80, np.array([1, 1]), np.array([1., 4.]))
])
def test_get_surviving_clones_for_non_mutation(non_mutating_sim, times, label, expected_clones, expected_times):
    surviving_clones, times = non_mutating_sim.get_surviving_clones_for_non_mutation(times, label)
    np.testing.assert_array_equal(surviving_clones, expected_clones)
    np.testing.assert_array_equal(times, expected_times)


def test_get_clone_ancestors(mutating_sim):
    assert mutating_sim.get_clone_ancestors(21) == [21, 13, 0, -1]


def test_get_clone_descendants(mutating_sim):
    assert mutating_sim.get_clone_descendants(13) == [13, 20, 21]


def test_trim_tree(mutating_sim):
    trimmed_tree, sampled_clones = mutating_sim._trim_tree()
    assert trimmed_tree.to_dict() == {'-1': {'children': [{'0': {'children': [{
        '1': {'children': ['4']}}, '11', '13', '3', '5',
        {'6': {'children': [{'10': {'children': [{'12': {'children': ['19']}}]}}]}},
        {'8': {'children': ['15']}}
    ]}}]}}
    assert sampled_clones == [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 19]


def test_get_clone_descendants_trimmed(mutating_sim):
    trimmed_tree, _ = mutating_sim._trim_tree()
    assert mutating_sim._get_clone_descendants_trimmed(trimmed_tree, 1) == [1, 4]
    assert mutating_sim._get_clone_descendants_trimmed(trimmed_tree, 6) == [6, 10, 12, 19]


@pytest.mark.parametrize('selection,expected', [
    ('all', {0: [0, 1, 4, 11, 13, 20, 21, 14, 17, 18, 2, 3, 5, 6, 10, 12, 19, 16, 7, 8, 15, 9],
             1: [1, 4], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 10, 12, 19, 16], 7: [7], 8: [8, 15],
             9: [9], 10: [10, 12, 19, 16], 11: [11], 12: [12, 19], 13: [13, 20, 21], 14: [14],
             15: [15], 16: [16], 17: [17], 18: [18], 19: [19], 20: [20], 21: [21]}),
    ('s', {4: [4], 6: [6, 10, 12, 19, 16], 7: [7], 8: [8, 15], 11: [11], 16: [16], 17: [17], 18: [18], 19: [19]}),
    ('ns', {1: [1, 4], 2: [2], 3: [3], 5: [5], 9: [9], 10: [10, 12, 19, 16], 12: [12, 19], 13: [13, 20, 21],
            14: [14], 15: [15], 20: [20], 21: [21]}),
    ('label', {}),
    ('mutations', {1: [1, 4], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 10, 12, 19, 16], 7: [7], 8: [8, 15], 9: [9],
               10: [10, 12, 19, 16], 11: [11], 12: [12, 19], 13: [13, 20, 21], 14: [14], 15: [15], 16: [16],
               17: [17], 18: [18], 19: [19], 20: [20], 21: [21]}),
    ('non_zero', {0: [0, 1, 4, 11, 13, 3, 5, 6, 10, 12, 19, 8, 15], 1: [1, 4], 3: [3], 4: [4], 5: [5],
                  6: [6, 10, 12, 19], 8: [8, 15], 10: [10, 12, 19], 11: [11], 12: [12, 19], 13: [13],
                  15: [15], 19: [19]})
])
def test_track_mutations(mutating_sim, selection, expected):
    assert mutating_sim.track_mutations(selection) == expected


def test_create_mutant_clone_array(mutating_sim):
    expected = np.array([
        [16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 16.],
        [ 0., 2., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [ 0., 0., 1., 2., 4., 3., 3., 4., 4., 3., 3.],
        [ 0., 0., 0., 1., 1., 2., 2., 2., 3., 5., 5.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 2., 3., 3., 3., 4., 4., 5., 6.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 1., 2., 2., 2., 3., 5., 5.],
        [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 2., 3., 5., 5.],
        [ 0., 0., 0., 0., 0., 0., 0., 2., 2., 3., 2.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 2., 3., 5., 6.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    mutating_sim._create_mutant_clone_array()
    np.testing.assert_array_equal(
        mutating_sim.mutant_clone_array.toarray(),
        expected
    )


def test_get_idx_of_gene_mutated(mutating_sim):
    assert mutating_sim.get_idx_of_gene_mutated("A") == set(range(1,22))


@pytest.mark.parametrize('t,selection,index_given,gene_mutated,non_zero_only,expected', [
    (None, 'all', False, None, False, np.array([0, 0, 0, 0, 3, 5, 0, 6, 0, 5, 0, 5, 2, 0, 6, 0, 0, 0, 2, 0, 0])),
    (None, 'mutations', False, None, False, np.array([0, 0, 0, 0, 3, 5, 0, 6, 0, 5, 0, 5, 2, 0, 6, 0, 0, 0, 2, 0, 0])),
    (None, 'ns', False, None, False, np.array([0, 0, 0, 3, 0, 5, 5, 2, 0, 6, 0, 0])),
    (None, 's', False, None, False, np.array([0, 5, 0, 6, 0, 0, 0, 0, 2])),
    (None, 'label', False, None, False, np.array([])),
    (5, 'ns', False, None, False, np.array([1, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0])),
    (3, 'ns', True, None, False, np.array([1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])),
    (5, 'all', True, "A", False, np.array([1, 0, 0, 1, 3, 2, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    (None, 'all', False, None, True, np.array([3, 5, 6, 5, 5, 2, 6, 2])),
])
def test_get_mutant_clone_sizes(mutating_sim, t, selection, index_given,
                                gene_mutated, non_zero_only, expected):
    mutant_clone_sizes = mutating_sim.get_mutant_clone_sizes(
        t=t, selection=selection, index_given=index_given,
        gene_mutated=gene_mutated, non_zero_only=non_zero_only
    )
    np.testing.assert_array_equal(mutant_clone_sizes, expected)

@pytest.mark.parametrize('t,selection,index_given,gene_mutated,expected', [
    (None, 'mutations', False, None, np.array([13, 0, 2, 1, 0, 3, 2])),
    (None, 'ns', False, None,  np.array([7, 0, 1, 1, 0, 2, 1])),
    (None, 's', False, None,  np.array([6, 0, 1, 0, 0, 1, 1])),
    (5, 'ns', False, None,  np.array([9, 1, 1, 1])),
    (3, 'ns', True, None,  np.array([10, 1, 1])),
    (5, 'mutations', True, "A",  np.array([15, 2, 2, 2])),
])
def test_get_mutant_clone_size_distribution(mutating_sim, t, selection, index_given, gene_mutated, expected):
    csd = mutating_sim.get_mutant_clone_size_distribution(t=t, selection=selection, index_given=index_given,
                                                          gene_mutated=gene_mutated)
    np.testing.assert_array_equal(csd, expected)


@pytest.mark.parametrize('t,min_size,gene,expected', [
    (None, 1, None, 1.6666666666666667),
    (None, 3, None, 2),
    (5, 1, None, 1),
    (None, 1, "A", 1.6666666666666667),
])
def test_get_dnds(mutating_sim, t, min_size, gene, expected):
    dnds = mutating_sim.get_dnds(t=t, min_size=min_size, gene=gene)
    np.testing.assert_almost_equal(dnds, expected)


def test_get_labeled_population1(mutating_sim):
    np.testing.assert_array_equal(mutating_sim.get_labeled_population(), np.full(11, 16))
    np.testing.assert_array_equal(mutating_sim.get_labeled_population(1), np.full(11, 0))


def test_get_labeled_population2(non_mutating_sim):
    np.testing.assert_array_equal(non_mutating_sim.get_labeled_population(), np.full(8, 100))
    np.testing.assert_array_equal(
        non_mutating_sim.get_labeled_population(80),
        np.array([ 1.,  3.,  5., 10., 22., 21., 20., 23.])
    )


@pytest.mark.parametrize('t,selection,index_given,gene_mutated,expected', [
    (None, 'all', False, None, 34/8),
    (None, 'mutations', False, None, 34/8),
    (None, 'ns', False, None, 21/5),
    (None, 's', False, None, 13/3),
    (None, 'label', False, None, np.nan),
    (5, 'ns', False, None, 2),
    (3, 'ns', True, None, 1.5),
    (5, 'all', True, "A", 2),
])
def test_get_mean_clone_size(mutating_sim, t, selection, index_given, gene_mutated, expected):
    np.testing.assert_equal(mutating_sim.get_mean_clone_size(
        t=t, selection=selection, index_given=index_given, gene_mutated=gene_mutated), expected)


@pytest.mark.parametrize('t,index_given,gene_mutated,expected', [
    (5, False, None, (2, 2)),
    (3, True, None, (1.333333333, 1.5)),
    (5, True, "A", (2, 2)),
])
def test_get_mean_clone_sizes_syn_and_non_syn(mutating_sim, t, index_given, gene_mutated, expected):
    np.testing.assert_almost_equal(
        mutating_sim.get_mean_clone_sizes_syn_and_non_syn(t, index_given=index_given, gene_mutated=gene_mutated),
        expected
    )


@pytest.mark.parametrize('t,expected', [
    (1, 1.0087808873405937), (3, 1.0021587310987177), (7.5, 1.1039289571513093), (None, 1.163548951697842)
])
def test_get_average_fitness(mutating_sim, t, expected):
    np.testing.assert_almost_equal(mutating_sim.get_average_fitness(t=t), expected)


def test_expected_incomplete_moment(non_mutating_sim):
    np.testing.assert_almost_equal(
        non_mutating_sim._expected_incomplete_moment(1, 3),
        np.array([0.3678794, 0.1353353, 0.0497871])
    )


def test_absorb_small_clones(mutating_sim):
    mutating_sim.change_sparse_to_csr()
    clones_array, pop_array = mutating_sim._absorb_small_clones(min_size=6)
    expected_clones_array = np.array([
        [0., 0., 1., 0., -1., -1.],
        [8., 0., 1., 3., 0., 0.],
        [15., 0., 1.150584, 7., 8., 0.]
    ])
    expected_pop_array = np.array([
        [16., 16., 16., 14., 13., 13., 13., 12., 12., 11., 10.],
        [0., 0., 0., 2., 3., 3., 3., 2., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 2., 3., 5., 6.]
    ])
    np.testing.assert_array_almost_equal(clones_array, expected_clones_array)
    np.testing.assert_array_equal(pop_array.toarray(), expected_pop_array)


def test_get_children(mutating_sim):
    np.testing.assert_array_equal(mutating_sim._get_children(mutating_sim.clones_array, 8), [15])
