import pytest
import numpy as np
from pydantic import ValidationError
from clone_competition_simulation.fitness import (
    MutationGenerator,
    Gene,
    FixedValue,
    NormalDist,
    UniformDist,
    ExponentialDist,
    BoundedLogisticFitness,
    UnboundedFitness,
    EpistaticEffect,
    MutationCombination,
    ArrayCombination
)

@pytest.fixture(scope='module')
def genes():
    genes = [
        Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5, weight=0.5),
        Gene(name='mild_driver', mutation_distribution=FixedValue(1.01), synonymous_proportion=0.5, weight=1.5),
        Gene(name='random_driver', mutation_distribution=NormalDist(mean=1.01, var=0.05),
             synonymous_proportion=0.8),
        Gene(name='uniform_driver', mutation_distribution=UniformDist(low=0.7, high=1.01),
             synonymous_proportion=0.2),
        Gene(name='exp_driver', mutation_distribution=ExponentialDist(mean=1.03, offset=0.8),
             synonymous_proportion=0.2)]
    return genes

@pytest.fixture(scope='module')
def epistatics():
    epistatics = [
        EpistaticEffect(name="E1", gene_names=['mild_driver', 'random_driver'],
                        fitness_distribution=FixedValue(3)),
        EpistaticEffect(name="E2", gene_names=['neutral', 'uniform_driver', 'exp_driver'],
                        fitness_distribution=FixedValue(1))
    ]
    return epistatics

# Test the distributions
DRAW_COUNT = 0
def fake_draw1(mean=None, var=None):
    values = [-1, -2, 99]
    global DRAW_COUNT
    value = values[DRAW_COUNT]
    DRAW_COUNT += 1
    return value


def test_normal_dist_negative(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the normal distribution redraws until it gets a positive value.
    Args:
        monkeypatch:

    Returns:

    """
    n = NormalDist(var=0.1, mean=-10)
    assert n.get_mean() == -10
    global DRAW_COUNT
    DRAW_COUNT = 0
    with monkeypatch.context() as m:
        m.setattr(np.random, "normal", fake_draw1 )
        assert n() == 99


def test_fixed_value():
    f = FixedValue(99)
    assert f.get_mean() == 99
    assert f() == 99


def test_exponential_dist1():
    with pytest.raises(ValueError, match='mean must be greater than offset'):
        ExponentialDist(mean=1, offset=2)


def test_exponential_dist2(monkeypatch):
    global DRAW_COUNT
    DRAW_COUNT = 0
    with monkeypatch.context() as m:
        m.setattr(np.random, "exponential", fake_draw1 )
        exp = ExponentialDist(mean=2, offset=1)
        assert exp.get_mean() == 2
        assert exp() == 0  # First fake draw (-1) plus the offset


def test_uniform_dist1():
    with pytest.raises(ValueError, match='high bound must be higher than the low bound'):
        UniformDist(low=1, high=1)

    with pytest.raises(ValueError, match='high bound must be higher than the low bound'):
        UniformDist(low=1.1, high=1)


def test_uniform_dist2(monkeypatch):
    u = UniformDist(low=1, high=2)
    assert u.get_mean() == 1.5


@pytest.mark.parametrize("x", [0, 1, 10, np.array([0, 1, 10])])
def test_unbounded_fitness(x):
    u = UnboundedFitness()
    np.testing.assert_equal(u.fitness(x), x)
    np.testing.assert_equal(u.inverse(x), x)


@pytest.mark.parametrize("x,expected", [
    (0, 0.990990990990991),
    (1, 1),
    (10, 1.0552472168538105),
    (np.array([0, 1, 10]), np.array([0.990990990990991, 1, 1.0552472168538105]))
])
def test_bounded_logistic_fitness(x, expected):
    u = BoundedLogisticFitness(a=1.1, b=1.1)
    np.testing.assert_almost_equal(u.fitness(x), expected)
    np.testing.assert_almost_equal(u.inverse(expected), x)



def test_gene():
    with pytest.raises(ValidationError) as excinfo:
        Gene(name='neutral', mutation_distribution=FixedValue(0.01), synonymous_proportion=1, weight=-1)
        assert 'weight cannot be below zero' in str(excinfo.value)


def test_gene1():
    with pytest.raises(ValidationError) as excinfo:
        Gene(name='neutral', mutation_distribution=FixedValue(0.01), synonymous_proportion=-1)
        assert 'synonymous_proportion must be between 0 and 1' in str(excinfo.value)

    with pytest.raises(ValidationError)  as excinfo:
        Gene(name='neutral', mutation_distribution=FixedValue(0.01), synonymous_proportion=2)
        assert 'synonymous_proportion must be between 0 and 1' in str(excinfo.value)


@pytest.mark.parametrize("method,expected", [
    (MutationCombination.ADD, np.array([0.6, 4.2])),
    (MutationCombination.MULTIPLY, np.array([0.55, 6.6])),
    (MutationCombination.REPLACE, np.array([0.5, 3])),
    (MutationCombination.MAX, np.array([1.1, 3])),
    (MutationCombination.MIN, np.array([0.5, 2.2]))
])
def test_mutation_combination(method, expected):
    x = np.array([1.1, 2.2])
    y = np.array([0.5, 3])
    np.testing.assert_almost_equal(method.function(x, y), expected)

@pytest.mark.parametrize("method,expected", [
    (ArrayCombination.ADD, [2.3, 3.3]),
    (ArrayCombination.MULTIPLY, [2.42, 1.8]),
    (ArrayCombination.PRIORITY, [1.1, 0.3]),
    (ArrayCombination.MAX, [2.2, 3]),
    (ArrayCombination.MIN, [1.1, 0.3])
])
def test_mutation_combination(method, expected):
    x = np.array(
        [
            [2.2, np.nan, 1.1, np.nan],
            [1, 2, 3, 0.3]
        ]
    )
    np.testing.assert_almost_equal(method.function(x), expected)



def test_mutation_generation_validation(genes, epistatics):
    mut_gen = MutationGenerator(
        genes=genes, epistatics=epistatics,
        combine_mutations=MutationCombination.ADD,
        combine_array=ArrayCombination.MULTIPLY
    )
    assert mut_gen.num_genes == len(genes)
    assert mut_gen.gene_indices == {
        'neutral': 0, 'mild_driver': 1, 'random_driver': 2, "uniform_driver": 3, "exp_driver": 4,
        "E1": 5, "E2": 6
    }
    assert isinstance(mut_gen.mutation_combination_class, UnboundedFitness)
    expected_dists = [FixedValue, FixedValue, NormalDist, UniformDist, ExponentialDist]
    for dist, exp_dist in zip(mut_gen.mutation_distributions, expected_dists):
        assert isinstance(dist, exp_dist)
    np.testing.assert_equal(mut_gen.synonymous_proportion, [0.5, 0.5, 0.8, 0.2, 0.2])
    np.testing.assert_almost_equal(mut_gen.overall_synonymous_proportion,0.44)
    np.testing.assert_almost_equal(mut_gen.relative_weights_cumsum, [0.1, 0.4, 0.6, 0.8, 1.])
    assert mut_gen.epistatics_dict == {
        (1, 2): epistatics[0],
        (0, 3, 4): epistatics[1]
    }
    np.testing.assert_equal(mut_gen.epistatic_cols, np.arange(6, 8))
    assert mut_gen.combine_fitness_function == MutationCombination.ADD.function
    assert mut_gen.combine_array_function == ArrayCombination.MULTIPLY.function


@pytest.fixture
def mut_gen(genes, epistatics):
    return MutationGenerator(
        genes=genes, epistatics=epistatics,
        combine_mutations=MutationCombination.ADD,
        combine_array=ArrayCombination.MULTIPLY
    )


@pytest.fixture
def mut_gen_non_multi_gene(genes):
    return MutationGenerator(
        genes=genes,
        multi_gene_array=False,
        combine_mutations=MutationCombination.ADD,
        combine_array=ArrayCombination.MULTIPLY
    )


def test_gene_selection(monkeypatch, mut_gen):
    with monkeypatch.context() as m:
        m.setattr(np.random, "rand", lambda x, y: np.array([
            [0.3], [0.7], [0.1], [0.99]
        ]))
        selected_gene_indices = mut_gen._get_genes(4)
        np.testing.assert_equal(selected_gene_indices, [1, 3, 0, 4])


def test_update_fitness_arrays_multi_gene_array(mut_gen):
    old_mutation_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ],
    ])
    genes_mutated = np.array([0, 1])
    syns = np.array([1, 0])

    new_fitnesses, new_mutation_arrays = mut_gen._update_fitness_arrays(old_mutation_arrays, genes_mutated, syns)
    np.testing.assert_almost_equal(new_fitnesses, np.array([2, 1.01]))
    np.testing.assert_almost_equal(new_mutation_arrays, np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1, np.nan, 1.01, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]))


def test_update_fitness_arrays_non_multi_gene_array(mut_gen_non_multi_gene):
    old_mutation_arrays = np.array([[2.], [1.]])
    genes_mutated = np.array([0, 1])
    syns = np.array([1, 0])

    new_fitnesses, new_mutation_arrays = mut_gen_non_multi_gene._update_fitness_arrays(old_mutation_arrays,
                                                                                       genes_mutated, syns)
    np.testing.assert_almost_equal(new_fitnesses, np.array([2, 1.01]))
    np.testing.assert_almost_equal(new_mutation_arrays, np.array([
        [2],[1.01]
    ]))


def test_epistatic_combination(mut_gen):
    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3., np.nan, np.nan, np.nan, np.nan ],  # First epistatic effect
        [1, 1, np.nan, np.nan, 2., 3., np.nan, np.nan],  # Second epistatic effect
        [1, 1, 2, 3, 2., 3., np.nan, np.nan],  # Both epistatic effects
    ])
    new_fitness_array, epistatic_fitness_array = mut_gen._epistatic_combinations(fitness_arrays)
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, 2, 3., np.nan, np.nan, 3, np.nan],  # First epistatic effect
            [1, 1, np.nan, np.nan, 2., 3., np.nan, 1],  # Second epistatic effect
            [1, 1, 2, 3, 2., 3., 3, 1],  # Both epistatic effects
        ])
    )
    np.testing.assert_almost_equal(
        epistatic_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, np.nan, np.nan, np.nan, np.nan, 3, np.nan],  # First epistatic effect
            [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],  # Second epistatic effect
            [1, np.nan, np.nan, np.nan, np.nan, np.nan, 3, 1],  # Both epistatic effects
        ])
    )


def test_combine_vectors1(mut_gen):
    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3., np.nan, np.nan, np.nan, np.nan],  # First epistatic effect
        [1, 1, np.nan, np.nan, 2., 3., np.nan, np.nan],  # Second epistatic effect
        [1, 1, 2, 3, 2., 3., np.nan, np.nan],  # Both epistatic effects
    ])
    new_fitness_array, full_fitness_arrays = mut_gen.combine_vectors(fitness_arrays)
    np.testing.assert_almost_equal(
        full_fitness_arrays,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, 2, 3., np.nan, np.nan, 3, np.nan],  # First epistatic effect
            [1, 1, np.nan, np.nan, 2., 3., np.nan, 1],  # Second epistatic effect
            [1, 1, 2, 3, 2., 3., 3, 1],  # Both epistatic effects
        ])
    )
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([2, 3, 1, 3])
    )


def test_combine_vectors2(mut_gen_non_multi_gene):
    fitness_arrays = np.array([[2.], [1.]])
    new_fitness_array, full_fitness_arrays = mut_gen_non_multi_gene.combine_vectors(fitness_arrays)
    np.testing.assert_almost_equal(
        full_fitness_arrays,
        np.array([[2.], [1.]])
    )
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([2., 1.])
    )
