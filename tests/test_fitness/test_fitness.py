import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError
from src.clone_competition_simulation.fitness import (
    FitnessCalculator,
    Gene,
    FixedValue,
    NormalDist,
    UniformDist,
    ExponentialDist,
    BoundedLogisticFitness,
    UnboundedFitness,
    EpistaticEffect,
    add_fitness,
    multiply_fitness,
    replace_fitness,
    add_array_fitness,
    multiply_array_fitness,
    priority_array_fitness,
    max_fitness,
    min_fitness,
    max_array_fitness,
    min_array_fitness
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


def test_uniform_dist2():
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
    assert 'weight must be greater than 0' in str(excinfo)


def test_gene1():
    with pytest.raises(ValidationError) as excinfo:
        Gene(name='neutral', mutation_distribution=FixedValue(0.01), synonymous_proportion=-1)
    assert 'synonymous_proportion must be between 0 and 1' in str(excinfo.value)

    with pytest.raises(ValidationError)  as excinfo:
        Gene(name='neutral', mutation_distribution=FixedValue(0.01), synonymous_proportion=2)
    assert 'synonymous_proportion must be between 0 and 1' in str(excinfo.value)


def test_epistatic_effect():
    e = EpistaticEffect(name='E1', gene_names=["Gene1", "Gene2"], 
                        fitness_distribution=FixedValue(1))
    assert e.gene_names == {"Gene1", "Gene2"}
    assert e.fitness_distribution() == 1


def test_epistatic_effect1():
    with pytest.raises(ValidationError) as excinfo:
        EpistaticEffect(name='E1', gene_names=[], 
                        fitness_distribution=FixedValue(1))
        
    assert "Must have at least two genes in an epistatic effect" in str(excinfo)


def test_epistatic_effect2():
    with pytest.raises(ValidationError) as excinfo:
        EpistaticEffect(name='E1', gene_names="gene1, gene2", 
                        fitness_distribution=FixedValue(1))
    assert "Input should be a valid set" in str(excinfo)


@pytest.mark.parametrize("method,expected", [
    (add_fitness, np.array([0.6, 4.2])),
    (multiply_fitness, np.array([0.55, 6.6])),
    (replace_fitness, np.array([0.5, 3])),
    (max_fitness, np.array([1.1, 3])),
    (min_fitness, np.array([0.5, 2.2]))
])
def test_mutation_combination(method, expected):
    x = np.array([1.1, 2.2])
    y = np.array([0.5, 3])
    np.testing.assert_almost_equal(method(x, y), expected)

@pytest.mark.parametrize("method,expected", [
    (add_array_fitness, [2.3, 3.3]),
    (multiply_array_fitness, [2.42, 1.8]),
    (priority_array_fitness, [1.1, 0.3]),
    (max_array_fitness, [2.2, 3]),
    (min_array_fitness, [1.1, 0.3])
])
def test_mutation_combination(method, expected):
    x = np.array(
        [
            [2.2, np.nan, 1.1, np.nan],
            [1, 2, 3, 0.3]
        ]
    )
    np.testing.assert_almost_equal(method(x), expected)


def test_mutation_generation_validation(genes, epistatics):
    fit_calc = FitnessCalculator(
        genes=genes, epistatics=epistatics,
        combine_mutations=add_fitness,
        combine_array=multiply_array_fitness
    )
    assert fit_calc._num_genes == len(genes)
    assert fit_calc._gene_indices == {
        'neutral': 0, 'mild_driver': 1, 'random_driver': 2, "uniform_driver": 3, "exp_driver": 4,
        "E1": 5, "E2": 6
    }
    assert isinstance(fit_calc.mutation_combination_class, UnboundedFitness)
    expected_dists = [FixedValue, FixedValue, NormalDist, UniformDist, ExponentialDist]
    for dist, exp_dist in zip(fit_calc._mutation_distributions, expected_dists):
        assert isinstance(dist, exp_dist)
    np.testing.assert_equal(fit_calc._synonymous_proportions, [0.5, 0.5, 0.8, 0.2, 0.2])
    np.testing.assert_almost_equal(fit_calc._overall_synonymous_proportion,0.44)
    np.testing.assert_almost_equal(fit_calc._relative_weights_cumsum, [0.1, 0.4, 0.6, 0.8, 1.])
    assert set(fit_calc._epistatics_info[0].gene_columns) == {1, 4, 5}
    assert set(fit_calc._epistatics_info[1].gene_columns) == {2, 3}
    

    np.testing.assert_equal(fit_calc._epistatic_cols, np.arange(6, 8))
    assert fit_calc.combine_mutations == add_fitness
    assert fit_calc.combine_array == multiply_array_fitness


@pytest.mark.parametrize("method,expected", [
    ("add", add_fitness),
    ("multiply", multiply_fitness),
    ("replace", replace_fitness),
    ("max", max_fitness),
    ("min", min_fitness)
])
def test_combine_mutations_validation(genes, epistatics, method, expected):
    fit_calc = FitnessCalculator(
        genes=genes, epistatics=epistatics,
        combine_mutations=method,
        combine_array=multiply_array_fitness
    )
    assert fit_calc.combine_mutations == expected


@pytest.mark.parametrize("method,expected", [
    ("add", add_array_fitness),
    ("multiply", multiply_array_fitness),
    ("priority", priority_array_fitness),
    ("max", max_array_fitness),
    ("min", min_array_fitness)
])
def test_combine_array_validation(genes, epistatics, method, expected):
    fit_calc = FitnessCalculator(
        genes=genes, epistatics=epistatics,
        combine_mutations=multiply_fitness,
        combine_array=method
    )
    assert fit_calc.combine_array == expected


def test_custom_fitness_combination():
    def custom_combine(old, new):
        return old + new + 1

    fit_calc = FitnessCalculator(
        genes=[Gene(name="test", mutation_distribution=FixedValue(1.1), synonymous_proportion=0)],
        combine_mutations=custom_combine,
        combine_array=multiply_array_fitness
    )
    assert fit_calc.combine_mutations == custom_combine


def test_custom_array_combination():
    def custom_combine_array(arr):
        return np.nansum(arr, axis=1)

    fit_calc = FitnessCalculator(
        genes=[Gene(name="test", mutation_distribution=FixedValue(1.1), synonymous_proportion=0)],
        combine_mutations=add_fitness,
        combine_array=custom_combine_array
    )
    assert fit_calc.combine_array == custom_combine_array


@pytest.fixture
def fit_calc(genes, epistatics):
    return FitnessCalculator(
        genes=genes, epistatics=epistatics,
        combine_mutations=add_fitness,
        combine_array=multiply_array_fitness
    )


@pytest.fixture
def fit_calc_non_multi_gene(genes):
    return FitnessCalculator(
        genes=genes,
        multi_gene_array=False,
        combine_mutations=add_fitness,
        combine_array=multiply_array_fitness
    )


def test_gene_selection(monkeypatch, fit_calc):
    with monkeypatch.context() as m:
        m.setattr(np.random, "rand", lambda x, y: np.array([
            [0.3], [0.7], [0.1], [0.99]
        ]))
        selected_gene_indices = fit_calc._get_genes(4)
        np.testing.assert_equal(selected_gene_indices, [1, 3, 0, 4])


def test_update_fitness_arrays_multi_gene_array(fit_calc):
    old_mutation_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ],
    ])
    genes_mutated = np.array([0, 1])
    syns = np.array([1, 0])

    new_fitnesses, new_mutation_arrays = fit_calc._update_fitness_arrays(old_mutation_arrays, genes_mutated, syns)
    np.testing.assert_almost_equal(new_fitnesses, np.array([2, 1.01]))
    np.testing.assert_almost_equal(new_mutation_arrays, np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1, np.nan, 1.01, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]))


def test_update_fitness_arrays_non_multi_gene_array(fit_calc_non_multi_gene):
    old_mutation_arrays = np.array([[2.], [1.]])
    genes_mutated = np.array([0, 1])
    syns = np.array([1, 0])

    new_fitnesses, new_mutation_arrays = fit_calc_non_multi_gene._update_fitness_arrays(old_mutation_arrays,
                                                                                       genes_mutated, syns)
    np.testing.assert_almost_equal(new_fitnesses, np.array([2, 1.01]))
    np.testing.assert_almost_equal(new_mutation_arrays, np.array([
        [2],[1.01]
    ]))


def test_epistatic_combination(fit_calc):
    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3., np.nan, np.nan, np.nan, np.nan ],  # First epistatic effect
        [1, 1, np.nan, np.nan, 2., 3., np.nan, np.nan],  # Second epistatic effect
        [1, 1, 2, 3, 2., 3., np.nan, np.nan],  # Both epistatic effects
    ])
    new_fitness_array, epistatic_fitness_array = fit_calc._epistatic_combinations(fitness_arrays)
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


def test_epistatic_combination2(genes: list[Gene]) :
    epistatics = [
        EpistaticEffect(name="E1", gene_names=['mild_driver', 'random_driver'],
                        fitness_distribution=FixedValue(10)),
        EpistaticEffect(name="E2", gene_names=['mild_driver', 'random_driver', 'exp_driver'],
                        fitness_distribution=FixedValue(100))
    ]
    fit_calc = FitnessCalculator(
        genes=genes,
        epistatics=epistatics,
        multi_gene_array=False,
        combine_mutations=add_fitness,
        combine_array=add_array_fitness
    )

    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3, np.nan, np.nan, np.nan, np.nan ],  # First epistatic effect
        [1, 1, 2, 3, 4, 5, np.nan, np.nan],  # Both epistatic effects
        [1, 1, 2, 3, np.nan, 5, 10, np.nan]  # First epistatic effect already applied
    ])
    new_fitness_array, epistatic_fitness_array = fit_calc._epistatic_combinations(fitness_arrays)
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, 2, 3., np.nan, np.nan, 10, np.nan],  # First epistatic effect
            [1, 1, 2, 3, 4, 5, np.nan, 100],  # Both epistatic effect
            [1, 1, 2, 3, np.nan, 5, 10, 100],  # Both, where first effect was already applied
        ])
    )
    np.testing.assert_almost_equal(
        epistatic_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, np.nan, np.nan, np.nan, np.nan, 10, np.nan],  # First epistatic effect
            [1, 1, np.nan, np.nan, 4, np.nan, np.nan, 100],  # Both epistatic effects
            [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, 100],  # First had already been applied
        ])
    )


def test_epistatic_combination3(genes: list[Gene]):
    # test priority combinations with epistatics. 
    # A case with superseding (no change from normal)
    # and a case without superseding (only take the last one)
    epistatics = [
        EpistaticEffect(name="E2", gene_names=['mild_driver', 'random_driver', 'exp_driver'],
                        fitness_distribution=FixedValue(100)),
        EpistaticEffect(name="E1", gene_names=['mild_driver', 'random_driver'],
                        fitness_distribution=FixedValue(10)),
        EpistaticEffect(name="E3", gene_names=['random_driver', 'uniform_driver', 'exp_driver'],
                        fitness_distribution=FixedValue(200))

    ]
    fit_calc = FitnessCalculator(
        genes=genes,
        epistatics=epistatics,
        multi_gene_array=False,
        combine_mutations=add_fitness,
        combine_array=priority_array_fitness
    )

    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3., np.nan, np.nan, np.nan, np.nan, np.nan ],  # First epistatic effect
        [1, 1, 2, 3, np.nan, 5, np.nan, np.nan, np.nan],  # First two epistatic effects
        [1, 1, 2, 3, 4., 5, np.nan, np.nan, np.nan],  # All epistatic effects
    ])
    new_fitness_array, epistatic_fitness_array = fit_calc._epistatic_combinations(fitness_arrays)
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, 2, 3., np.nan, np.nan, np.nan, 10, np.nan],  # First epistatic effect
            [1, 1, 2, 3, np.nan, 5, 100, np.nan, np.nan],  # First two epistatic effects
            [1, 1, 2, 3, 4, 5, 100, np.nan, 200],  # All epistatic effects
        ])
    )
    np.testing.assert_almost_equal(
        epistatic_fitness_array,
        np.array([
            [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
            [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, 10    , np.nan],  # First epistatic effect
            [1, 1, np.nan, np.nan, np.nan, np.nan, 100   , np.nan, np.nan],  # First two epistatic effects
            [1, 1, np.nan, np.nan, np.nan, np.nan, 100   , np.nan, 200],  # All epistatic effects
        ])
    )

    new_fitness_array, full_fitness_arrays = fit_calc.combine_vectors(epistatic_fitness_array)
    np.testing.assert_almost_equal(
        full_fitness_arrays,
        np.array([
            [  1.,   2.,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan],
            [  1.,   1.,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  10.,  np.nan],
            [  1.,   1.,  np.nan,  np.nan,  np.nan,  np.nan, 100.,  np.nan,  np.nan],
            [  1.,   1.,  np.nan,  np.nan,  np.nan,  np.nan, 100.,  np.nan, 200.]
        ])
    )

    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([2, 10, 100, 200])
    )


def test_epistatic_combinations4():
    # The case from issue 60
    NOTCH1_HOM_FITNESS = 100
    NOTCH1_HET_FITNESS = 50
    NOTCH2_NOTCH1_HET_FITNESS = 30
    NOTCH2_NOTCH1_HOM_FITNESS = 7
    NOTCH2_TRP53_FITNESS = 2
    TRP53_FITNESS = 1.1
    NOTCH2_FITNESS = 1.3

    alleles = [
        Gene(name="Notch1_allele1", mutation_distribution=FixedValue(NOTCH1_HET_FITNESS), synonymous_proportion=0), 
        Gene(name="Notch1_allele2", mutation_distribution=FixedValue(NOTCH1_HET_FITNESS), synonymous_proportion=0), 
        Gene(name="Trp53", mutation_distribution=FixedValue(TRP53_FITNESS), synonymous_proportion=0), 
        Gene(name="Notch2", mutation_distribution=FixedValue(NOTCH2_FITNESS), synonymous_proportion=0), 
    ]
    epistatics = [EpistaticEffect(name='Notch1_hom', 
                                  gene_names=['Notch1_allele1', 'Notch1_allele2'], 
                                  fitness_distribution=FixedValue(NOTCH1_HOM_FITNESS)),
                  EpistaticEffect(name='Trp53_Notch1_het1', 
                                  gene_names=['Notch1_allele1', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH1_HET_FITNESS)),
                  EpistaticEffect(name='Trp53_Notch1_het2', 
                                  gene_names=['Notch1_allele2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH1_HET_FITNESS)),
                  EpistaticEffect(name='Notch2_Notch1_het1', 
                                  gene_names=['Notch1_allele1', 'Notch2'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HET_FITNESS)),
                  EpistaticEffect(name='Notch2_Notch1_het2', 
                                  gene_names=['Notch1_allele2', 'Notch2'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HET_FITNESS)),
                  EpistaticEffect(name='Notch2_Trp53', 
                                  gene_names=['Notch2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH2_TRP53_FITNESS)),

                  EpistaticEffect(name='Trp53_Notch1_hom', 
                                  gene_names=['Notch1_allele1', 'Notch1_allele2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH1_HOM_FITNESS)),
                  EpistaticEffect(name='Notch2_Notch1_hom', 
                                  gene_names=['Notch1_allele1', 'Notch1_allele2', 'Notch2'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HOM_FITNESS)),
                  EpistaticEffect(name='Notch2_Trp53_Notch1_het1', 
                                  gene_names=['Notch1_allele1', 'Notch2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HET_FITNESS)),
                  EpistaticEffect(name='Notch2_Trp53_Notch1_het2', 
                                  gene_names=['Notch1_allele2', 'Notch2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HET_FITNESS)),

                  EpistaticEffect(name='Notch2_Trp53_Notch1_hom', 
                                  gene_names=['Notch1_allele1', 'Notch1_allele2', 'Notch2', 'Trp53'], 
                                  fitness_distribution=FixedValue(NOTCH2_NOTCH1_HOM_FITNESS))
                 ]
    fit_calc = FitnessCalculator(
        genes = alleles,
        combine_mutations = replace_fitness,
        epistatics = epistatics,
        multi_gene_array=True,
    )
    fitness_combinations = fit_calc.plot_fitness_combinations()

    expected_index = ['Background', 'Notch1_allele1', 'Notch1_allele2',
       'Notch1_allele1 + Notch1_allele2', 'Trp53', 'Notch1_allele1 + Trp53',
       'Notch1_allele2 + Trp53', 'Notch1_allele1 + Notch1_allele2 + Trp53',
       'Notch2', 'Notch1_allele1 + Notch2', 'Notch1_allele2 + Notch2',
       'Notch1_allele1 + Notch1_allele2 + Notch2', 'Trp53 + Notch2',
       'Notch1_allele1 + Trp53 + Notch2', 'Notch1_allele2 + Trp53 + Notch2',
       'Notch1_allele1 + Notch1_allele2 + Trp53 + Notch2']
    expected_values = [
        1, NOTCH1_HET_FITNESS, NOTCH1_HET_FITNESS, 
        NOTCH1_HOM_FITNESS, TRP53_FITNESS, NOTCH1_HET_FITNESS, 
        NOTCH1_HET_FITNESS, NOTCH1_HOM_FITNESS, NOTCH2_FITNESS, 
        NOTCH2_NOTCH1_HET_FITNESS, NOTCH2_NOTCH1_HET_FITNESS, 
        NOTCH2_NOTCH1_HOM_FITNESS, NOTCH2_TRP53_FITNESS, 
        NOTCH2_NOTCH1_HET_FITNESS, NOTCH2_NOTCH1_HET_FITNESS, 
        NOTCH2_NOTCH1_HOM_FITNESS
    ]

    assert fitness_combinations.equals(
        pd.Series(expected_values, index=expected_index
        )
    )


def test_combine_vectors1(fit_calc):
    fitness_arrays = np.array([
        [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # No epistatics
        [1, 1, 2, 3., np.nan, np.nan, np.nan, np.nan],  # First epistatic effect
        [1, 1, np.nan, np.nan, 2., 3., np.nan, np.nan],  # Second epistatic effect
        [1, 1, 2, 3, 2., 3., np.nan, np.nan],  # Both epistatic effects
    ])
    new_fitness_array, full_fitness_arrays = fit_calc.combine_vectors(fitness_arrays)
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


def test_combine_vectors2(fit_calc_non_multi_gene):
    fitness_arrays = np.array([[2.], [1.]])
    new_fitness_array, full_fitness_arrays = fit_calc_non_multi_gene.combine_vectors(fitness_arrays)
    np.testing.assert_almost_equal(
        full_fitness_arrays,
        np.array([[2.], [1.]])
    )
    np.testing.assert_almost_equal(
        new_fitness_array,
        np.array([2., 1.])
    )
