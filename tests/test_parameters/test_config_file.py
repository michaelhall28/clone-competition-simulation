import os

import pytest
import numpy as np
from pydantic import ValidationError

from src.clone_competition_simulation.parameters import (
    Parameters, 
    Algorithm, 
    FitnessParameters, 
    TimeParameters, 
    PopulationParameters
)
from src.clone_competition_simulation.fitness import (
    multiply_fitness, 
    add_array_fitness, 
    BoundedLogisticFitness,
    NormalDist, 
    FixedValue,
    ExponentialDist,
)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_validation_from_config1():
    """Test all the parameters from the config file. 
    """
    p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config.yml"))
    assert p.algorithm == Algorithm.MORAN

    assert p.population.initial_cells == 200
    np.testing.assert_array_equal(p.population.initial_size_array, [100, 100])

    assert p.times.division_rate == 1.2
    assert p.times.max_time == 10
    assert p.times.samples == 5
    np.testing.assert_array_equal(p.times.times, [0, 2, 4, 6, 8, 10])

    
    np.testing.assert_array_equal(p.fitness.initial_fitness_array, 
                                  [[1., 1, np.nan], 
                                   [1., np.nan, 2]])
    np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])
    np.testing.assert_array_equal(p.fitness.initial_mutant_gene_array, 
                                  [0, 1])

    fitness_calculator = p.fitness.fitness_calculator
    assert fitness_calculator.multi_gene_array is True
    assert fitness_calculator.mutation_combination_class.__class__ == BoundedLogisticFitness
    assert fitness_calculator.mutation_combination_class.a == 1.2
    assert fitness_calculator.mutation_combination_class.b == 2.3
    assert fitness_calculator.combine_mutations == multiply_fitness
    assert fitness_calculator.combine_array == add_array_fitness
    gene1 = fitness_calculator.genes[0]
    assert gene1.name == 'Gene1'
    assert gene1.synonymous_proportion == 0.5
    assert gene1.mutation_distribution.__class__ == NormalDist
    assert gene1.mutation_distribution.mean == 1.2
    assert gene1.mutation_distribution.var == 1.1
    assert gene1.weight == 1
    gene2 = fitness_calculator.genes[1]
    assert gene2.name == 'Gene2'
    assert gene2.synonymous_proportion == 0.8
    assert gene2.mutation_distribution.__class__ == FixedValue
    assert gene2.mutation_distribution.mean == 1.1
    assert gene2.weight == 2
    assert len(fitness_calculator.epistatics) == 1
    epistatic_effect = fitness_calculator.epistatics[0]
    assert epistatic_effect.gene_names == ['Gene1', 'Gene2']
    assert epistatic_effect.fitness_distribution.__class__ == ExponentialDist
    assert epistatic_effect.fitness_distribution.mean == 1.5
    assert epistatic_effect.fitness_distribution.offset == 1.1


    np.testing.assert_array_equal(p.labels.initial_label_array, [1, 2])
    np.testing.assert_array_equal(p.labels.label_times, [3, 6])
    np.testing.assert_array_equal(p.labels.label_frequencies, [0.04, 0.1])
    np.testing.assert_array_equal(p.labels.label_values, [3, 4])
    np.testing.assert_array_equal(p.labels.label_fitness, [2, 3])
    np.testing.assert_array_equal(p.labels.label_genes, 
                                  ["Gene1", "Gene2"])


    assert p.differentiated_cells.r == 0.1
    assert p.differentiated_cells.gamma == 1.4
    assert p.differentiated_cells.stratification_sim_proportion == 0.99


def test_validation_from_config2():
    p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config.yml"),
                   algorithm=Algorithm.BRANCHING, population={'initial_cells': 1000})
    assert p.algorithm == Algorithm.BRANCHING
    assert p.times.division_rate == 1
    assert p.times.max_time == 10
    assert p.times.samples == 100
    assert p.population.initial_cells == 1000
    np.testing.assert_array_equal(p.fitness.initial_fitness_array, [1])
    np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])


def test_validation_from_config3():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config2.yml"))

    assert "config_file_settings.population.initial_cells\n  Input should be a valid integer" in str(exc_info)


def test_validation_from_config4():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config3.yml"))

    assert "config_file_settings.algorithm\n  Input should be 'WF', 'WF2D', 'Moran', 'Moran2D'" in str(exc_info)


def test_validation_from_config5(monkeypatch):
    with monkeypatch.context() as m:
        monkeypatch.setenv('CCS_RUN_CONFIG', os.path.join(CURRENT_DIR, "test_run_config.yml"))

        p = Parameters()
        assert p.algorithm == Algorithm.MORAN
        assert p.times.division_rate == 1
        assert p.times.max_time == 10
        assert p.times.samples == 100
        assert p.population.initial_cells == 100
        np.testing.assert_array_equal(p.fitness.initial_fitness_array, [1])
        np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])
        del os.environ["CCS_RUN_CONFIG"]


def test_validation_from_config_with_typo():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config4.yml"))

    assert "Extra inputs are not permitted" in str(exc_info)

