import numpy as np
import pytest
from pydantic import ValidationError

from src.clone_competition_simulation.fitness import (ArrayCombination,
                                                      FitnessCalculator,
                                                      FixedValue, Gene,
                                                      MutationCombination,
                                                      UnboundedFitness)
from src.clone_competition_simulation.parameters.fitness_validation import \
    FitnessValidator
from src.clone_competition_simulation.parameters.population_validation import \
    PopulationValidator


def test_gene1():
    with pytest.raises(ValidationError) as exc_info:
        gene = Gene()

    assert 'name\n' in str(exc_info)

    with pytest.raises(ValidationError) as exc_info:
        gene = Gene(name="test")

    assert 'name\n' not in str(exc_info)
    assert 'mutation_distribution\n' in str(exc_info)

    with pytest.raises(ValidationError) as exc_info:
        gene = Gene(name="test", mutation_distribution=FixedValue(1.1))

    assert 'name\n' not in str(exc_info)
    assert 'mutation_distribution\n' not in str(exc_info)
    assert 'synonymous_proportion\n' in str(exc_info)


def test_gene2():
    gene = Gene(name="test", mutation_distribution=FixedValue(1.1), synonymous_proportion=0.1)
    assert gene.name == "test"
    assert gene.mutation_distribution() == 1.1
    assert gene.synonymous_proportion == 0.1


def test_fitness_calculator1():
    with pytest.raises(ValidationError) as exc_info:
        fit_calc = FitnessCalculator()

    assert 'genes' in str(exc_info)


def test_fitness_calculator2(gene):
    fit_calc = FitnessCalculator(genes=[gene])

    assert fit_calc.genes == [gene]
    assert fit_calc.combine_mutations == MutationCombination.MULTIPLY
    assert fit_calc.combine_array == ArrayCombination.MULTIPLY
    assert isinstance(fit_calc.mutation_combination_class, UnboundedFitness)
    assert not fit_calc.multi_gene_array


def test_fitness_validation_missing_parameters1(validated_time_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator(tag="Full")

    assert 'algorithm' in str(exc_info)

    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator(algorithm="WF2D", tag="Full")

    assert 'algorithm\n' not in str(exc_info)
    assert 'times\n' in str(exc_info)

    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator(algorithm="WF2D", times=validated_time_parameters, tag="Full")

    assert 'algorithm\n' not in str(exc_info)
    assert 'times\n' not in str(exc_info)
    assert 'population\n' in str(exc_info)


def test_fitness_validation1(empty_fitness_params, validated_time_parameters,
                               validated_population_parameters):
    p = FitnessValidator(algorithm="WF2D", tag="Full",
                         config_file_settings=empty_fitness_params,
                         times=validated_time_parameters,
                         population=validated_population_parameters
                         )
    assert p.fitness_calculator is None
    np.testing.assert_array_equal(p.mutation_rates, np.array([[0, 0]]))


def test_fitness_validation_mutant_array(empty_fitness_params, validated_time_parameters,
                               validated_population_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator(
            algorithm="WF2D", tag="Full",
            config_file_settings=empty_fitness_params,
            times=validated_time_parameters,
            population=validated_population_parameters, 
            fitness_calculator=FitnessCalculator(
                genes=[Gene(name="gene1", mutation_distribution=FixedValue(1.1), synonymous_proportion=0.1)]
            ),
            initial_mutant_gene_array=[None, "gene1", None]
        )

    assert 'Inconsistent initial_size_array and initial_mutant_gene_array' in str(exc_info)


def test_fitness_validation_mutant_array2(empty_fitness_params, validated_time_parameters, empty_population_parameters):
    p = FitnessValidator(
        algorithm="WF2D", tag="Full",
        config_file_settings=empty_fitness_params,
        times=validated_time_parameters,
        population=PopulationValidator(
            tag="Full",
            algorithm="WF",
            config_file_settings=empty_population_parameters,
            initial_size_array=[100, 100, 100]
        ), 
        fitness_calculator=FitnessCalculator(
            genes=[Gene(name="gene1", mutation_distribution=FixedValue(1.1), synonymous_proportion=0.1)]
        ),
        initial_mutant_gene_array=[None, "gene1", None]
    )
    np.testing.assert_array_equal(p.initial_mutant_gene_array, np.array([np.nan, 0, np.nan]))


def test_fitness_validation_mutant_array3(empty_fitness_params, validated_time_parameters, empty_population_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator(
            algorithm="WF2D", tag="Full",
            config_file_settings=empty_fitness_params,
            times=validated_time_parameters,
            population=PopulationValidator(
                tag="Full",
                algorithm="WF",
                config_file_settings=empty_population_parameters,
                initial_size_array=[100, 100, 100]
            ), 
            fitness_calculator=FitnessCalculator(
                genes=[Gene(name="gene1", mutation_distribution=FixedValue(1.1), synonymous_proportion=0.1)]
            ),
            initial_mutant_gene_array=[None, "gene1", "gene2"]
        )

    assert 'Gene name gene2 not found' in str(exc_info)
