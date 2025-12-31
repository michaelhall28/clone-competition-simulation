import numpy as np
import pytest
from pydantic import ValidationError

from clone_competition_simulation.fitness import (
    MutationGenerator,
    Gene,
    FixedValue,
    MutationCombination,
    ArrayCombination,
    UnboundedFitness
)
from clone_competition_simulation.parameters.fitness_validation import (
    FitnessValidator
)


def test_gene1():
    with pytest.raises(ValidationError) as exc_info:
        gene = Gene()
        assert 'name' in str(exc_info)
        assert 'mutation_distribution' in str(exc_info)
        assert 'synonymous_proportion' in str(exc_info)


def test_gene2():
    gene = Gene(name="test", mutation_distribution=FixedValue(1.1), synonymous_proportion=0.1)
    assert gene.name == "test"
    assert gene.mutation_distribution() == 1.1
    assert gene.synonymous_proportion == 0.1


def test_mutation_generation1():
    with pytest.raises(ValidationError) as exc_info:
        mut_gen = MutationGenerator()
        assert 'genes' in str(exc_info)


def test_mutation_generation2(gene):
    mut_gen = MutationGenerator(genes=[gene])

    assert mut_gen.genes == [gene]
    assert mut_gen.combine_mutations == MutationCombination.MULTIPLY
    assert mut_gen.combine_array == ArrayCombination.MULTIPLY
    assert isinstance(mut_gen.mutation_combination_class, UnboundedFitness)
    assert not mut_gen.multi_gene_array


def test_fitness_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = FitnessValidator()

        assert 'tag' not in str(exc_info)
        assert 'algorithm' in str(exc_info)
        assert 'times' in str(exc_info)
        assert 'population' in str(exc_info)


def test_fitness_validation1(empty_fitness_params, validated_time_parameters,
                               validated_population_parameters):
    p = FitnessValidator(algorithm="WF2D", tag="Full",
                         config_file_settings=empty_fitness_params,
                         times=validated_time_parameters,
                         population=validated_population_parameters
                         )
    assert p.mutation_generator is None
    np.testing.assert_array_equal(p.mutation_rates, np.array([[0, 0]]))
