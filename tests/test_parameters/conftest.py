import pytest

from clone_competition_simulation import (
    PopulationParameters,
    TimeParameters,
    DifferentiatedCellsParameters,
    FitnessParameters, Gene, FixedValue,
    MutationGenerator
)
from clone_competition_simulation.parameters.fitness_validation import FitnessValidator
from clone_competition_simulation.parameters.population_validation import PopulationValidator
from clone_competition_simulation.parameters.times_validation import TimeValidator


@pytest.fixture
def empty_population_parameters():
    return PopulationParameters(tag="Base")


@pytest.fixture
def validated_population_parameters(empty_population_parameters):
    return PopulationValidator(
        tag="Full",
        algorithm="WF",
        config_file_settings=empty_population_parameters,
        initial_cells=100
    )


@pytest.fixture
def empty_time_parameters():
    return TimeParameters(tag="Base")


@pytest.fixture
def validated_time_parameters(empty_time_parameters, validated_population_parameters):
    return TimeValidator(
        tag="Full",
        algorithm="WF",
        config_file_settings=empty_time_parameters,
        population=validated_population_parameters,
        division_rate=1,
        max_time=10
    )


@pytest.fixture
def empty_diff_cell_params():
    return DifferentiatedCellsParameters(tag='Base')


@pytest.fixture
def empty_fitness_params():
    return FitnessParameters(tag='Base')


@pytest.fixture
def gene():
    return Gene(
        name="test",
        mutation_distribution=FixedValue(1.1),
        synonymous_proportion=0.1
    )


@pytest.fixture
def mutation_generator(gene):
    return MutationGenerator(
        genes=[gene],
        multi_gene_array=False
    )


@pytest.fixture
def validated_fitness_parameters(empty_fitness_params, validated_time_parameters,
                                 validated_population_parameters,
                                 mutation_generator
                                 ):
    return FitnessValidator(
        tag="Full",
        algorithm="WF2D",
        config_file_settings=empty_fitness_params,
        population=validated_population_parameters,
        times=validated_time_parameters,
        mutation_generator=mutation_generator,
    )

@pytest.fixture
def mutation_generator_multi_gene(gene):
    return MutationGenerator(
        genes=[gene],
        multi_gene_array=True
    )


@pytest.fixture
def validated_fitness_parameters_multi_gene(empty_fitness_params, validated_time_parameters,
                                 validated_population_parameters,
                                 mutation_generator_multi_gene
                                 ):
    return FitnessValidator(
        tag="Full",
        algorithm="WF2D",
        config_file_settings=empty_fitness_params,
        population=validated_population_parameters,
        times=validated_time_parameters,
        mutation_generator=mutation_generator_multi_gene,
    )



