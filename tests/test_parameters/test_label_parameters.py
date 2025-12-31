import numpy as np
import pytest
from pydantic import ValidationError

from clone_competition_simulation.parameters.label_validation import LabelParameters, LabelValidator


@pytest.fixture
def empty_label_parameters():
    return LabelParameters(tag="Base")


def test_label_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = LabelValidator()

        assert 'tag' not in str(exc_info)
        assert 'algorithm' in str(exc_info)
        assert 'population' in str(exc_info)
        assert 'fitness' in str(exc_info)


def test_label_array_types():
    p = LabelParameters(
        label_genes=np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_array_equal(p.label_genes, np.array([1, 2, 3]))
    assert p.label_genes.dtype == np.int64


def test_label_validation1(validated_population_parameters, validated_fitness_parameters, empty_label_parameters):
    p = LabelValidator(
        algorithm="WF2D",
        tag="Full",
        population=validated_population_parameters,
        fitness=validated_fitness_parameters,
        config_file_settings=empty_label_parameters,
        label_array=1,
        label_times=2,
        label_frequencies=0.01,
        label_values=3,
        label_genes=-1,
    )

    np.testing.assert_array_equal(p.label_array, np.array([1]))
    np.testing.assert_array_equal(p.label_times, np.array([2]))
    np.testing.assert_array_equal(p.label_frequencies, np.array([0.01]))
    np.testing.assert_array_equal(p.label_values, np.array([3]))
    np.testing.assert_array_equal(p.label_genes, np.array([-1]))


def test_label_validation2(validated_population_parameters,
                           validated_fitness_parameters_multi_gene,
                           empty_label_parameters):
    p = LabelValidator(
        algorithm="WF2D",
        tag="Full",
        population=validated_population_parameters,
        fitness=validated_fitness_parameters_multi_gene,
        config_file_settings=empty_label_parameters,
        label_array=1,
        label_times=2,
        label_frequencies=0.01,
        label_values=3,
        label_genes=0,
    )

    np.testing.assert_array_equal(p.label_array, np.array([1]))
    np.testing.assert_array_equal(p.label_times, np.array([2]))
    np.testing.assert_array_equal(p.label_frequencies, np.array([0.01]))
    np.testing.assert_array_equal(p.label_values, np.array([3]))
    np.testing.assert_array_equal(p.label_genes, np.array([0]))


def test_label_validation3(validated_population_parameters,
                           validated_fitness_parameters,
                           empty_label_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = LabelValidator(
            algorithm="WF2D",
            tag="Full",
            population=validated_population_parameters,
            fitness=validated_fitness_parameters,
            config_file_settings=empty_label_parameters,
            label_array=1,
            label_times=2,
            label_frequencies=0.01,
            label_values=3,
            label_genes=0,
        )

        assert 'tag' not in str(exc_info)
        assert ('Applying labels with mutations to particular genes requires a '
                'mutation generator with multi_gene_array=True') in str(exc_info)
