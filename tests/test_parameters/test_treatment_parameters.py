import numpy as np
import pytest
from pydantic import ValidationError

from src.clone_competition_simulation.parameters.treatment_validation import TreatmentParameters, TreatmentValidator


def test_treatment_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = TreatmentValidator()

        assert 'tag' not in str(exc_info)
        assert 'algorithm' in str(exc_info)
        assert 'population' in str(exc_info)
        assert 'fitness' in str(exc_info)


@pytest.fixture
def empty_treatment_parameters():
    return TreatmentParameters(tag="Base")


def test_treatment_validation1(validated_population_parameters, validated_fitness_parameters, empty_treatment_parameters):
    p = TreatmentValidator(
        algorithm="WF2D",
        tag="Full",
        population=validated_population_parameters,
        fitness=validated_fitness_parameters,
        config_file_settings=empty_treatment_parameters,
        treatment_timings=[1],
        treatment_effects=[[2.3]],
        treatment_replace_fitness=False
    )

    np.testing.assert_array_equal(p.treatment_timings, np.array([0., 1.]))
    np.testing.assert_array_equal(p.treatment_effects, np.array([[1.], [2.3]]))
    assert p.treatment_replace_fitness is False

