import numpy as np
import pytest
from pydantic import ValidationError

from src.clone_competition_simulation.parameters.times_validation import TimeValidator, TimeParameters


def test_time_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = TimeValidator(tag="Full")

    assert 'algorithm\n' in str(exc_info)


def test_time_validation_missing_parameters2(empty_time_parameters, validated_population_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = TimeValidator(
            algorithm="WF2D",
            tag="Full",
            config_file_settings=empty_time_parameters,
            population=validated_population_parameters
        )


    error_msg = str(exc_info)
    assert 'algorithm\n' not in error_msg
    assert "Division rate not defined. Define or set other time-related" in error_msg


def test_time_validation_missing_parameters3(empty_time_parameters, validated_population_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = TimeValidator(
            algorithm="WF2D",
            tag="Full",
            config_file_settings=empty_time_parameters,
            division_rate=1,
            population=validated_population_parameters
        )


    error_msg = str(exc_info)
    assert 'algorithm\n' not in error_msg
    assert "Max time not defined. Define or set other time-related settings" in error_msg


def test_array_types():
    p = TimeParameters(
        times=np.array([1, 2, 3])
    )
    np.testing.assert_array_equal(p.times, np.array([1., 2., 3.]))
    assert p.times.dtype == np.float64


def test_time_validation1(empty_time_parameters, validated_population_parameters):
    p = TimeValidator(
        algorithm="WF2D",
        tag="Full",
        population=validated_population_parameters,
        config_file_settings=empty_time_parameters,
        division_rate=1,
        max_time=10
    )

    assert p.max_time == 10
    assert p.division_rate == 1
    np.testing.assert_almost_equal(p.times, np.linspace(0, 10, 11))
    np.testing.assert_array_equal(p.sample_points, np.arange(0, 11))
    assert p.simulation_steps == 10



