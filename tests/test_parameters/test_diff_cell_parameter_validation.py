import pytest
from pydantic import ValidationError
from src.clone_competition_simulation.parameters.differentiated_cells_validation import (
    DifferentiatedCellsValidator
)


def test_diff_cell_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = DifferentiatedCellsValidator()

        assert 'tag' not in str(exc_info)
        assert 'algorithm' in str(exc_info)
        assert 'times' in str(exc_info)


def test_diff_cell_validation1(empty_diff_cell_params, validated_time_parameters):
    p = DifferentiatedCellsValidator(algorithm="WF2D", tag="Full",
                                     config_file_settings=empty_diff_cell_params,
                                     times=validated_time_parameters)
    assert p.r is None
    assert p.gamma is None
    assert not p.diff_cell_simulation


def test_diff_cell_validation2(empty_diff_cell_params, validated_time_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = DifferentiatedCellsValidator(
            algorithm="WF2D", tag="Full",
            config_file_settings=empty_diff_cell_params,
            times=validated_time_parameters,
            r=0.1)
        assert 'Cannot run WF2D algorithms with B cells.' in str(exc_info)


def test_diff_cell_validation3(empty_diff_cell_params, validated_time_parameters):
    with pytest.raises(ValidationError) as exc_info:
        p = DifferentiatedCellsValidator(
            algorithm="WF", tag="Full",
            config_file_settings=empty_diff_cell_params,
            times=validated_time_parameters,
            r=0.1)
        assert 'Cannot run WF algorithms with B cells.' in str(exc_info)


@pytest.mark.parametrize(
    "r,gamma,expected_msg", [
        (0.1, None, 'Must provide both r and gamma to run with B cells. Please provide gamma'),
        (None, 0.1, 'Must provide both r and gamma to run with B cells. Please provide r'),
        (0.6, 0.1, 'Must have 0<=r<=0.5'),
        (-0.1, 0.1, 'Must have 0<=r<=0.5'),
        (0.2, -0.1, 'gamma must be > 0')
    ]
)
def test_diff_cell_validation4(empty_diff_cell_params, validated_time_parameters, r,gamma, expected_msg):
    with pytest.raises(ValidationError) as exc_info:
        p = DifferentiatedCellsValidator(
            algorithm="Moran", tag="Full",
            config_file_settings=empty_diff_cell_params,
            times=validated_time_parameters,
            r=r, gamma=gamma)
        assert expected_msg in str(exc_info)


def test_diff_cell_validation5(empty_diff_cell_params, validated_time_parameters):
    p = DifferentiatedCellsValidator(
        algorithm="Moran", tag="Full",
        config_file_settings=empty_diff_cell_params,
        times=validated_time_parameters,
        r=0.1, gamma=0.2)

    assert p.r == 0.1
    assert p.gamma == 0.2




