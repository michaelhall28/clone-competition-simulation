import numpy as np
import pytest
from pydantic import ValidationError

from src.clone_competition_simulation.parameters.population_validation import PopulationValidator, PopulationParameters


def test_population_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        pop = PopulationValidator()

        assert 'tag' not in str(exc_info)
        assert 'algorithm' in str(exc_info)


def test_population_validation_missing_parameters2(empty_population_parameters):
    with pytest.raises(ValidationError) as exc_info:
        pop = PopulationValidator(algorithm="WF2D", tag="Full",
                                  config_file_settings=empty_population_parameters)


        error_msg = str(exc_info)
        assert 'tag' not in error_msg
        assert 'algorithm' not in error_msg
        assert """Must provide one of:
        	initial_cells
        	initial_size_array
        	grid_shape (Moran2D/WF2D only)
        	initial_grid (Moran2D/WF2D only)""" in error_msg

def test_array_types():
    p = PopulationParameters(
        initial_size_array=np.array([1., 2., 3.])
    )
    np.testing.assert_array_equal(p.initial_size_array, np.array([1, 2, 3]))
    assert p.initial_size_array.dtype == np.int64

    p = PopulationParameters(
        initial_grid=np.arange(16, dtype=np.float64)
    )
    np.testing.assert_array_equal(p.initial_grid, np.arange(16, dtype=np.int64))
    assert p.initial_grid.dtype == np.int64

