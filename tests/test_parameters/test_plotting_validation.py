import pytest
from pydantic import ValidationError

from clone_competition_simulation.parameters.plotting_validation import PlottingParameters, PlottingValidator


def test_plotting_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = PlottingValidator()

        assert 'tag' not in str(exc_info)


@pytest.fixture
def empty_plotting_parameters():
    return PlottingParameters(tag="Base")


def test_plotting_validation1(empty_plotting_parameters):
    p = PlottingValidator(
        tag="Full",
        algorithm="WF2D",
        config_file_settings=empty_plotting_parameters,
        figsize=(1, 2)
    )

    assert p.figsize == (1, 2)
    assert p.colourscales.name == "random"
