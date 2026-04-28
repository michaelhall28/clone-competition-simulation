import pytest
from pydantic import ValidationError

from src.clone_competition_simulation.parameters.plotting_validation import PlottingParameters, PlottingValidator
from src.clone_competition_simulation.plotting.plot_colours import DEFAULT_COLOUR_RULE, PlotColourMaps


def test_plotting_validation_missing_parameters1():
    with pytest.raises(ValidationError) as exc_info:
        p = PlottingValidator(tag="Full")

    assert 'algorithm\n' in str(exc_info)


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
    assert p.plot_colour_maps.colour_rules[0] == DEFAULT_COLOUR_RULE


def test_plotting_validation2(empty_plotting_parameters):
    p = PlottingValidator(
        tag="Full",
        algorithm="WF2D",
        config_file_settings=empty_plotting_parameters,
        figsize=(1, 2), 
        plot_colour_maps=PlotColourMaps()
    )

    assert p.figsize == (1, 2)
    assert p.plot_colour_maps.colour_rules[0] == DEFAULT_COLOUR_RULE
