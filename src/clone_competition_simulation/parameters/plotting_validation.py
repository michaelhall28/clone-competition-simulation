from typing import Annotated, Literal
import matplotlib

from pydantic import BeforeValidator, ConfigDict, Tag

from ..plotting import PlotColourMaps, ColourRule
from .validation_utils import (ParameterBase, ValidationBase,
                               assign_config_settings)


class PlottingParameters(ParameterBase):
    """Parameters controlling plotting options for simulation output.

    Fields:
        figsize:
            Size of the figure used for plotting, specified as a tuple of the form
            (width, height) in inches. If not provided, default figure sizing is
            used by the plotting backend.

            Example:
                figsize = (10, 8)
                figsize = (12, 6)

        plot_colour_maps:
            A PlotColourMaps object defining the colour maps used for each clone in
            plots. If not provided, the default colour maps from the plotting module
            are used.

            Example:
                plot_colour_maps = PlotColourMaps(...) # Complex object, see docs
    """
    _field_name = "plotting"
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    figsize: tuple[int, int] | None = None
    plot_colour_maps: PlotColourMaps | dict | None = None


class PlottingValidator(PlottingParameters, ValidationBase):
    """Validate and prepare plotting parameters for simulation output.

    This validator reads plotting parameters from configuration and ensures
    that colour maps are properly initialized with defaults if not provided.
    """
    _default_strat_sim = 1
    tag: Literal['Full']
    config_file_settings: PlottingParameters | None = None

    def _validate_model(self):
        """Validate and initialize plotting parameters.

        Reads plotting parameters from configuration and applies defaults.
        Initializes colour maps with a default PlotColourMaps object if
        not explicitly provided.
        """
        if self.plot_colour_maps is None:
            self.plot_colour_maps = self._get_plot_colour_maps_from_config()
        self.figsize = self.get_value_from_config("figsize")

    def _get_plot_colour_maps_from_config(self) -> PlotColourMaps:
        """Retrieve the PlotColourMaps object from configuration.

        Returns:
            PlotColourMaps: The colour maps for plotting clones and mutations.
        """
        plot_colour_maps_dict = self.config_file_settings.plot_colour_maps
        if plot_colour_maps_dict is None:
            return PlotColourMaps()
        
        if 'colour_rules' in plot_colour_maps_dict:
            colour_rules = _get_colour_rules_from_config_file(
                plot_colour_maps_dict['colour_rules'])
            plot_colour_maps_dict['colour_rules'] = colour_rules
        return PlotColourMaps(**plot_colour_maps_dict)


def _get_colour_rules_from_config_file(colour_rules_dict: list[dict]) -> list[ColourRule]:
    """Create colour_rules objects from config file settings

    Replaces the colourmap string with the MatPlotlib colourmap object.

    Parameters
    ----------
    colour_rules_dict : dict
        Dictionary of gene parameters from the config file.

    Returns
    -------
    Gene
        A Gene object initialized with the provided parameters.
    """
    colour_rules = []
    for rule_dict in colour_rules_dict:
        rule_dict['colourmap'] = getattr(matplotlib.cm, rule_dict['colourmap'])
        rule = ColourRule(
            **rule_dict
        )
        colour_rules.append(rule)
    return colour_rules


plotting_validation_type = Annotated[
    (Annotated[PlottingParameters, Tag("Base")] | Annotated[PlottingValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
