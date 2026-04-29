from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Tag

from ..plotting import PlotColourMaps
from .validation_utils import (ParameterBase, ValidationBase,
                               assign_config_settings)


class PlottingParameters(ParameterBase):
    """Parameters that control plotting options for simulation output.

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
    plot_colour_maps: PlotColourMaps | None = None


class PlottingValidator(PlottingParameters, ValidationBase):
    _default_strat_sim = 1
    tag: Literal['Full']
    config_file_settings: PlottingParameters | None = None

    def _validate_model(self):
        self.plot_colour_maps = self.get_value_from_config("plot_colour_maps")
        if self.plot_colour_maps is None:
            self.plot_colour_maps = PlotColourMaps()
        self.figsize = self.get_value_from_config("figsize")


plotting_validation_type = Annotated[
    (Annotated[PlottingParameters, Tag("Base")] | Annotated[PlottingValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
