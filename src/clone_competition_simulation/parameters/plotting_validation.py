from typing import Annotated, Literal
from pydantic import (
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import assign_config_settings, ValidationBase, ParameterBase
from ..plotting import PlotColourMaps


class PlottingParameters(ParameterBase):
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
