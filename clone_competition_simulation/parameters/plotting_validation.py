from typing import Annotated, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import assign_config_settings, ValidationBase
from clone_competition_simulation.plotting import ColourScale, get_default_random_colourscale


class PlottingParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    figsize: tuple[int, int] | None = None
    colourscales: ColourScale | None = None


class PlottingValidator(PlottingParameters, ValidationBase):
    _default_strat_sim = 1
    tag: Literal['Full']
    config_file_settings: PlottingParameters | None = None

    def _validate_model(self):
        self.colourscales = self.get_value_from_config("colourscales")
        if self.colourscales is None:
            self.colourscales = get_default_random_colourscale()
        self.figsize = self.get_value_from_config("figsize")


plotting_validation_type = Annotated[
    (Annotated[PlottingParameters, Tag("Base")] | Annotated[PlottingValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
