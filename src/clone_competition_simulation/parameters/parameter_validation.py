import os
import re
from pathlib import Path
from typing import Callable, Any, Self

from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    ModelWrapValidatorHandler,
    ValidationError
)
from pydantic_settings import (
    BaseSettings,
    YamlConfigSettingsSource,
)

from .algorithm_validation import Algorithm
from .differentiated_cells_validation import differentiated_cells_validation_type
from .fitness_validation import fitness_validation_type
from .label_validation import label_validation_type
from .plotting_validation import plotting_validation_type
from .population_validation import population_validation_type
from .times_validation import times_validation_type
from .treatment_validation import treatment_validation_type
from .validation_utils import ValidationModelField, AlwaysValidateNoneField
from ..simulation_algorithms.branching_process import SimpleBranchingProcess
from ..simulation_algorithms.general_differentiated_cell_class import (
    Moran2DWithDiffcells, MoranWithDiffCells, BranchingWithDiffCells)
from ..simulation_algorithms.general_sim_class import GeneralSimClass
from ..simulation_algorithms.moran import MoranSim
from ..simulation_algorithms.moran2D import Moran2D
from ..simulation_algorithms.stop_conditions import WFStop, WF2DStop, MoranStop, Moran2DStop
from ..simulation_algorithms.wf import WrightFisherSim
from ..simulation_algorithms.wf2D import WrightFisher2D


class RunSettingsBase(BaseSettings):
    algorithm: Algorithm | None = None
    population: population_validation_type = ValidationModelField
    times: times_validation_type = ValidationModelField
    fitness: fitness_validation_type = ValidationModelField
    labels: label_validation_type = ValidationModelField
    treatment: treatment_validation_type = ValidationModelField
    differentiated_cells: differentiated_cells_validation_type = ValidationModelField
    plotting: plotting_validation_type = ValidationModelField

    end_condition_function: Callable[[GeneralSimClass], None] | None = None
    progress: int | None = None
    tmp_store: Path | None = None


class ConfigFileSettings(BaseModel):
    run_config_file: Path | None = AlwaysValidateNoneField
    config_file_settings: RunSettingsBase | None = AlwaysValidateNoneField

    @field_validator("run_config_file", mode="before")
    @classmethod
    def validate_config_file_path(cls, value) -> Path:
        if value is not None:
            return value
        if env_path := os.environ.get('RUN_CONFIG', None):
            run_config_file = Path(env_path)
        else:
            run_config_file = Path("run_config.yaml")
        return run_config_file

    @field_validator("config_file_settings", mode="before")
    @classmethod
    def load_config_file_settings(cls, data, info):
        y = YamlConfigSettingsSource(cls, yaml_file=info.data['run_config_file'])
        data = y()
        for field, field_info in RunSettingsBase.model_fields.items():
            if field_info.json_schema_extra and field_info.json_schema_extra.get("config_validation"):
                if field not in data:
                    data[field] = {}
                data[field]['tag'] = "Base"
        return data


ERROR_PATTERNS_TO_IGNORE = [
    # These validation errors are caused by previous errors. Filter out.
    (".*", ".*", "tag"),
    (".*", "Base", ".*"),
    (".*", "Full", "population"),
    (".*", "Full", "times"),
    (".*", "Full", "fitness"),
]
ERROR_PATTERNS_TO_DEDUPLICATE = [
    # Will be raised in multiple places, but only want to raise these once
    (".*", "Full", "algorithm"),
]

def match_loc(loc: tuple[str,...], match_pattern: tuple[str,...]) -> bool:
    """
    Match the loc from a validation error e.g. (population, Full, algorithm) agains the patterns we want to exclude.
    Args:
        loc:
        match_pattern:

    Returns:

    """
    if len(loc) != len(match_pattern):
        return False
    for l_pattern, m_pattern in zip(loc, match_pattern):
        if not re.match(m_pattern, l_pattern):
            return False
    return True


class Parameters(RunSettingsBase, ConfigFileSettings):

    @model_validator(mode='wrap')
    @classmethod
    def clean_validation_errors(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """
        The tags are used for class discrimination but the validation errors are not helpful for users.
        Args:
            data:
            handler:

        Returns:

        """
        try:
            model = handler(data)
            # Get any values from the config file that aren't in the nested models
            for field, field_info in cls.model_fields.items():
                if ((not field_info.json_schema_extra or field_info.json_schema_extra.get("config_validation") is None)
                        and getattr(model, field) is None):
                    setattr(model, field, getattr(model.config_file_settings, field))
            return model
        except ValidationError as e:
            clean_errors = []
            deduplicated_patterns = set()
            config_file_error = False
            for error in e.errors():
                loc = error['loc']
                if any(match_loc(loc, pattern) for pattern in ERROR_PATTERNS_TO_IGNORE):
                    continue
                if any(match_loc(loc, pattern) for pattern in deduplicated_patterns):
                    continue
                if loc[0] == "config_file_settings":
                    config_file_error = True
                for pattern in ERROR_PATTERNS_TO_DEDUPLICATE:
                    if match_loc(loc, pattern):
                        deduplicated_patterns.add(pattern)

                # Clean up the error loc
                if len(loc) == 3 and loc[2] == "algorithm":
                    # This will cause errors in the validation of the nested models,
                    # but is due either to the main model, or to an error in the config file settings
                    if config_file_error:
                        continue
                    error['loc'] = (loc[2],)
                elif "Full" in loc or "Base" in loc:
                    error['loc'] = tuple([lc for lc in loc if (lc not in ["Full", "Base"])])
                clean_errors.append(error)

            raise ValidationError.from_exception_data(title=cls.__name__, line_errors=clean_errors)


    @property
    def non_zero_calc(self):
        """
        If the algorithm uses the entire current population to calculate the next step, this is set to true.
        Speeds up those simulations.
        Returns:

        """
        return not self.algorithm.two_dimensional

    def _select_simulator_class(self):
        sim_class = None
        if self.differentiated_cells.diff_cell_simulation:  # Simulations including B cells.
            if self.end_condition_function is not None:
                raise ValueError(
                    'Cannot use an end_condition_function for the simulations including differentiated cells')
            if self.algorithm == Algorithm.MORAN:
                sim_class = MoranWithDiffCells
            elif self.algorithm == Algorithm.MORAN2D:
                sim_class = Moran2DWithDiffcells
            elif self.algorithm == Algorithm.BRANCHING:
                sim_class = BranchingWithDiffCells
        else:
            if self.algorithm == Algorithm.WF:
                if self.end_condition_function is not None:
                    sim_class = WFStop
                else:
                    sim_class = WrightFisherSim
            elif self.algorithm == Algorithm.MORAN:
                if self.end_condition_function is not None:
                    sim_class = MoranStop
                else:
                    sim_class = MoranSim
            elif self.algorithm == Algorithm.MORAN2D:
                if self.end_condition_function is not None:
                    sim_class = Moran2DStop
                else:
                    sim_class = Moran2D
            elif self.algorithm == Algorithm.BRANCHING:
                if self.end_condition_function is not None:
                    raise ValueError('Cannot use an end_condition_function for the branching algorithm')
                sim_class = SimpleBranchingProcess
            elif self.algorithm == Algorithm.WF2D:
                if self.end_condition_function is not None:
                    sim_class = WF2DStop
                else:
                    sim_class = WrightFisher2D

        if sim_class is None:
            raise ValueError("No simulation class defined")

        return sim_class

    def get_simulator(self):
        sim_class = self._select_simulator_class()
        if self.end_condition_function is not None:
            return sim_class(self, self.end_condition_function)
        else:
            return sim_class(self)
