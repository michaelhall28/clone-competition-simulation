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

from .algorithm_validation import Algorithm, AlgorithmClass
from .differentiated_cells_validation import differentiated_cells_validation_type
from .fitness_validation import fitness_validation_type
from .label_validation import label_validation_type
from .plotting_validation import plotting_validation_type
from .population_validation import population_validation_type
from .times_validation import times_validation_type
from .treatment_validation import treatment_validation_type
from .validation_utils import ValidationModelField, AlwaysValidateNoneField
from ..simulation_algorithms.branching_process import Branching
from ..simulation_algorithms.differentiated_cells import (
    Moran2DWithDiffcells, MoranWithDiffCells, BranchingWithDiffCells)
from ..simulation_algorithms.base_sim_class import BaseSimClass
from ..simulation_algorithms.moran import Moran
from ..simulation_algorithms.moran2D import Moran2D
from ..simulation_algorithms.wf import WF
from ..simulation_algorithms.wf2D import WF2D


class RunSettingsBase(BaseSettings):
    """Base settings for a simulation run.

    Aggregates all parameter validators for the simulation, including algorithm,
    population, timing, fitness, labels, treatments, differentiated cells, and
    plotting configuration.

    Attributes
    ----------
    algorithm : Algorithm | None
        The simulation algorithm to use (e.g., Moran, WF, Branching).
    population : population_validation_type
        Population parameters (e.g. number of cells, grid size)
    times : times_validation_type
        Simulation timing parameters.
    fitness : fitness_validation_type
        Fitness and mutation parameters.
    labels : label_validation_type
        Label parameters.
    treatment : treatment_validation_type
        Treatment parameters.
    differentiated_cells : differentiated_cells_validation_type
        Differentiated cell parameters.
    plotting : plotting_validation_type
        Plotting parameters.
    end_condition_function : Callable | None
        Optional custom function to determine early stopping conditions.
    tmp_store : Path | None
        Optional path for temporary storage of simulation state.
    """
    algorithm: Algorithm | None = None
    population: population_validation_type = ValidationModelField
    times: times_validation_type = ValidationModelField
    fitness: fitness_validation_type = ValidationModelField
    labels: label_validation_type = ValidationModelField
    treatment: treatment_validation_type = ValidationModelField
    differentiated_cells: differentiated_cells_validation_type = ValidationModelField
    plotting: plotting_validation_type = ValidationModelField

    end_condition_function: Callable[[BaseSimClass], None] | None = None
    tmp_store: Path | None = None


class ConfigFileSettings(BaseModel):
    """Load and validate settings from a YAML configuration file.

    Reads simulation parameters from a YAML file specified by path or
    environment variable, and converts them into the nested validator structure.

    Attributes
    ----------
    run_config_file : Path | None
        Path to the YAML configuration file. If None, checks the
        ``CCS_RUN_CONFIG`` environment variable.
    config_file_settings : RunSettingsBase | None
        Parsed settings from the configuration file.
    """
    run_config_file: Path | None = AlwaysValidateNoneField
    config_file_settings: RunSettingsBase | None = AlwaysValidateNoneField

    @field_validator("run_config_file", mode="before")
    @classmethod
    def validate_config_file_path(cls, value) -> Path | None:
        """Validate and resolve the configuration file path.

        Parameters
        ----------
        value : Path | None
            Provided path to configuration file.

        Returns
        -------
        Path | None
            Resolved configuration file path, or None if not specified.

        Notes
        -----
        Falls back to the ``CCS_RUN_CONFIG`` environment variable if no path
        is explicitly provided.
        """
        if value is not None:
            return value
        if env_path := os.environ.get('CCS_RUN_CONFIG', None):
            return Path(env_path)
        return None

    @field_validator("config_file_settings", mode="before")
    @classmethod
    def load_config_file_settings(cls, data, info):
        """Load and parse YAML configuration file settings.

        Parameters
        ----------
        data : Any
            Input data (unused; settings are loaded from file).
        info : ValidationInfo
            Validation context containing the run_config_file path.

        Returns
        -------
        dict
            Parsed settings from the YAML file, or empty dict if no file is provided.
        """
        run_config_file = info.data['run_config_file']
        if run_config_file is not None:
            y = YamlConfigSettingsSource(cls, yaml_file=info.data['run_config_file'])
            data = y()
        else:
            data = {}
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

def match_loc(loc: tuple[str, ...], match_pattern: tuple[str, ...]) -> bool:
    """Match a validation error location against a regex pattern.

    Used for filtering out or deduplicating validation errors based on their location in the nested model structure.

    Parameters
    ----------
    loc : tuple[str, ...]
        Location tuple from a Pydantic validation error (e.g., ``('population', 'Full', 'algorithm')``)
    match_pattern : tuple[str, ...]
        Pattern with regex strings to match against each location component.

    Returns
    -------
    bool
        True if all components of ``loc`` match the corresponding regex in
        ``match_pattern``, and the tuples have the same length.
    """
    if len(loc) != len(match_pattern):
        return False
    for l_pattern, m_pattern in zip(loc, match_pattern):
        if not re.match(m_pattern, l_pattern):
            return False
    return True


class Parameters(RunSettingsBase, ConfigFileSettings):
    """Central configuration class aggregating all simulation parameters.

    Combines base run settings and configuration file loading, with custom
    validation to filter and deduplicate errors from nested validators.

    Attributes
    ----------
    algorithm : Algorithm
        The selected simulation algorithm.
    population : population_validation_type
        Population configuration.
    times : times_validation_type
        Simulation timing.
    fitness : fitness_validation_type
        Fitness and mutations.
    labels : label_validation_type
        Labels configuration.
    treatment : treatment_validation_type
        Treatments configuration.
    differentiated_cells : differentiated_cells_validation_type
        B cell simulation configuration.
    plotting : plotting_validation_type
        Plotting settings.
    non_zero_calc : bool
        Read-only property. True if algorithm uses full population for calculations.
    """

    @model_validator(mode='wrap')
    @classmethod
    def clean_validation_errors(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Filter and clean validation errors from nested models.

        Removes errors from tag discrimination fields (which are internal) and
        deduplicates repeated errors from different validation sources for
        cleaner user-facing error messages.

        Parameters
        ----------
        data : Any
            Input data to validate.
        handler : ModelWrapValidatorHandler[Self]
            Pydantic validator handler.

        Returns
        -------
        Self
            Validated Parameters instance.

        Raises
        ------
        ValidationError
            If validation fails, with cleaned error messages.
        """
        try:
            model = handler(data)
            # Get any values from the config file that aren't in the nested models
            for field, field_info in cls.model_fields.items():
                if field == "run_config_file":  # Not in the config file
                    continue
                if field_info.json_schema_extra and field_info.json_schema_extra.get("config_validation") is not None:
                    # One of the sub-models. The parameters from the config file are dealt with in those models
                    continue
                if getattr(model, field) is not None:
                    # The fields is already defined by a value in the init
                    continue
                # If we got here, then we want to grab the value from the config file (if there is one)
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
        """Check if algorithm uses full population for step calculations.

        Returns
        -------
        bool
            True for non-2D algorithms (WF, Moran, Branching).
            False for 2D algorithms (WF2D, Moran2D).

        Notes
        -----
        Algorithms using full population calculations can be sped up by removing 
        zero-population clones from the calculation. 
        2D algorithms use the grid instead for populations calculations.
        """
        return not self.algorithm.two_dimensional

    def _select_simulator_class(self):
        """Select the appropriate simulator class based on algorithm and parameters.

        Chooses the simulator class based on the selected algorithm and whether
        differentiated cell (B cell) simulation is enabled.

        Returns
        -------
        type
            The simulator class (e.g., Moran, WF, Branching, etc.)

        Raises
        ------
        ValueError
            If differentiated cells are requested with an unsupported algorithm,
            or if an inconsistent algorithm/parameter combination is detected.
        """
        sim_class = None
        if self.differentiated_cells.diff_cell_simulation:  # Simulations including B cells.
            if self.algorithm.algorithm_class == AlgorithmClass.WF:
                raise ValueError(
                    'Cannot simulate differentiated cells for Wright-Fisher simulations')
            if self.algorithm == Algorithm.MORAN:
                sim_class = MoranWithDiffCells
            elif self.algorithm == Algorithm.MORAN2D:
                sim_class = Moran2DWithDiffcells
            elif self.algorithm == Algorithm.BRANCHING:
                sim_class = BranchingWithDiffCells
        else:
            if self.algorithm == Algorithm.WF:
                sim_class = WF
            elif self.algorithm == Algorithm.WF2D:
                sim_class = WF2D
            elif self.algorithm == Algorithm.MORAN:
                sim_class = Moran
            elif self.algorithm == Algorithm.MORAN2D:
                sim_class = Moran2D
            elif self.algorithm == Algorithm.BRANCHING:
                if self.end_condition_function is not None:
                    raise ValueError('Cannot use an end_condition_function for the branching algorithm')
                sim_class = Branching
            

        if sim_class is None:
            raise ValueError("No simulation class defined")

        return sim_class

    def get_simulator(self):
        """Instantiate and return a simulator for the configured parameters.

        Returns
        -------
        BaseSimClass
            A configured simulator instance ready to run the simulation.
        """
        sim_class = self._select_simulator_class()
        return sim_class(self)
