from typing import Callable
import os
from pathlib import Path
from pydantic import (
    BaseModel,
    field_validator,
    Field,
)
from pydantic_settings import (
    BaseSettings,
    YamlConfigSettingsSource,
)
from simulation_algorithms.wf import WrightFisherSim
from simulation_algorithms.moran import MoranSim
from simulation_algorithms.moran2D import Moran2D
from simulation_algorithms.branching_process import SimpleBranchingProcess
from simulation_algorithms.wf2D import WrightFisher2D
from simulation_algorithms.general_differentiated_cell_class import Moran2DWithDiffcells, MoranWithDiffCells, BranchingWithDiffCells
from simulation_algorithms.stop_conditions import WFStop, WF2DStop, MoranStop, Moran2DStop
from clone_competition_simulation.parameters.times_validation import times_validation_type
from clone_competition_simulation.parameters.population_validation import population_validation_type
from clone_competition_simulation.parameters.fitness_validation import fitness_validation_type
from clone_competition_simulation.parameters.label_validation import label_validation_type
from clone_competition_simulation.parameters.treatment_validation import treatment_validation_type
from clone_competition_simulation.parameters.differentiated_cells_validation import differentiated_cells_validation_type
from clone_competition_simulation.parameters.plotting_validation import plotting_validation_type
from clone_competition_simulation.parameters.algorithm_validation import Algorithm, ALGORITHMS
from clone_competition_simulation.parameters.validation_utils import ValidationModelField, AlwaysValidateNoneField


class RunSettingsBase(BaseSettings):
    algorithm: Algorithm | None = None
    population: population_validation_type = ValidationModelField
    times: times_validation_type = ValidationModelField
    fitness: fitness_validation_type = ValidationModelField
    labels: label_validation_type = ValidationModelField
    treatment: treatment_validation_type = ValidationModelField
    differentiated_cells: differentiated_cells_validation_type = ValidationModelField
    plotting: plotting_validation_type = ValidationModelField

    end_condition_function: Callable | None = None
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


class SimulationRunSettings(RunSettingsBase, ConfigFileSettings):
    pass

    @property
    def non_zero_calc(self):
        """
        If the algorithm uses the entire current population to calculate the next step, this is set to true.
        Speeds up those simulations.
        Returns:

        """
        return not ALGORITHMS[self.algorithm].two_dimensional

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
