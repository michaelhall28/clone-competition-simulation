from typing import Annotated, Literal
import numpy as np
from scipy.stats import expon
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag,
    BeforeValidator
)
from .algorithm_validation import AlgorithmClass
from .validation_utils import assign_config_settings, ValidationBase
from .times_validation import TimeValidator


class DifferentiatedCellsParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    r: float | None = None
    gamma: float | None = None
    stratification_sim_percentile: float | None = None


class DifferentiatedCellsValidator(DifferentiatedCellsParameters, ValidationBase):
    _default_strat_sim = 1
    tag: Literal['Full']
    times: TimeValidator
    config_file_settings: DifferentiatedCellsParameters | None = None

    diff_cell_sim_switches: np.ndarray | None = None
    diff_sim_starts: np.ndarray | None = None  # Times to simulated differentiated cells (if at all)
    diff_sim_ends: np.ndarray | None = None

    diff_cell_simulation: bool = False

    def _validate_model(self):
        self.r = self.get_value_from_config("r")
        self.gamma = self.get_value_from_config("gamma")
        self.stratification_sim_percentile = self.get_value_from_config("stratification_sim_percentile")

        if self.r is not None or self.gamma is not None:
            if self.algorithm.algorithm_class not in (AlgorithmClass.MORAN, AlgorithmClass.BRANCHING):
                raise ValueError(
                    'Cannot run {} algorithms with B cells. Change algorithm or do not supply r or gamma arguments'.format(
                        self.algorithm.algorithm_class))
            if self.r is None:
                raise ValueError(
                    'Must provide both r and gamma to run with B cells. Please provide r')
            if self.gamma is None:
                raise ValueError(
                    'Must provide both r and gamma to run with B cells. Please provide gamma')

            if self.stratification_sim_percentile is None:
                # If not set explicitly, use the conservative value (simulate every cell).
                self.stratification_sim_percentile = self._default_strat_sim

            if self.r > 0.5 or self.r < 0:
                raise ValueError('Must have 0<=r<=0.5')
            if self.gamma <= 0:
                raise ValueError('gamma must be > 0')

            self._get_diff_cell_simulation_times()
            self.diff_cell_simulation = True

    def _get_diff_cell_simulation_times(self):
        """
        Need to simulate for a period prior to the sample time.
        Use a specified percentile of the exponential distribution of stratification times
        to find the minimum time to simulate.
        For the Moran models, find the last simulation step prior to this minimum time before the sample point.
        For the Branching process, the times can be compared to the time of the next progenitor division
        """
        if self.stratification_sim_percentile < 1:
            min_diff_sim_time = expon.ppf(self.stratification_sim_percentile, scale=1 / self.gamma)
            if self.algorithm.algorithm_class == AlgorithmClass.BRANCHING:
                diff_sim_starts = self.times.times - min_diff_sim_time
                diff_sim_starts[diff_sim_starts < 0] = 0
                self.diff_cell_sim_switches = self._merge_time_intervals(diff_sim_starts, self.times.times) + [np.inf]
            else:
                steps_per_unit_time = self.times.simulation_steps / self.times.max_time
                sim_steps_for_diff_sims = min_diff_sim_time * steps_per_unit_time
                sim_steps_to_start_diff = (self.times.sample_points - sim_steps_for_diff_sims).astype(int)
                sim_steps_to_start_diff[sim_steps_to_start_diff < 0] = 0
                self.diff_cell_sim_switches = self._merge_time_intervals(sim_steps_to_start_diff,
                                                                         self.times.sample_points)
        else:
            self.diff_cell_sim_switches = [0, np.inf]

    def _merge_time_intervals(self, starts, ends):
        merged_starts = [starts[0]]
        merged_ends = []
        gaps = starts[1:] - ends[:-1]
        for i, g in enumerate(gaps):
            if g <= 0:
                # No gap. Merged with next section
                pass
            else:
                merged_ends.append(ends[i])
                merged_starts.append(starts[i + 1])
        merged_ends.append(ends[-1])
        all_events = sorted(merged_starts + merged_ends)
        return all_events


differentiated_cells_validation_type = Annotated[
    (Annotated[DifferentiatedCellsParameters, Tag("Base")] | Annotated[DifferentiatedCellsValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
