from typing import Annotated, Literal
import numpy as np
from scipy.stats import expon
from pydantic import (
    ConfigDict,
    Tag,
    BeforeValidator
)
from .algorithm_validation import AlgorithmClass
from .validation_utils import assign_config_settings, ValidationBase, ParameterBase
from .times_validation import TimeValidator


class DifferentiatedCellsParameters(ParameterBase):
    """Parameters that control the simulation of differentiated cells (B cells).

    Fields:
        r:
            The proportion of each symmetric division type. This value
            must be between 0 and 0.5. When r=0, all divisions are asymmetric (one
            differentiated cell produces one differentiated and one progenitor cell).
            When r=0.5, 50% of divisions produce two differentiated cells and the 
            other 50% produce two progenitor cells.

            Example:
                r = 0.1  # 10% PP and 10% DD divisions
                r = 0.0  # All asymmetric divisions

        gamma:
            The stratification rate (for an exponential distribution), representing 
            the rate at which differentiated cells leave the basal layer. This must
            be greater than 0. Higher values mean faster differentiation.

            Example:
                gamma = 1.0  # Differentiation rate of 1 per unit time
                gamma = 0.5  # Slower differentiation

        stratification_sim_proportion:
            The target proportion (between 0 and 1) of differentiated cells that will
            be simulated and observed at the sample points in simulations. This is to
            avoid simulating unobserved differentiated cells that will be born and die 
            between sample points and therefore waste computation. A value of 1 means
            simulate all differentiated cells (conservative approach). Lower values
            speed up simulations by not simulating cells very unlikely to survive to
            observation. 

            Example:
                stratification_sim_proportion = 1.0  # Simulate all differentiated cells
                stratification_sim_proportion = 0.99  # Only simulate cells with 99% chance of surviving to observation
                stratification_sim_proportion = 0.90  # Only simulate cells with 90% chance of surviving to observation

    Examples:
        Basic differentiated cell simulation:
            r = 0.1
            gamma = 1.0
            stratification_sim_proportion = 1.0

        Faster simulation with partial differentiated cell tracking:
            r = 0.2
            gamma = 2.0
            stratification_sim_proportion = 0.8
    """
    _field_name = "differentiated_cells"
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    r: float | None = None
    gamma: float | None = None
    stratification_sim_proportion: float | None = None


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
        self.stratification_sim_proportion = self.get_value_from_config("stratification_sim_proportion")

        if self.r is not None or self.gamma is not None:
            if self.algorithm.algorithm_class not in (AlgorithmClass.MORAN, AlgorithmClass.BRANCHING):
                raise ValueError(
                    f'Cannot run {self.algorithm.algorithm_class} algorithms with B cells. '
                    f'Change algorithm or do not supply r or gamma arguments')
            if self.r is None:
                raise ValueError(
                    'Must provide both r and gamma to run with B cells. Please provide r')
            if self.gamma is None:
                raise ValueError(
                    'Must provide both r and gamma to run with B cells. Please provide gamma')

            if self.stratification_sim_proportion is None:
                # If not set explicitly, use the conservative value (simulate every cell).
                self.stratification_sim_proportion = self._default_strat_sim

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
        if self.stratification_sim_proportion < 1:
            min_diff_sim_time = expon.ppf(self.stratification_sim_proportion, scale=1 / self.gamma)
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
