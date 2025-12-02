from typing import Annotated, Literal, Self
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag, BeforeValidator
)
from .algorithm_validation import AlgorithmClass
from .validation_utils import assign_config_settings, ValidationBase
from .population_validation import PopulationValidator
from loguru import logger


class TimeParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    times: np.ndarray | None = None
    max_time: float | None = None
    simulation_steps: int | None = None
    division_rate: float | None = None
    samples: int | None = None
    sample_times: np.ndarray | None = None


class TimeValidator(TimeParameters, ValidationBase):
    tag: Literal['Full']
    population: PopulationValidator
    config_file_settings: TimeParameters | None = None

    sample_points: np.ndarray | None = None

    def _validate_model(self):
        if self.times is not None:
            diff = np.diff(self.times)
            if np.any(diff <= 0):
                raise ValueError("Times must be in increasing order")
            if self.max_time is not None:
                if self.max_time != self.times[-1]:
                    raise ValueError(
                        "Max time does not match times given. Do not need to provide max_time if times provided.")
            else:
                self.max_time = self.times[-1]

        if self.simulation_steps is not None:
            if self.validation_category.algorithm_class == AlgorithmClass.BRANCHING:
                raise ValueError(
                    'Cannot specify number of simulations steps for the branching process algorithm.\n'
                    'Please provide a max_time and division_rate instead')
            if self.max_time is not None:
                if self.division_rate is not None:
                    # All defined, check they are consistent
                    sim_steps = self._calculate_simulation_steps()
                    if sim_steps != self.simulation_steps:  # Raise error if not consistent
                        st = 'Simulation_steps does not match max_time and division_rate.\n' \
                             'Provide only two of the three or ensure all are consistent.\n' \
                             'simulation_steps={0}, steps calculated from time and division rate={1}'.format(
                            self.simulation_steps, sim_steps
                        )
                        raise ValueError(st)

                else:
                    # simulation_steps and max_time given. Calculate division rate
                    self.division_rate = self._get_division_rate()
                    logger.debug(f'Division rate for the simulation is {self.division_rate}')
            else:
                if self.division_rate is None:
                    # Simulation steps defined but not max_time and division_rate
                    # Use the default division rate
                    self.division_rate = self.config_file_settings.division_rate
                    if self.division_rate is None:
                        raise ValueError("Division rate not defined. Define or set other time-related settings")
                # simulation_steps and division_rate given, calculate max_time
                self.max_time = self._get_max_time()
                logger.debug(f'Max time for the simulation is {self.max_time}')
        else:  # No simulation steps or time points defined. Calculate from max_time and division rate if given
            if self.division_rate is None:
                self.division_rate = self.config_file_settings.division_rate
                if self.division_rate is None:
                    raise ValueError("Division rate not defined. Define or set other time-related settings")
            if self.max_time is None:
                self.max_time = self.config_file_settings.max_time
                if self.max_time is None:
                    raise ValueError("Max time not defined. Define or set other time-related settings")
            self.simulation_steps = self._calculate_simulation_steps()

        self._check_samples()
        self._get_sample_times()

    def _calculate_simulation_steps(self):
        alg_class = self.validation_category.algorithm_class
        if alg_class == AlgorithmClass.MORAN:
            sim_steps = round(self.max_time * self.division_rate * self.population.initial_cells)
        elif alg_class == AlgorithmClass.WF:
            sim_steps = round(self.max_time * self.division_rate)
        elif alg_class == AlgorithmClass.BRANCHING:
            sim_steps = None
        else:
            raise ValueError('Calculation of sim_steps for {0} not implemented.'.format(alg_class))
        if sim_steps == 0:
            raise ValueError('Zero simulation steps for the given algorithm, max_time and division rate')
        return sim_steps

    def _get_max_time(self):
        alg_class = self.validation_category.algorithm_class
        if alg_class == AlgorithmClass.MORAN:
            max_time = self.simulation_steps/self.division_rate/self.population.initial_cells
        elif alg_class == AlgorithmClass.WF:
            max_time = self.simulation_steps / self.division_rate
        else:
            raise ValueError(
                f'Calculation of max_time for {self.algorithm} algorithm not implemented.'
                f' Please provide division_rate as an argument.')
        return max_time

    def _get_division_rate(self):
        alg_class = self.validation_category.algorithm_class
        if alg_class == AlgorithmClass.MORAN:
            div_rate = self.simulation_steps/self.max_time/self.population.initial_cells
        elif alg_class == AlgorithmClass.WF:
            div_rate = self.simulation_steps / self.max_time
        else:
            raise ValueError(f'Calculation of division rate for {self.algorithm} algorithm not implemented. '
                                  f'Please provide division_rate as an argument.')
        return div_rate

    def _check_samples(self):
        if self.times is None:
            if self.samples is None:
                self.samples = self.config_file_settings.samples
                if self.samples is None:
                    raise ValueError("Number of samples not defined. Define or set other time-related settings")
        else:
            self.samples = len(self.times)

        if self.simulation_steps is not None:
            if self.samples > self.simulation_steps:
                self.samples = self.simulation_steps
                if self.times is not None:
                    logger.debug('Fewer simulation steps than number of time points requested!')

    def _get_sample_times(self):
        if self.times is None:
            # The time points at each sample.
            self.times = np.linspace(0, self.max_time, self.samples + 1)

        if self.validation_category.algorithm_class != AlgorithmClass.BRANCHING:
            steps_per_unit_time = self.simulation_steps / self.max_time
            # Which points to take a sample
            sample_points_float = self.times * steps_per_unit_time
            self.sample_points = np.around(sample_points_float).astype(int)
            # Can sometimes have duplicates for close time points and slow division rate
            self.sample_points = np.unique(self.sample_points)
            if self.validation_category.algorithm_class == AlgorithmClass.MORAN:
                rounded_times = self.sample_points / self.division_rate / self.population.initial_cells
            else:
                rounded_times = self.sample_points / self.division_rate

            times_changed = False
            if len(self.times) != len(rounded_times):
                times_changed = True
            elif not np.allclose(self.times, rounded_times) > 0:
                times_changed = True
            if times_changed:
                logger.debug(f'Times rounded to match simulation steps:  Times used: {rounded_times}')
            self.times = rounded_times
        else:
            self.sample_points = None


times_validation_type = Annotated[
    (Annotated[TimeParameters, Tag("Base")] | Annotated[TimeValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
