from typing import Any
from simulation_algorithms.moran import MoranSim
from simulation_algorithms.moran2D import Moran2D
from simulation_algorithms.wf import WrightFisherSim
from simulation_algorithms.wf2D import WrightFisher2D
import numpy as np


class EndConditionError(Exception):
    pass


class StopConditionClass:
    """
    Class that is inherited to combine with the usual simulations.
    After every recorded time point, will check if the stop condition is met.
    The stop condition function passed to the class must raise an error that then stops the simulation
    """

    def __init__(self, parameters, stop_condition_function):
        """

        :param parameters: Parameters class object.
        :param stop_condition_function: An function that takes a simulation object as an argument and raises a
        StopConditionError if the simulation should stop.
        """
        self.stop_time: float | None = None
        self.stop_condition_result: Any | None = None   # Spare attribute to place any relevant result from the stopping
        self.stop_function = stop_condition_function
        super().__init__(parameters)

    def run_sim(self, continue_sim=False):
        try:
            super().run_sim(continue_sim)
        except EndConditionError as e:
            # Record the stop time
            self.stop_time = self.times[self.plot_idx-1]
            pass

    def _record_results(self, i, current_population, non_zero_clones):
        """
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :param non_zero_clones:
        :return:
        """
        if i == self.sample_points[self.plot_idx]:  # Regularly take a sample for the plot
            self._take_sample(current_population, non_zero_clones)
            self.stop_function(self)

        if self.progress:
            if i % self.progress == 0:
                print(i, end=', ', flush=True)


# Define the simulation classes. Will not work for the branching simulations or the differentiated cell simulations.

class WFStop(StopConditionClass, WrightFisherSim):
    pass


class WF2DStop(StopConditionClass, WrightFisher2D):
    pass


class MoranStop(StopConditionClass, MoranSim):
    pass


class Moran2DStop(StopConditionClass, Moran2D):
    pass


def stop_condition_first_occurrence(self):
    """
    An example of a stop condition function.

    Stops the simulation at the first instance of a clone type.
    Assume the mutant we care about is the last in the fitness array
    Records the time of the first instance
    :param self:
    :return:
    """
    if np.any(~np.isnan(self.raw_fitness_array[:, -1])):
        self.stop_condition_result = self.times[self.plot_idx-1]  # Save the last time point to stop_condition_result
        raise EndConditionError()
