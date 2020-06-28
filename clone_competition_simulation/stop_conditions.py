from clone_competition_simulation.general_sim_class import GeneralSimClass
from clone_competition_simulation.general_2D_class import GeneralHexagonalGridSim
from clone_competition_simulation.moran import MoranSim
from clone_competition_simulation.moran2D import Moran2D
from clone_competition_simulation.wf import WrightFisherSim
from clone_competition_simulation.wf2D import WrightFisher2D
from clone_competition_simulation.branching_process import SimpleBranchingProcess
from clone_competition_simulation.general_differentiated_cell_class import BranchingWithDiffCells, MoranWithDiffCells, Moran2DWithDiffcells
import numpy as np


class EndConditionError(Exception):
    pass


class StopConditionClass:
    # After every recorded time point, will check if the end condition is met.
    # The end condition function passed to the class must raise an error that then stops the simulation

    def __init__(self, parameters, end_condition_function):
        self.end_condition = None
        self.end_function = end_condition_function
        super().__init__(parameters)

    def run_sim(self, continue_sim=False):
        try:
            super().run_sim(continue_sim)
        except EndConditionError as e:
            pass

    def _record_results(self, i, current_population, non_zero_clones):
        """
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :return:
        """
        if i == self.sample_points[self.plot_idx]:  # Regularly take a sample for the plot
            self._take_sample(current_population, non_zero_clones)
            self.end_function(self)

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


def end_condition_first_occurrence(self):
    # Stops at first instance of a clone type.
    # Assume the mutant we care about is the last in the fitness array
    # Records the time of the first instance
    if np.any(~np.isnan(self.raw_fitness_array[:, -1])):
        self.end_condition = self.times[self.plot_idx-1]   # Save the last time point to the end_condition attribute
        raise EndConditionError()
