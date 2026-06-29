import gzip
from typing import TYPE_CHECKING

import dill as pickle
import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress

from .exceptions import EndConditionError

if TYPE_CHECKING:
    from .current_data import CurrentData


console = Console()


class SimulationLoopMixin:
    def run_sim(self, continue_sim: bool=False) -> None:
        """Main function for running simulations. Sets up the simulation and then runs through all
        the simulation steps. 

        Args:
            continue_sim (bool, optional): Continues a simulation from a previous state. 
             Defaults to False. Should run through sim.continue_sim() instead of running this function directly. 
        """
        if self.i > 0:
            # Not the first time it has been run
            if self.finished:
                print('Simulation already run')
                return
            elif continue_sim:
                print('Continuing from step', self.i)
            else:
                print('Simulation already started but incomplete')
                return

        # Set up the data to hold the current state of the simulation.
        # The state of this data will be recorded at each sample point
        current_data = self.current_data_cls.from_sim(self)

        # Change treatment if required (can change fitness of clones)
        if self._check_treatment_time():
            self._change_treatment(initial=True)

        # Add a label (similar to a lineage tracing label) if requested
        if self._check_label_time():
            current_data = self._add_label(current_data,
                                           self.label_frequencies[self.label_count],
                                           self.label_values[self.label_count],
                                           self.label_fitness[self.label_count],
                                           self.label_genes[self.label_count])

        with Progress(console=console, redirect_stdout=True, redirect_stderr=True) as progress:
            task = progress.add_task("Running simulation...", total=self.sim_length)
            try:
                while self.plot_idx < self.sim_length:
                    # Run step of the simulation
                    # Each step can be a generation (Wright-Fisher), a single birth-death-mutation event (Moran) or
                    # a single birth or death event (Branching)
                    current_data = self._sim_step(self.i, current_data)

                    self.i += 1
                    self._record_results(self.i, current_data, progress, task)  # Record the current state

                    # Add a label (similar to a lineage tracing label) if requested
                    if self._check_label_time():
                        current_data = self._add_label(current_data,
                                                       self.label_frequencies[self.label_count],
                                                       self.label_values[self.label_count],
                                                       self.label_fitness[self.label_count],
                                                       self.label_genes[self.label_count])

                    # Change treatment if required (can change fitness of clones)
                    if self._check_treatment_time():
                        self._change_treatment()

            except EndConditionError:
                # The simulation has stopped early because the end condition has been met
                # Record the stop time
                self.stop_time = self.times[self.plot_idx-1]
                progress.update(task, completed=self.plot_idx)
                logger.info(f'Simulation stopped early at time {self.stop_time} as end condition met')
                pass
            else:
                progress.update(task, completed=self.sim_length)

        # Clean up the results arrays
        self._finish_up()
        self.finished = True

    def continue_sim(self) -> None:
        """Continue a simulation from a previous state (e.g. saved to a pickle part way through a simulation). 
        Restores the random state.
        """
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        self.run_sim(continue_sim=True)

    def _sim_step(self, i: int, current_data: "CurrentData") -> "CurrentData":
        """Runs a single step of a simulation. Will vary depending on the algorithm. 
        Overwritten by each algorithm class.

        Args:
            i (int): the current simulation step.
            current_data (CurrentData): Object storing the current state of the simulation, e.g. 
             the current clone sizes or grid array. Will vary depending on the algorithm. 

        Returns:
            CurrentData: the current data updated following the simnulation step. 
        """
        raise NotImplementedError()

    def _finish_up(self) -> None:
        """
        Some of the algorithms may require some tidying up at the end,
        for example, removing unused rows in the arrays. Overwritten where relevant. 
        :return:
        """
        pass

    def _record_results(self, i: int, current_data: "CurrentData", progress: Progress, task: int) -> None:
        """
        Check if the current step is one of the sample points
        Record the results at the point the simulation is up to.
        Update the progress bar.

        :param i (int): The current simulation step.
        :param current_population (CurrentData):
        :param progress: The progress object for the progress bar. 
        :param task: The task number for the progress bar.
        :return:
        """
        if i == self.sample_points[self.plot_idx]:
            self._take_sample(current_data)
            if self.stop_function is not None:
                self.stop_function(self)
            progress.update(task, advance=1)

    def _take_sample(self, current_data: "CurrentData") -> None:
        """Store the current state of the simulation in the population array.  
        If storing partially completed simulation states, dump the 
        simulation to a pickle. 

        Parameters
        ----------
        current_data : CurrentData
            Current state of the simulation
        """
        current_data.update_population_array(self.population_array, self.plot_idx)
        self.plot_idx += 1
        if self.tmp_store is not None:
            if self.store_rotation == 0:
                self.pickle_dump(self.tmp_store)
                self.store_rotation = 1
            else:
                self.pickle_dump(str(self.tmp_store) + '1')
                self.store_rotation = 0

    def pickle_dump(self, filename: str) -> None:
        """Stores the simulation in a pickle file. 
        Stores the current random state so it can be restored when reloading. 

        Args:
            filename (str): The name of the file to store the pickle in.
        """
        self.random_state = np.random.get_state()
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=4)
