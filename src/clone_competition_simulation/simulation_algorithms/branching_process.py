"""
A class for running a branching process of clone growth.
This is based on the single progenitor model (Clayton et al 2007).

For easier comparison with the Moran and Wright–Fisher models, 
the differentiated cells in the basal layer are not
simulated. A class including those cells is available in 
general_differentiated_cell_class.py (accessed by defining r and gamma 
when setting up the Parameters).
"""
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import lil_matrix
from rich.progress import Progress

from .base_sim_class import BaseSimClass

if TYPE_CHECKING:
    from ..parameters.parameter_validation import Parameters


class OverPopulationError(BaseException): pass


class Branching(BaseSimClass):
    """
    A simplified version of the single progenitor model
    Progenitor cells either divide to form two new progenitor cells or they die
    The chance of division/death is determined by the fitness of the clone
    Cell fitness does not alter the cell turnover rate.
    The maximum fitness is 2. This results in no cell death and a 
    cell division at every step, so all fitness values
    above 2 cannot increase the division probability any further.

    Unlike the Wright-Fisher or Moran models, the total population size is not fixed.
    Each cell acts independently of all other cells.

    Clones are simulated one at a time from the start to the end of the simulation time.
    This means you cannot stop the simulation early and get the full results up to a particular time point, it will be
    partial results (possibly up to the end of the simulation time) for some proportion of the cell population.

    """
    def __init__(self, parameters: "Parameters"):
        self.time = 0
        super().__init__(parameters)

        # Optional limit on the population size.
        # THIS DOES NOT ACT AS A CARRYING CAPACITY
        # This prevent overly long/memory-hungry simulations by stopping if a population limit is reached.
        # It will not necessarily stop at the simulated time the population limit is reached since clones
        # are simulated one at a time from start to end of the simulation.
        # If at any point the population_limit is reached, the simulation stops.
        self.population_limit = parameters.population.population_limit

        # Set the mutation rate at the start of the simulation.
        self.current_mutation_rate = self.mutation_rates[0][1]
        # Set up the next time when the mutation rate will change (if any).
        self.mutation_rate_idx = 0
        if self.mutation_rate_idx + 1 < len(self.mutation_rates):
            self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx+1][0]
        else:
            self.next_rate_change_time = np.inf

        self.new_mutations = {}  # Store clone_id and start time for each newly created clone.

    def _precalculate_mutations(self) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.int_]]]:
        """No precalculation for this algorithm

        We can't pre-calculate the number of mutations for this 
        simulation type since the cell population is not static. 
        Just return 0 and an empty array

        Returns
        -------
        tuple[int, np.ndarray[tuple[int], np.dtype[np.int_]]]
            Zero and an empty array
        """
        return 0, np.array([])

    def _reset_to_start(self, start_time: float) -> None:
        """Reset conditions to the birth time of a new clone

        Sets the conditions (e.g. mutation rate) for the birth time 
        of a new clone.
        This may be to the initial conditions of the simulation, 
        or if the clone was created by a mutation during
        the simulation, it will reset to the conditions at the birth 
        time of the clone.

        Parameters
        ----------
        start_time : float
            The time the clone was "born"
        """
        self.time = start_time

        # Set the correct mutation rate
        self.current_mutation_rate = self.mutation_rates[0][1]
        self.mutation_rate_idx = 0
        if self.mutation_rate_idx + 1 < len(self.mutation_rates):
            self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx + 1][0]
        else:
            self.next_rate_change_time = np.inf
        self.plot_idx = 0

        # Set the correct time for the next labelling event (if any)
        self.label_count = 0
        if self.label_times is not None:
            self.next_label_time = self.label_times[0]
            while self.time > self.next_label_time:
                self.label_count += 1
                if len(self.label_times) > self.label_count:
                    self.next_label_time = self.label_times[self.label_count]
                else:
                    self.next_label_time = np.inf
        else:
            self.next_label_time = np.inf

        # Set the correct treatment (will affect the fitness).
        self.current_fitness_multiplier = 1  # The effect of the current treatment
        self.treatment_count = -1
        if self.parameters.treatment.treatment_timings is None:
            self.treatment_timings = [0, np.inf]
            self.treatment_effects = [1, 1]  # Always neutral
            self.next_treatment_time = np.inf
        else:
            self.current_fitness_multiplier = self.treatment_effects[0]  # The effect of the current treatment
            self.next_treatment_time = self.treatment_timings[0]
            while self.time >= self.next_treatment_time:
                self.treatment_count += 1
                self.current_fitness_multiplier = self.treatment_effects[self.treatment_count]
                self.next_treatment_time = self.treatment_timings[self.treatment_count+1]
            self.clones_array[:, self.fitness_idx] = self._apply_treatment(self.raw_fitness_array,
                                                                           self.raw_fitness_array)

    def run_sim(self, continue_sim=False):
        """Run the simulation

        This overrides the method from GeneralSimClass because clones 
        are simulated one at a time, rather than simulating the 
        entire cell population from the start time to end time.

        Parameters
        ----------
        continue_sim : bool, optional
            Continues a simulation from a previous state. 
            Defaults to False. Should run through sim.continue_sim()
            instead of running this function directly. 

        Raises
        ------
        OverPopulationError
            Raised if the population limit is exceeded
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

        # Set up the initial treatment, if any
        self._change_treatment()

        with Progress() as progress:
            task = progress.add_task("Simulating initial clones", total=len(self.clones_array))
            # Simulate the clones that exist at the initiation of the simulation.
            for clone_id in range(len(self.clones_array)):
                self._run_for_clone(clone_id=clone_id, start_time=0)

                # Stop early if the population size has got too big.
                if self.population_limit is not None:
                    total_pop = self.population_array[:, -1].sum()
                    if total_pop > self.population_limit:
                        self.finished = True
                        raise OverPopulationError('Ending early as population limit exceeded')
                progress.update(task, advance=1)

        # Simulated of the clones that have been created by mutations during the simulation of the original clones.
        # Continue until all mutant clones have been simulated of the population limit has been exceeded.
        with Progress() as progress:
            task = progress.add_task("Simulating mutant clones", 
                                     total=len(self.clones_array) + len(self.new_mutations))
            while self.new_mutations:
                clone_id += 1
                start_time = self.new_mutations.pop(clone_id, None)
                if start_time is not None:
                    self._run_for_clone(clone_id, start_time)
                    if self.population_limit is not None:
                        total_pop = self.population_array[:, -1].sum()
                        if total_pop > self.population_limit:
                            self.finished = True
                            raise OverPopulationError('Ending early as population limit exceeded')
                        
                progress.update(task, completed=len(self.clones_array), 
                                total=len(self.clones_array) + len(self.new_mutations))
        
        # Tidy up the results arrays.
        self._finish_up()
        self.finished = True

    def _run_for_clone(self, clone_id: int, start_time: float):
        """Simulate an individual clone.

        Parameters
        ----------
        clone_id : int
            Id of the clone
        start_time : float
            Birth time of the clone

        Raises
        ------
        OverPopulationError
            Raised if the clone size exceeds the population limit
        """
        # Set conditions to those at the birth time of the clone.
        self._reset_to_start(start_time)

        # Set up the current clone size and the lists to record sizes over time
        if clone_id < len(self.initial_size_array):
            current_population = self.initial_size_array[clone_id]  # One of the initial cells
            clone_sizes = [current_population]
            clone_times = []
        else:
            current_population = 1  # New mutations always start from one cell
            clone_sizes = [0, current_population]
            clone_times = [start_time]

        while self.time < self.max_time:
            # Run until the next cell division (or death event).
            # Will update the clone size and self.time
            current_population = self._sim_step(clone_id, current_population)  # Run step of the simulation
            clone_sizes.append(current_population)
            clone_times.append(self.time)

            # Add any labels introduced
            while self._check_label_time():
                current_population = self._add_label(clone_id, current_population,
                                                     self.label_frequencies[self.label_count],
                                                     self.label_values[self.label_count],
                                                     self.label_fitness[self.label_count],
                                                     self.label_genes[self.label_count]
                                                     )

            # Adjust the treatment if it has changed.
            while self._check_treatment_time():
                self._change_treatment()

            if current_population == 0:
                # The population can go extinct in this simulation. Must then stop the sim.
                self.time = self.max_time
                clone_times.append(self.time)
            elif self.population_limit is not None:
                # Check if the clone is too large and the simulation must be stopped.
                if current_population > self.population_limit:
                    raise OverPopulationError("Ending early as single clone exceeded population limit")

        self._record_results(clone_id, clone_sizes, clone_times)

    def _sim_step(self, clone_id: int, current_population: int) -> int:
        """Run a step of the simulation

        A step of the simulation here is up until the next "division". 
        Division can produce two differentiated cells
        (not explicitly simulated here), so is effectively cell death.

        Parameters
        ----------
        clone_id : int
            Id of theclone
        current_population : int
            Current clone size

        Returns
        -------
        int
            Clone size after the sim step
        """

        # Division rate is taken as r*lambda.
        # The rate of either a symmetric AA or BB division is then 2*r*lambda = 2*division_rate
        # This then matches with the Moran model.
        # This branching model requires twice as many simulations steps as the Moran because the divisions and deaths
        # happen in different steps
        self.time += np.random.exponential(1 / (current_population * 2 * self.division_rate))

        # Update the mutation rate
        while self.time > self.next_rate_change_time:
            self.mutation_rate_idx += 1
            self.current_mutation_rate = self.mutation_rates[self.mutation_rate_idx][1]
            if self.mutation_rate_idx + 1 < len(self.mutation_rates):
                self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx + 1][0]
            else:
                self.next_rate_change_time = np.inf

        # Fitness=1 is balanced.
        # If a random draw from [0,2) is taken.
        # If the draw is above the fitness, the cell will die.
        # If the draw is below the fitness, the cell will divide (and possibly mutate).
        # This means the higher the fitness, the more the clone will proliferate.
        # Fitnesses above 2 are essentially infinite, the clone will not die.
        if self.does_cell_divide(clone_id):
            # The cell divides.
            # Add any new mutations to the new cell.
            if self.current_mutation_rate > 0:
                new_muts = np.random.poisson(self.current_mutation_rate)
            else:
                new_muts = 0
            if new_muts > 0:
                if self.next_mutation_index + new_muts >= self.population_array.shape[0]:
                    self._extend_arrays(clone_id, min_extension=self.population_array.shape[
                                                                             0] - self.next_mutation_index + new_muts)

                # Current population does not change. Record the existence of a new clone at this timepoint
                # and simulate later.
                parents = np.concatenate([[clone_id], np.arange(self.next_mutation_index,
                                                                self.next_mutation_index + new_muts - 1)])

                self.plot_idx = np.searchsorted(self.times, self.time)  # Make sure the "generation_born" is correct
                self._draw_mutations_for_single_cell(parents)

                # Add the last mutation for simulation later
                self.new_mutations[self.next_mutation_index - 1] = self.time
            else:
                current_population += 1
        else:
            # The cell dies (or divides to produce two differentiated cells).
            # Reduce the clone size by 1.
            current_population -= 1

        return current_population
    
    def does_cell_divide(self, clone_id: int) -> bool:
        """Determines whether the cell will divide or die. 

        Parameters
        ----------
        clone_id : int
             The id of the clone the cell belongs to

        Returns
        -------
        bool
            True if the cell will divide, False if it will die
        """
        # Fitness=1 is balanced.
        # If a random draw from [0,2) is taken.
        # If the draw is above the fitness, the cell will die.
        # If the draw is below the fitness, the cell will divide (and possibly mutate).
        # This means the higher the fitness, the more the clone will proliferate.
        # Fitnesses above 2 are essentially infinite, the clone will not die.
        return np.random.uniform(0, 2) <= self.clones_array[clone_id, self.fitness_idx]

    def _record_results(self, clone_id: int, clone_sizes: list[int], 
                        clone_times: list[float]) -> None:
        """Record the results at the point the simulation is up to.

        :param clone_id: Int.
        :param clone_sizes: List of clone sizes (integers)
        :param clone_times: List of times associated with the changes in clone size.
        :return:

        Parameters
        ----------
        clone_id : int
            ID of the clone
        clone_sizes : list[int]
            List of clone sizes
        clone_times : list[float]
            List of times associated with the changes in clone size.
        """
        j = 0
        a = []
        for i, t in enumerate(self.times):
            while t > clone_times[j]:
                j += 1
            a.append(clone_sizes[j])

        self.population_array[clone_id] = a

    def _extend_arrays(self, clone_id: int, min_extension=1) -> None:
        """Add more rows to the population, clones and raw fitness arrays
        
        We cannot pre-calculate the number of mutations (and therefore 
        clones) for this algorithm as the population is not fixed, 
        so we must extend the arrays once they get full.

        Base the extension size on the initial population, 
        the mutation rate and the number of remaining clones to simulate.

        Parameters
        ----------
        clone_id : int
            ID of the clone
        min_extension : int, optional
            Minimum number of rows to add, by default 1
        """
        remaining_clones = max(len(self.initial_size_array) - clone_id, 0) + len(self.new_mutations)
        starting_clones = len(self.initial_size_array)
        proportion_finished = remaining_clones / starting_clones

        chunk_increase = max(int(self.current_mutation_rate * self.division_rate *
                                 self.total_pop * proportion_finished + 1), min_extension)

        s = self.population_array.shape[0]
        new_pop_array = lil_matrix((s + chunk_increase, self.sim_length))
        new_pop_array[:s] = self.population_array
        self.population_array = new_pop_array

        self.clones_array = np.concatenate([self.clones_array, np.zeros((chunk_increase, 6))], axis=0)

        self.raw_fitness_array = np.concatenate([
            self.raw_fitness_array,
            np.full((chunk_increase, self.raw_fitness_array.shape[1]), np.nan)
            ], 
            axis=0
        )

    def _add_label(self, clone_id: int, current_population: int, 
                   label_frequency: float, label: int, 
                   label_fitness: float | None, 
                   label_gene_name: str | None) -> int:
        """Add some labelling at the current label frequency.

        The labelling is not exact, so each cell has the same chance.
        Each labelled cell is subsequently tracked as an independent clone. 

        Parameters
        ----------
        clone_id : int
            ID of the clone
        current_population : int
            Current number of cells in the clone
        label_frequency : float
            Proportion of cells that are labelled (on average)
        label : int
            Label
        label_fitness : float or None, 
            Fitness applied with the label
        label_gene_name : str or None
            Gene associated with the label

        Returns
        -------
        int
            New clone population after any lablelled cells are subtracted
        """
        self.plot_idx = np.searchsorted(self.times, self.time)  # Makes sure the "generation_born" is correct
        marked_cells = np.random.binomial(current_population, label_frequency)
        if marked_cells + self.next_mutation_index >= len(self.clones_array):
            self._extend_arrays(current_population, min_extension=marked_cells)

        for cell in range(marked_cells):
            current_population -= 1
            self._add_labelled_clone(clone_id, label, label_fitness, 
                                     label_gene_name)
            self.new_mutations[self.next_mutation_index - 1] = self.time

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        return current_population

    def _check_label_time(self) -> bool:
        """Check it is time to apply labels

        Returns
        -------
        bool
            True if time has reached the next label time. False otherwise. 
        """
        if not np.isinf(self.time) and self.time >= self.next_label_time:
            return True
        return False

    def _check_treatment_time(self) -> bool:
        """Check if it is time to change treatment.

        Returns
        -------
        bool
            True if the next treatment time has been reached. False otherwise. 
        """
        if not np.isinf(self.time) and self.time >= self.next_treatment_time:
            return True
        return False

    def _adjust_raw_times(self, array: ArrayLike) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert the time values to a NumPy array

        No further adjustments needed for this algorithm

        Parameters
        ----------
        array : ArrayLike
            List or array of times

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float64]]
            Times as a NumPy array
        """
        if array is not None:
            array = np.array(array)
        return array

    def _finish_up(self) -> None:
        """Remove unused rows from arrays
        
        Some of the plotting/post processing steps assume that all rows 
        in the arrays are used in the simulation, so remove rows that 
        have not been used
        """
        self.clones_array = self.clones_array[:self.next_mutation_index]
        self.population_array = self.population_array[:self.next_mutation_index]
        self.raw_fitness_array = self.raw_fitness_array[:self.next_mutation_index]
