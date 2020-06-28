from clone_competition_simulation.general_sim_class import GeneralSimClass
import numpy as np
from scipy.sparse import lil_matrix


class OverPopulationError(BaseException): pass


class SimpleBranchingProcess(GeneralSimClass):
    """
    A simplified version of the single progenitor model
    Progenitor cells either divide to form two new progenitor cells or they die
    The chance of division/death is determined by the fitness of the clone
    Unlike the Wright-Fisher or Moran models, the total population size is not fixed.
    Each cell/clone acts independently of all other cells.
    """
    def __init__(self, parameters):

        self.time = 0
        super().__init__(parameters)
        self.population_limit = parameters.population_limit

        self.current_mutation_rate = self.mutation_rates[0][1]
        self.mutation_rate_idx = 0
        if self.mutation_rate_idx + 1 < len(self.mutation_rates):
            self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx+1][0]
        else:
            self.next_rate_change_time = np.inf

        self.new_mutations = {}  # Store clone_id and start time for each newly created clone.

    def _reset_to_start(self, start_time):
        # Settings such as treatment regime must be reset for each clone
        self.time = start_time
        self.current_mutation_rate = self.mutation_rates[0][1]
        self.mutation_rate_idx = 0
        if self.mutation_rate_idx + 1 < len(self.mutation_rates):
            self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx + 1][0]
        else:
            self.next_rate_change_time = np.inf
        self.plot_idx = 0

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

        self.current_fitness_multiplier = 1  # The effect of the current treatment
        self.treatment_count = -1
        if self.parameters.treatment_timings is None:
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
        # Functions which runs any of the simulation types.
        # self.sim_step will include the differences between the methods.
        # Each step can be a generation (Wright-Fisher) or a single birth-death-mutation event (Moran).

        # In it is possible for a clone to not survive until the next sample after being created by a mutation.
        # This will leave an all zeros row in the population array.
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

        self._change_treatment()

        for clone_id in range(len(self.clones_array)):
            self._run_for_clone(clone_id=clone_id, start_time=0)
            if self.population_limit is not None:
                total_pop = self.population_array[:, -1].sum()
                if total_pop > self.population_limit:
                    self.finished = True
                    raise OverPopulationError('Ending early as population limit exceeded')

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

        if self.progress:
            self.i = 1
            print('Finished')

        self._finish_up()
        self.finished = True

    def _run_for_clone(self, clone_id, start_time):
        self._reset_to_start(start_time)
        if clone_id < len(self.initial_size_array):
            current_population = self.initial_size_array[clone_id]  # One of the initial cells
            clone_sizes = [current_population]
            clone_times = []
        else:
            current_population = 1  # New mutations always start from one cell
            clone_sizes = [0, current_population]
            clone_times = [start_time]

        while self.time < self.max_time:
            current_population = self._sim_step(clone_id, current_population)  # Run step of the simulation
            clone_sizes.append(current_population)
            clone_times.append(self.time)

            while self._check_label_time():
                current_population = self._add_label(clone_id, current_population,
                                                     self.label_frequencies[self.label_count],
                                                     self.label_values[self.label_count],
                                                     self.label_fitness[self.label_count],
                                                     self.label_genes[self.label_count]
                                                     )

            while self._check_treatment_time():
                self._change_treatment()
            if current_population == 0:
                # The population can go extinct in this simulation. Must then stop the sim.
                self.time = self.max_time
                clone_times.append(self.time)
            elif self.population_limit is not None:
                if current_population > self.population_limit:
                    raise OverPopulationError("Ending early as single clone exceeded population limit")

        self._record_results(clone_id, clone_sizes, clone_times)

    def _sim_step(self, c, current_population):

        # Division rate is taken as r*lambda.
        # The rate of either a symmetric AA or BB division is then 2*r*lambda = 2*division_rate
        # This then matches with the Moran model.
        # This branching model required twice as many simulations steps as the Moran as the divisions and deaths
        # happen in different steps
        self.time += np.random.exponential(1 / (current_population * 2 * self.division_rate))

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
        if np.random.uniform(0, 2) <= self.clones_array[c, self.fitness_idx]:
            if self.current_mutation_rate > 0:
                new_muts = np.random.poisson(self.current_mutation_rate)
            else:
                new_muts = 0
            if new_muts > 0:
                if self.next_mutation_index + new_muts >= self.population_array.shape[0]:
                    self._extend_arrays(c, min_extension=self.population_array.shape[
                                                                             0] - self.next_mutation_index + new_muts)

                # Current population does not change. Record the existence of a new clone at this timepoint
                # and simulate later.
                parents = np.concatenate([[c], np.arange(self.next_mutation_index,
                                                         self.next_mutation_index + new_muts - 1)])

                self.plot_idx = np.searchsorted(self.times, self.time)  # Make sure the "generation_born" is correct
                self._draw_mutations_for_single_cell(parents)

                # Add the last mutation for simulation later
                self.new_mutations[self.next_mutation_index - 1] = self.time
            else:
                current_population += 1
        else:
            current_population -= 1

        return current_population

    def _record_results(self, c, clone_sizes, clone_times):
        """
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :return:
        """
        j = 0
        a = []
        for i, t in enumerate(self.times):
            while t > clone_times[j]:
                j += 1
            a.append(clone_sizes[j])

        self.population_array[c] = a

    def _extend_arrays(self, clone_id, min_extension=1):
        """
        We cannot pre-calculate the number of mutations (and therefore clones) as the population is not fixed
        so we must extend the arrays once they get full

        Base extension on the initial population, the mutation rate and the number of remaining clones to simulate.
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

        self.raw_fitness_array = np.concatenate([self.raw_fitness_array,
                                                 np.full((chunk_increase, self.raw_fitness_array.shape[1]), np.nan)],
                                                axis=0)

    def _add_label(self, clone_id, current_population, label_frequency, label, label_fitness, label_gene):
        # parent_idx, label, label_fitness, label_gene
        """
        Add some labelling at the current label frequency.
        The labelling is not exact, so each cell has same chance.
        Use a Poisson distribution of events for each clone.
        """
        self.plot_idx = np.searchsorted(self.times, self.time)  # Make sure the "generation_born" is correct
        marked_cells = np.random.binomial(current_population, label_frequency)
        if marked_cells + self.next_mutation_index >= len(self.clones_array):
            self._extend_arrays(current_population, min_extension=marked_cells)

        for cell in range(marked_cells):
            current_population -= 1
            self._add_labelled_clone(clone_id, label, label_fitness, label_gene)
            self.new_mutations[self.next_mutation_index - 1] = self.time

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        return current_population

    def _check_label_time(self):
        if not np.isinf(self.time) and self.time >= self.next_label_time:
            return True
        return False

    def _check_treatment_time(self):
        if not np.isinf(self.time) and self.time >= self.next_treatment_time:
            return True
        return False

    def _adjust_raw_times(self, array):
        if array is not None:
            array = np.array(array)
        return array

    def _finish_up(self):
        """
        Some of the plotting/post processing steps assume that all rows in the arrays are used in the simulation
        Remove rows that have not been used
        """
        self.clones_array = self.clones_array[:self.next_mutation_index]
        self.population_array = self.population_array[:self.next_mutation_index]
        self.raw_fitness_array = self.raw_fitness_array[:self.next_mutation_index]
