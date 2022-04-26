"""
A class to run non-spatial Wright-Fisher-style simulations
"""

from clone_competition_simulation.general_sim_class import GeneralSimClass
import numpy as np


class WrightFisherSim(GeneralSimClass):
    """Runs a simulation of the clonal growth, mutation and competition"""
    def __init__(self, parameters):
        """

        :param parameters: a Parameters object containing the settings for the simulation
        """
        super().__init__(parameters)

    def _adjust_raw_times(self, array):
        """Takes an array of time points and converts to number of simulation steps"""
        if array is not None:
            array = np.array(array) * self.division_rate

        return array

    def _sim_step(self, i, current_population, non_zero_clones):
        """
        A single step of the Wright-Fisher process.
        At each step, we add mutations and then draw the new generation.
        Mutations are introduced at a certain rate.
        The number of mutations per step is drawn from a Poisson distribution
        The mutations are then assigned at random to any cells.
        The number of mutations at each generation is calculated prior to the main simulation starting.

        The next generation is drawn from the previous in proportion to the population size and the cell fitnesses
        We draw from a multinomial distribution. Selection of N from a list of probabilities
        The probability of each clone is the clone population * clone fitness / sum(c' pop * c' fitness)
        where the sum is over all clones
        :param i: Int. The simulation step number.
        :param current_population: Array of clone sizes. Only for the clones with at least 1 cell.
        :param non_zero_clones: Array of the clone numbers of the current clones that have at least 1 cell.
        :return:
        """
        # Get the number of mutations to add during this generation.
        total_mutations = self.mutations_to_add[i]
        # Update the cell population with these new mutations.
        current_population, non_zero_clones = self._assign_mutations(total_mutations,
                                                                     current_population,
                                                                     non_zero_clones)

        # Draw the new generation of cells from the old generation.
        # First, calculate the relative weight of each clone (population size multiplied by the fitness)
        weights = current_population * self.clones_array[non_zero_clones, self.fitness_idx]
        relative_weights = weights / weights.sum()
        # Then draw the new population.
        current_population = np.random.multinomial(self.total_pop, relative_weights)
        gr_z = np.nonzero(current_population > 0)[0]  # The indices of clones alive at this point in the current pop
        non_zero_clones = non_zero_clones[gr_z]  # Convert those indices to the original clone numbers
        current_population = current_population[gr_z]  # Only keep the currently alive clones in current pop
        return current_population, non_zero_clones

    def _precalculate_mutations(self):
        """Before the full simulation starts, we can simulation the number of mutations introduced in each generation
        We can then make the arrays the full size at the start."""
        generations = self.sample_points[-1]  # Length of the simulation
        self.mutations_to_add = []
        self.new_mutation_count = 0
        # Convert the time in the mutation_rates array to a simulation step
        self.mutation_rates[:, 0] = self.mutation_rates[:, 0] * self.division_rate
        mut_rate_idx = 0
        mutation_rate = self.mutation_rates[mut_rate_idx][1]
        if mut_rate_idx + 1 < len(self.mutation_rates):
            next_rate_change_time = self.mutation_rates[mut_rate_idx+1][0]
        else:
            next_rate_change_time = np.inf

        # Loop through the simulation steps (one step=one generation of cells) and assign a mutation count to each
        for i in range(generations):
            # Update the mutation rate if needed.
            if i >= next_rate_change_time:
                mut_rate_idx += 1
                mutation_rate = self.mutation_rates[mut_rate_idx][1]
                if mut_rate_idx + 1 < len(self.mutation_rates):
                    next_rate_change_time = self.mutation_rates[mut_rate_idx + 1][0]
                else:
                    next_rate_change_time = np.inf

            # Randomly draw the number of mutations for this simulation step
            total_mutations = self._calc_num_mutations(mutation_rate)
            # Track the total amount of new mutations in the whole simulation (used to make the results arrays)
            self.new_mutation_count += total_mutations
            # And store the number of mutations for this sim step.
            self.mutations_to_add.append(total_mutations)

    def _calc_num_mutations(self, mutation_rate):
        """
        Draws the number of mutations from a poisson distribution
        :param mutation_rate: Mutation rate per cell per generation
        :return: Int.
        """
        total_mutations = np.random.poisson(mutation_rate * self.total_pop)
        return total_mutations

    def _assign_mutations(self, total_mutations, current_population, non_zero_clones):
        """
        Adds new mutations to cells.

        Note: it is possible for a more than one mutation to be added to the same cell in the same generation.
        This would result in a clone added to the results with a zero population for the entire simulation, since
        as soon as it is added, the only cell of the clone is mutated again and moved to a new clone.
        :return:
        """
        if total_mutations > 0:
            # List out all cells and the clone they belong to.
            flattened_pop = np.repeat(np.arange(current_population.size), current_population)
            # Randomly select the cells to mutate (can have more than one per cell)
            coords = np.random.randint(self.total_pop, size=total_mutations)  # the positions of the parents
            unique, counts = np.unique(coords, return_counts=True)  # Count the mutations per cell.

            # Remove mutated cells from their parent clones. Will be added as new clones below.
            u1, c1 = np.unique(flattened_pop[unique], return_counts=True)
            current_population[u1] -= c1

            new_surviving_clone_numbers = []
            for i in range(1, counts.max() + 1):
                start_mut_index = self.next_mutation_index
                parents_mut = flattened_pop[unique[counts == i]]  # cells getting exactly i new mutations
                parents_mut = non_zero_clones[parents_mut]
                self._draw_multiple_mutations_and_add_to_array(parents_mut)
                num_cells = len(parents_mut)

                if i == 1:
                    new_surviving_clone_numbers.extend(range(start_mut_index, self.next_mutation_index))
                else:
                    current_parent_idx = start_mut_index

                for j in range(2, i + 1):
                    next_parent_idx = self.next_mutation_index

                    # Re-mutate the cells just mutated
                    self._draw_multiple_mutations_and_add_to_array(
                        range(current_parent_idx, current_parent_idx + num_cells))
                    current_parent_idx = next_parent_idx
                    if j == i:
                        new_surviving_clone_numbers.extend(range(current_parent_idx, self.next_mutation_index))

            current_population = np.concatenate([current_population, np.ones(len(unique))])

            non_zero_clones = np.concatenate([non_zero_clones, new_surviving_clone_numbers])

        return current_population, non_zero_clones
