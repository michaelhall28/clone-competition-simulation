"""
A class to run Moran-style simulations.
"""
import numpy as np
from numpy.typing import ArrayLike
from .base_sim_class import BaseSimClass, NonSpatialCurrentData
from ..utils import find_ge


class Moran(BaseSimClass):
    """
    Runs a simulation of the clonal growth, mutation and competition.
    It inherits most functions from GeneralSimClass
    """
    current_data_cls = NonSpatialCurrentData

    def __init__(self, parameters):

        super().__init__(parameters)

    def _adjust_raw_times(self, array: ArrayLike) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Takes an array of time points and converts to number of simulation steps
        This is for the Moran simulations. Overwrite for the other cases
        :param array: Numpy array or list of time points.
        """
        if array is not None:
            array = np.array(array) * self.division_rate * self.total_pop
        return array

    def _precalculate_mutations(self) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.int_]]]:
        """
        The timing of mutations that occur during the simulation can be calculated in advance.
        This speeds up the simulation a little.
        :return:
        """
        total_divisions = self.sample_points[-1]  # Length of the simulation
        mms = []
        self.mutation_rates[:, 0] = self.mutation_rates[:, 0] * self.division_rate * self.total_pop

        t_ends = list(self.mutation_rates[1:, 0]) + [total_divisions]


        for (t_start, mut_rate), t_end in zip(self.mutation_rates, t_ends):
            mms.append(np.random.poisson(mut_rate, int(t_end) - int(t_start)))  # integer

        mutations_to_add = np.concatenate(mms)
        new_mutation_count = mutations_to_add.sum()
        return new_mutation_count, mutations_to_add

    def _sim_step(self, i, current_data: NonSpatialCurrentData) -> NonSpatialCurrentData:
        """One cell is selected to die at random. Another cell is selected to replicate and replace the dead cell
        with its offspring. The replicating cell is selected in proportion with its relative fitness"""

        birth_idx = self.get_dividing_cell(current_data=current_data) # Clone to add a cell

        death_idx = self.get_differentiating_cell(current_data=current_data)  # Clone to remove a cell

        current_population, non_zero_clones = current_data.current_population, current_data.non_zero_clones

        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned a mutation
            new_muts = np.concatenate([[non_zero_clones[birth_idx]],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])
            # New mutation means extending the current_population.
            # Only have to add one clone to the current population. The rest with not be non-zero clones.
            current_population = np.concatenate([current_population, [1]])

            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation
            non_zero_clones = np.concatenate([non_zero_clones, [self.next_mutation_index - 1]])
        else:
            current_population[birth_idx] += 1
        current_population[death_idx] -= 1
        if current_population[death_idx] == 0:  # A clone has gone extinct. Remove from the current arrays
            current_population = np.concatenate([current_population[:death_idx], current_population[death_idx + 1:]])
            non_zero_clones = np.concatenate([non_zero_clones[:death_idx], non_zero_clones[death_idx + 1:]])

        current_data.update(
            current_population=current_population, 
            non_zero_clones=non_zero_clones
        )
        return current_data
    
    def get_dividing_cell(self, current_data: NonSpatialCurrentData) -> int:
        """Selects the clone that will divide in this simulation step

        This selects the clone based on the clone fitness and the number of cells in the clone. 

        Args:
            current_data (CurrentData): Contains the current clone cell populations and the indices of the living clones

        Returns:
            int: The index of the clone that will divide (index of the current_population array)
        """
        # Select population to replicate cell
        # Select random number to select which population
        birth_selector = np.random.random()
        
        # Select the living clones from the clones array, and just return the fitness values
        living_clone_fitness = self.clones_array[current_data.non_zero_clones, self.fitness_idx]  

        # make cumulative list of the fitnesses
        fitness_cumsum = np.cumsum(current_data.current_population * living_clone_fitness, axis=0)

        # Pick out the selected population
        # birth_idx is the index for the current population. The clone number is non_zero_clones[birth_idx]
        birth_idx = find_ge(fitness_cumsum, birth_selector * fitness_cumsum[-1])
        return birth_idx
    
    def get_differentiating_cell(self, current_data: NonSpatialCurrentData) -> int:
        """This selects the clone that will lose a cell in this simulation step 

        The cell is selected at random from the entire population (so the selection of the clone is proportional to the clone size)

        Args:
            current_data (CurrentData): Contains the current clone cell populations and the indices of the living clones

        Returns:
            int: The index of the clone that will lost a cell (index of the current_population array)
        """
        # Select replaced population
        # death_idx is the index for the current population. The clone number is non_zero_clones[death_idx]
        death_selector = np.random.random()
        cumsum = np.cumsum(current_data.current_population, axis=0)
        death_idx = find_ge(cumsum, death_selector * cumsum[-1])

        return death_idx

