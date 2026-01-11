"""
A class to run Moran-style simulations on a 2D hexagonal grid
"""

import numpy as np
from .moran import MoranSim
from .general_2D_class import GeneralHexagonalGridSim, get_neighbour_map


class Moran2D(GeneralHexagonalGridSim, MoranSim):
    """
    Runs a simulation of the clonal growth, mutation and competition.
    It inherits most functions from GeneralSimClass and MoranSim
    """
    def __init__(self, parameters):

        super().__init__(parameters)
        self.grid = parameters.population.initial_grid.copy()  # The 2D grid for the simulation.
                                                    # Copy in case same grid used for other simulations.
        self.grid_shape = parameters.population.grid_shape
        self.grid_array = np.ravel(self.grid)
        self.neighbour_map = get_neighbour_map(
            grid_shape=self.grid_shape,
            cell_in_own_neighbourhood=parameters.population.cell_in_own_neighbourhood
        )
        self.grid_results = [self.grid.copy()]

        # Cell death is not dependent on fitness in this version of the Moran algorithm.
        # Can therefore calculate the positions of all the dying cells in advance to save time.
        self.death_coords = np.random.randint(0, self.total_pop, size=self.parameters.times.simulation_steps)

    def _sim_step(self, i, current_population, non_zero_clones):

        death_idx, coord = self._random_death(i)
        birth_idx = self._get_divider(coord)
        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned at least one mutation
            new_muts = np.concatenate([[birth_idx],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])

            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation to the current population
            new_cell = self.next_mutation_index - 1
        else:
            new_cell = birth_idx

        # Update the clone sizes
        current_population[new_cell] += 1
        current_population[death_idx] -= 1
        # Update the grid
        self.grid_array[coord] = new_cell

        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            grid = np.reshape(self.grid_array, self.grid_shape)
            self.grid_results.append(grid.copy())

        return current_population, non_zero_clones

    def _random_death(self, i):
        """
        Returns the clone_id and coordinate of the cell to die at step i.
        These have been precalculated at the start of the simulation for efficiency.
        :param i: Int. the simulation step
        :return: Tuple. (clone_id (int), coordinate in 1-D map of grid (int))
        """
        coord = self.death_coords[i]
        cell = self.grid_array[coord]
        return cell, coord

    def _get_divider(self, coord):
        """
        Selects the cell that will divide to fill the gap left by self._random_death
        :param coord: Position of the dividing cell in the 1-D map of the grid.
        :return: coord of the neighbouring dividing cell (int).
        """
        # Get the indices of the neighbouring cells.
        neighbour_clones = self.grid_array[self.neighbour_map[coord]]

        # Get the fitness of those neighbouring cells
        weights = self.clones_array[neighbour_clones, self.fitness_idx]
        # Convert to relative fitness
        weights_sum = weights.sum()
        rel_weights = weights / weights_sum

        # Randomly select a neighbour, with the probabilities weighted by the relative fitness
        return neighbour_clones[np.random.multinomial(1, rel_weights).argmax()]
