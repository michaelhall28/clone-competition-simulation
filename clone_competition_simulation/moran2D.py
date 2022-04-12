"""
A class to run Moran-style simulations on a 2D hexagonal grid
"""

import numpy as np
from clone_competition_simulation.moran import MoranSim
from clone_competition_simulation.general_2D_class import GeneralHexagonalGridSim


class Moran2D(GeneralHexagonalGridSim, MoranSim):
    """
    Runs a simulation of the clonal growth, mutation and competition.
    It inherits most functions from GeneralSimClass and MoranSim
    """
    def __init__(self, parameters):

        super().__init__(parameters)
        self.cell_in_own_neighbourhood = parameters.cell_in_own_neighbourhood
        self.grid = parameters.initial_grid.copy()  # The 2D grid for the simulation.
                                                    # Copy in case same grid used for other simulations.
        self.grid_shape = parameters.grid_shape
        self.grid_array = np.ravel(self.grid)
        self.grid_shape = parameters.grid_shape
        self.neighbour_map = self.make_base_array_edge_corrected()
        self.grid_results = [self.grid.copy()]
        self.grid_results = [self.grid.copy()]

        # Cell death is not dependent on fitness in this version of the Moran algorithm.
        # Can therefore calculate the positions of all the dying cells in advance to save time.
        self.death_coords = np.random.randint(0, self.total_pop, size=self.parameters.simulation_steps)

    def _sim_step(self, i, current_population, non_zero_clones):

        death_idx, coord = self._random_death(i)
        birth_idx = self._get_divider(coord)
        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned a mutation
            new_muts = np.concatenate([[birth_idx],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])

            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation
            new_cell = self.next_mutation_index - 1
        else:
            new_cell = birth_idx

        current_population[new_cell] += 1
        current_population[death_idx] -= 1
        self.grid_array[coord] = new_cell

        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            grid = np.reshape(self.grid_array, self.grid_shape)
            self.grid_results.append(grid.copy())

        return current_population, non_zero_clones

    def _random_death(self, i):
        coord = self.death_coords[i]
        cell = self.grid_array[coord]
        return cell, coord

    def _get_divider(self, coord):
        neighbour_clones = self.grid_array[self.neighbour_map[coord]]
        weights = self.clones_array[neighbour_clones, self.fitness_idx]
        weights_sum = weights.sum()
        rel_weights = weights / weights_sum
        return neighbour_clones[np.random.multinomial(1, rel_weights).argmax()]
