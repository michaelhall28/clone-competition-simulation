"""
A class to run Moran-style simulations on a 2D hexagonal grid
"""

from clone_competition_simulation.wf import WrightFisherSim
from clone_competition_simulation.general_2D_class import GeneralHexagonalGridSim
import numpy as np


class WrightFisher2D(GeneralHexagonalGridSim, WrightFisherSim):
    """
    Runs a simulation of the clonal growth, mutation and competition
    On a hexagonal grid with periodic boundary conditions
    """

    def __init__(self, parameters):
        """

        :param parameters: a Parameters object containing the settings for the simulation
        """
        super().__init__(parameters)
        self.cell_in_own_neighbourhood = parameters.cell_in_own_neighbourhood
        self.grid = parameters.initial_grid.copy()
        self.grid_array = np.ravel(self.grid)
        self.grid_shape = parameters.grid_shape
        self.neighbour_map = self.make_base_array_edge_corrected()

        self.grid_results = [self.grid.copy()]

    def _sim_step(self, i, current_population, non_zero_clones):
        """
        A single step of the 2D Wright-Fisher process.
        At each step, we add mutations and then draw the new generation.
        Mutations are introduced at a certain rate.
        The number of mutations per step is drawn from a Poisson distribution
        The mutations are then assigned at random to any cells.
        The number of mutations at each generation is calculated prior to the main simulation starting.

        The next generation is drawn from the previous in proportion to the population size and the cell fitnesses
        The principle is the same as the non-spatial Wright-Fisher process, but the parent cells are restricted to the
        immediate neighbourhood of the offspring cells.
        :param i: Int. The simulation step number.
        :param current_population: Not used as input. Kept for consistency with the other algorithms.
        :param non_zero_clones: Not used as input. Kept for consistency with the other algorithms.
        :return:
        """
        # Get the number of mutations to add in this generation
        total_mutations = self.mutations_to_add[i]
        # Add those mutations and update the grid_array
        self._assign_mutations(total_mutations)

        # Draw the new generation of cells from the old generation.
        self._get_next_generation()

        # Record the current number of cells in each clone
        current_population = np.bincount(self.grid_array, minlength=len(self.clones_array))

        # Update the results grids if this is one of the sample times.
        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            self.grid = np.reshape(self.grid_array, (self.grid_shape))
            self.grid_results.append(self.grid.copy())

        return current_population, non_zero_clones

    def _assign_mutations(self, total_mutations):
        """
        Note: it is possible for a more than one mutation to be added to the same cell in the same generation.
        This would result in a clone added to the results with a zero population for the entire simulation, since
        as soon as it is added, the only cell of the clone is mutated again and moved to a new clone.
        :return:
        """
        if total_mutations > 0:
            coords = np.random.randint(self.total_pop, size=total_mutations)  # the positions of the parents

            unique, counts = np.unique(coords, return_counts=True)

            for i in range(1, counts.max() + 1):
                start_mut_index = self.next_mutation_index
                coords_i = unique[counts == i]
                parents_mut = self.grid_array[coords_i]  # cells getting exactly i new mutations
                self._draw_multiple_mutations_and_add_to_array(parents_mut)
                num_cells = len(parents_mut)

                if i == 1:
                    self.grid_array[coords_i] = range(start_mut_index, self.next_mutation_index)
                else:
                    current_parent_start = start_mut_index

                for j in range(2, i + 1):
                    next_parent_start = self.next_mutation_index

                    # Re-mutate the cells just mutated
                    self._draw_multiple_mutations_and_add_to_array(range(current_parent_start,
                                                                         current_parent_start + num_cells))
                    current_parent_start = next_parent_start
                    if j == i:
                        self.grid_array[coords_i] = range(current_parent_start, self.next_mutation_index)

    def _select_dividors(self, rel_weights, neighbour_clones):
        """
        Select the new cells.
        :param rel_weights: An array of the relative fitness values of the neighbourhood cells for every grid position.
        A row for each cell position in the grid, and a column for each neighbour position (6 or 7 columns)
        :param neighbour_clones: Array of the clone ids for all of the neighbour cells.
        Same dimensions as the rel_weights array.
        :return: Array. Length of the total number of cells in the grid.
        """
        # From Warren Weckesser on stack overflow
        # https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
        s = rel_weights.cumsum(axis=1)
        r = np.random.rand(self.total_pop, 1)
        k = (s < r).sum(axis=1)
        return neighbour_clones[np.arange(self.total_pop), k]

    def _get_next_generation(self):
        """
        Draw the new generation of cells.
        :return:
        """
        # Make an array of the clone ids of all cell neighbourhoods
        # A row for each cell position in the grid, and a column for each neighbour position (6 or 7 columns)
        neighbour_clones = self.grid_array[self.neighbour_map]
        ## Get the relative fitness of each cell in each of these cell neighbourhoods.
        # Get the fitness from self.clones_array
        weights = self.clones_array[neighbour_clones, self.fitness_idx]
        # Convert the fitness into relative fitness compared to the neighbourhood
        weights_sum = weights.sum(axis=1, keepdims=True)
        rel_weights = weights / weights_sum
        # Draw the new grid array. 
        self.grid_array = self._select_dividors(rel_weights, neighbour_clones)

