"""
A class to run Moran-style simulations on a 2D hexagonal grid
"""
from .wf import WrightFisher
from .general_2D_class import GeneralHexagonalGridSim, get_neighbour_map, SpatialCurrentData
import numpy as np


class WrightFisher2D(GeneralHexagonalGridSim, WrightFisher):
    """
    Runs a simulation of the clonal growth, mutation and competition
    On a hexagonal grid with periodic boundary conditions
    """
    current_data_cls = SpatialCurrentData

    def __init__(self, parameters):
        """

        :param parameters: a Parameters object containing the settings for the simulation
        """
        super().__init__(parameters)
        initial_grid = parameters.population.initial_grid.copy()
        self.grid_array = np.ravel(initial_grid)
        self.grid_shape = parameters.population.grid_shape
        self.neighbour_map = get_neighbour_map(
            grid_shape=self.grid_shape,
            cell_in_own_neighbourhood=parameters.population.cell_in_own_neighbourhood
        )

        self.grid_results = [initial_grid]

    def _sim_step(self, i, current_data: SpatialCurrentData) -> SpatialCurrentData:
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
        :param current_data: The current state of the grid and population counts
        :return:
        """

        # Get the number of mutations to add in this generation
        total_mutations = self.mutations_to_add[i]
        # Add those mutations and update the grid_array
        current_data = self._assign_mutations(total_mutations, current_data=current_data)

        # Draw the new generation of cells from the old generation.
        current_data = self._get_next_generation(current_data=current_data)

        # Update the results grids if this is one of the sample times.
        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            self.grid = np.reshape(current_data.grid_array, (self.grid_shape))
            self.grid_results.append(self.grid.copy())

        return current_data

    def _assign_mutations(self, total_mutations, current_data: SpatialCurrentData) -> SpatialCurrentData:
        """
        Note: it is possible for a more than one mutation to be added to the same cell in the same generation.
        This would result in a clone added to the results with a zero population for the entire simulation, since
        as soon as it is added, the only cell of the clone is mutated again and moved to a new clone.
        :return:
        """
        grid_array = current_data.grid_array

        if total_mutations > 0:
            coords = np.random.randint(self.total_pop, size=total_mutations)  # the positions of the parents

            unique, counts = np.unique(coords, return_counts=True)

            for i in range(1, counts.max() + 1):
                start_mut_index = self.next_mutation_index
                coords_i = unique[counts == i]
                parents_mut = grid_array[coords_i]  # cells getting exactly i new mutations
                self._draw_multiple_mutations_and_add_to_array(parents_mut)
                num_cells = len(parents_mut)

                if i == 1:
                    grid_array[coords_i] = range(start_mut_index, self.next_mutation_index)
                else:
                    current_parent_start = start_mut_index

                for j in range(2, i + 1):
                    next_parent_start = self.next_mutation_index

                    # Re-mutate the cells just mutated
                    self._draw_multiple_mutations_and_add_to_array(range(current_parent_start,
                                                                         current_parent_start + num_cells))
                    current_parent_start = next_parent_start
                    if j == i:
                        grid_array[coords_i] = range(current_parent_start, self.next_mutation_index)

        current_data.update(grid_array=grid_array)
        return current_data

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

    def _get_next_generation(self, current_data: SpatialCurrentData) -> SpatialCurrentData:
        """
        Draw the new generation of cells.
        :return:
        """
        # Make an array of the clone ids of all cell neighbourhoods
        # A row for each cell position in the grid, and a column for each neighbour position (6 or 7 columns)
        neighbour_clones = current_data.grid_array[self.neighbour_map]
        ## Get the relative fitness of each cell in each of these cell neighbourhoods.
        # Get the fitness from self.clones_array
        weights = self.clones_array[neighbour_clones, self.fitness_idx]
        # Convert the fitness into relative fitness compared to the neighbourhood
        weights_sum = weights.sum(axis=1, keepdims=True)
        rel_weights = weights / weights_sum
        # Draw the new grid array. 
        new_grid_array = self._select_dividors(rel_weights, neighbour_clones)

        current_data.update(grid_array=new_grid_array)

        return current_data

