"""
A class to run Moran-style simulations on a 2D hexagonal grid
"""
from copy import deepcopy

import numpy as np

from .base_2D_class import (BaseHexagonalGridSim, SpatialCurrentData,
                            get_neighbour_map)
from .moran import Moran


class Moran2D(BaseHexagonalGridSim, Moran):
    """
    Runs a simulation of the clonal growth, mutation and competition.
    It inherits most functions from GeneralSimClass and MoranSim
    """
    current_data_cls = SpatialCurrentData

    def __init__(self, parameters):

        super().__init__(parameters)
        # The 2D grid for the simulation.
        # Copy in case same grid used for other simulations.
        initial_grid = deepcopy(parameters.population.initial_grid)  
                                                    
        self.grid_shape = parameters.population.grid_shape
        self.neighbour_map = get_neighbour_map(
            grid_shape=self.grid_shape,
            cell_in_own_neighbourhood=parameters.population.cell_in_own_neighbourhood
        )
        self.grid_results = [initial_grid]

        # Cell death is not dependent on fitness in this version of the Moran algorithm.
        # Can therefore calculate the positions of all the dying cells in advance to save time.
        self.death_coords = np.random.randint(0, self.total_pop, size=self.parameters.times.simulation_steps)

    def _sim_step(self, i: int, current_data: SpatialCurrentData) -> SpatialCurrentData:
        """Run one step of the simulation

        One cell is selected to die at random. 
        Another cell is selected to replicate and replace the dead cell
        with its offspring. The replicating cell is selected in 
        proportion with its relative fitness from the neighbourhood of 
        the dying cell

        Parameters
        ----------
        i : int
            Current step number
        current_data : SpatialCurrentData
            Current state of the simulation (i.e. the grid of cells)

        Returns
        -------
        SpatialCurrentData
            Updated state of the simulation
        """

        coord = self.get_differentiating_cell(i, current_data=current_data)
        birth_idx = self.get_dividing_cell(coord, current_data=current_data)
        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned at least one mutation
            new_muts = np.concatenate([[birth_idx],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])

            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation to the current population
            new_cell = self.next_mutation_index - 1
        else:
            new_cell = birth_idx

        grid_array = current_data.grid_array

        # Update the grid
        grid_array[coord] = new_cell

        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            grid = np.reshape(grid_array, self.grid_shape)
            self.grid_results.append(grid.copy())

        current_data.update(
            grid_array=grid_array
        )
        return current_data

    def get_differentiating_cell(self, i: int, current_data: SpatialCurrentData) -> int:
        """Selects the position of the cell to differentiate at step i

        These have been precalculated at the start of the simulation for efficiency.

        Parameters
        ----------
        i : int
            Current step number
        current_data : SpatialCurrentData
            Contains the current grid array. (Not used here, 
            but made available for any custom functions)

        Returns
        -------
        int
            the coordinate of the differentiating cell
        """
        coord = self.death_coords[i]
        return coord

    def get_dividing_cell(self, coord: int, current_data: SpatialCurrentData) -> int:
        """Select the cell that will divide to fill the gap left by self._random_death

        Parameters
        ----------
        coord : int
            Position of the dividing cell in the 1-D map of the grid.
        current_data : SpatialCurrentData
            Current state of the simulation (i.e. the grid_array)

        Returns
        -------
        int
            coord of the neighbouring dividing cell
        """
        # Get the indices of the neighbouring cells.
        neighbour_clones = current_data.grid_array[self.neighbour_map[coord]]

        # Get the fitness of those neighbouring cells
        weights = self.clones_array[neighbour_clones, self.fitness_idx]
        # Convert to relative fitness
        weights_sum = weights.sum()
        rel_weights = weights / weights_sum

        # Randomly select a neighbour, with the probabilities weighted by the relative fitness
        return neighbour_clones[np.random.multinomial(1, rel_weights).argmax()]
