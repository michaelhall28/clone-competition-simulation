import numpy as np
from clone_competition_simulation.animator import HexAnimator


class GeneralHexagonalGridSim(object):
    # Contains functions that can be used with any hexagonal grid.

    def make_base_array_edge_corrected(self):
        depth, width = self.grid_shape
        in_own_neighbourhood = [0] * self.cell_in_own_neighbourhood
        even_col_base = list(np.array([-width - 1, -width, -width + 1, -1, 1, width])) + in_own_neighbourhood
        odd_col_base = list(np.array([-width, -1, +1, width - 1, width, width + 1])) + in_own_neighbourhood
        start_row_base = list(np.array([+width - 1, -width, -width + 1, -1, 1, width])) + in_own_neighbourhood
        end_row_base = list(np.array([-width, -1, +1, width - 1, width, -width + 1])) + in_own_neighbourhood
        full_row = [start_row_base] + [odd_col_base, even_col_base] * int((width - 2) / 2) + [end_row_base]
        base_array = full_row * int(depth)
        res = np.array(base_array) + np.arange(width * depth).reshape((width * depth, 1))
        res = np.mod(res, (width * depth))
        return res

    def _add_label(self, current_population, non_zero_clones, label_frequency, label, label_fitness, label_gene):
        """
        Add some labelling at the current label frequency.
        The labelling is not exact, so each cell has same chance.
        Apply the mutants to the grid.
        """
        num_labels = np.random.binomial(self.total_pop, label_frequency)

        # Extend the arrays
        self._extend_arrays_fixed_amount(num_labels)
        # Extend the current population with zeros
        current_population = np.concatenate([current_population, np.zeros(num_labels, dtype=int)])

        mutant_locs = np.random.choice(self.total_pop, num_labels, replace=False)

        for m in mutant_locs:
            # Convert the random draws into array indices
            parent = self.grid_array[m]
            current_population[self.next_mutation_index] += 1
            current_population[parent] -= 1
            self.grid_array[m] = self.next_mutation_index
            self._add_labelled_clone(parent, label, label_fitness, label_gene)

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        return current_population, non_zero_clones

    def get_1D_coord(self, row, col):
        """
        Convert a 2D grid coordinate into the index for the same cell in the 1D array
        return int
        """
        return row * self.grid_shape[0] + col

    def get_2D_coord(self, idx):
        """
        Convert an index from the 1D array to a location on the 2D grid.
        return tuple: (row, column)
        """
        return idx // self.grid_shape[0], idx % self.grid_shape[0]

    def get_neighbour_coords_2D(self, idx, col=None):
        """
        Get the neighbouring coordinates in the 2D grid from either the index in the 1D array or the coordinates in the
        2D grid.
        return array of ints
        """
        if col is not None:
            # Given the 2D coordinates
            idx = self.get_1D_coord(idx, col)

        neighbours = self.neighbour_map[idx]
        return np.array([self.get_2D_coord(n) for n in neighbours])

    def get_neighbour_coords_1D(self, idx, col=None):
        """
        Get the neighbouring indices in the 1D array from either the index in the 1D array or the coordinates in the
        2D grid.
        return array of ints
        """
        if col is not None:
            # Given the 2D coordinates
            idx = self.get_1D_coord(idx, col)

        return self.neighbour_map[idx]

    def get_neighbours(self, idx, col=None):
        """
        Returns the clone ids of the neighbouring cells.
        :param idx: The index of the cell in the 1D array, or the row of the cell in the 2D grid if using with col
        :param col: The column of the cell in the 2D grid. If None, will use idx alone to get the cell from the 1D array
        :return:
        """
        if col is not None:
            # Given the 2D coordinates
            idx = self.get_1D_coord(idx, col)
        return self.grid_array[self.neighbour_map[idx]]

    def plot_grid(self, t=None, index_given=False, grid=None, figsize=None, figxsize=5, bitrate=500, dpi=100,
                  equal_aspect=False, ax=None):
        """
        Plot a hexagonal grid of clones.
        The colours will be based on the colourscale defined in the Parameter object used to run the simulation.
        By default will plot the final grid of the simulation. Can pass a time point (or time index) or any 2D grid of
        the correct size (matching the size of the simulation grid).

        :param t: time or index of the sample to get the distribution for.
        :param index_given: True if t is the index. False if t is a time.
        :param grid: 2D numpy array of integers.
        """
        if grid is None:
            if t is None: # By default, plot the final grid
                grid = self.grid_results[-1]
            elif not index_given:
                grid = self.grid_results[self._convert_time_to_index(t)]
            else:
                grid = self.grid_results[t]

        # The plotting uses the HexAnimator class (same process to produce a single frame of an animation)
        animator = HexAnimator(self, figxsize=figxsize, figsize=figsize, dpi=dpi, bitrate=bitrate,
                               equal_aspect=equal_aspect)
        animator.plot_grid(grid, ax)
