import numpy as np


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
