import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .current_data import CurrentData


class MutationManagerMixin:
    def _check_treatment_time(self) -> bool:
        """Returns True if it is time to switch treatment, False otherwise"""
        return self.i >= self.next_treatment_time

    def _change_treatment(self, initial: bool=False) -> None:
        """Switches to the next treatment and updates the fitness of all clones accordingly. 
        Also used at the start of the simulation to calculate the initial fitness of clones."""
        self.treatment_count += 1
        self.current_fitness_multiplier = self.treatment_effects[self.treatment_count]
        self.next_treatment_time = self.treatment_timings[self.treatment_count + 1]
        if initial:
            if self.fitness_calculator and self.fitness_calculator.multi_gene_array:
                self.clones_array[:self.initial_clones, self.fitness_idx] = self._apply_treatment(
                    fitness_arrays=self.raw_fitness_array[:self.initial_clones])
            else:
                self.clones_array[:self.initial_clones, self.fitness_idx] = self._apply_treatment(
                    fitness_values=self.raw_fitness_array[:self.initial_clones])
        else:
            if self.fitness_calculator and self.fitness_calculator.multi_gene_array:
                self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_arrays=self.raw_fitness_array)
            else:
                self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_values=self.raw_fitness_array)

    def _apply_treatment(self, fitness_values=None, fitness_arrays=None):
        # Apply the treatment affects to an array of fitnesses
        # fitness_values is 1D array of overall fitness for each clone
        # fitness_arrays is a 2D array with one row per clone and one column per gene plus a column for wild type
        if self.fitness_calculator and self.fitness_calculator.multi_gene_array:
            # Apply the treatment to the genes, then calculate the overall fitness.
            if not self.treatment_replace_fitness:
                # Multiply the fitness by the treatment effect
                adjusted_fitness_array = fitness_arrays * self.current_fitness_multiplier
                combined_fitness_array, _ = self.fitness_calculator.combine_vectors(adjusted_fitness_array)
            else:
                # Replace the fitness with the new fitness effect
                adjusted_fitness_array = np.tile(self.current_fitness_multiplier, (len(fitness_arrays), 1))
                adjusted_fitness_array[np.isnan(fitness_arrays)] = np.nan  # Leave all unmutated genes unmutated
                
                # Leave all untreated genes (nan multiplier) as they were.
                adjusted_fitness_array[:, np.isnan(self.current_fitness_multiplier)] = \
                    fitness_arrays[:,np.isnan(self.current_fitness_multiplier)]
                combined_fitness_array, _ = self.fitness_calculator.combine_vectors(adjusted_fitness_array)
            return combined_fitness_array
        else:
            # Applies per clone.
            if not self.treatment_replace_fitness:
                # Multiply the overall fitness of each clone.
                return fitness_values.T * self.current_fitness_multiplier
            else:
                # Replace fitness per clone with the new fitness (if not nan).
                new_fitness_values = self.current_fitness_multiplier
                new_fitness_values[np.isnan(new_fitness_values)] = fitness_values[np.isnan(new_fitness_values), 0]
                return new_fitness_values

    def _check_label_time(self):
        return self.i >= self.next_label_time

    def _add_label(self, current_data: "CurrentData", label_frequency: float, label: int,
                   label_fitness: float, label_gene_name: str | None) -> "CurrentData":
        """
        Add some labelling at the current label frequency.
        The labelling is not exact, so each cell has same chance.
        Use a Poisson distribution of events for each clone.
        """
        # Random draw for each clone base on clone size
        labels_per_clone = np.random.binomial(current_data.current_population, label_frequency)
        assert not np.any(labels_per_clone - current_data.current_population > 0), (labels_per_clone, current_data.current_population)

        num_labels = np.sum(labels_per_clone)
        self._extend_arrays_fixed_amount(num_labels)

        # Add the new clones to the current population. Cells will be removed from the old clones below.
        current_population = np.concatenate([current_data.current_population, np.ones(num_labels, dtype=int)])
        non_zero_clones = np.concatenate([current_data.non_zero_clones,
                                          np.arange(self.next_mutation_index,
                                                    self.next_mutation_index + num_labels)])

        for i, (c, n) in enumerate(zip(labels_per_clone, non_zero_clones)):
            for _ in range(c):
                current_population[i] -= 1
                self._add_labelled_clone(n, label, label_fitness, label_gene_name)

        gr_z = np.where(current_population > 0)[0]  # The indices of clones alive at this point in the current pop
        non_zero_clones = non_zero_clones[gr_z]  # Convert to the original clone numbers
        current_population = current_population[gr_z]  # Only keep the currently alive clones in current pop

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        current_data.update(current_population=current_population,
                            non_zero_clones=non_zero_clones)
        return current_data

    def _add_labelled_clone(self, parent_idx: int, label: int, label_fitness: float, label_gene_name: str | None) -> None:
        """Select a fitness for the new mutation and the cell in which the mutation occurs
        parent_idx = the id of the clone in which the mutation occurs
        """
        selected_clone = self.clones_array[parent_idx]
        old_fitness = selected_clone[self.fitness_idx]
        old_mutation_array = self.raw_fitness_array[parent_idx]
        new_fitness_array = old_mutation_array.copy()
        if label_gene_name is None:
            gene_mutated = np.nan  # Not a gene mutation. Any fitness change will be on wild type
            fitness_arr_col = 0
        else:
            gene_mutated = self.fitness_calculator.get_gene_number(label_gene_name)
            fitness_arr_col = gene_mutated + 1  # The first column of the fitness array is the wild type fitness, so add 1 to get the right column for the gene
        if label_fitness is not None:  # Fitness will replace what went before for that gene/wild type
            new_fitness_array[fitness_arr_col] = label_fitness
            new_fitness, self.raw_fitness_array[self.next_mutation_index] = self.fitness_calculator.combine_vectors(np.atleast_2d(new_fitness_array))
            new_fitness = new_fitness[0]  # We are only adding one clone at a time.
        else:
            new_fitness = old_fitness

        self.label_muts.add(self.next_mutation_index)

        # Add the new clone to the clone_array
        self.clones_array[self.next_mutation_index] = self.next_mutation_index, label, new_fitness, \
                                                      self.plot_idx, parent_idx, gene_mutated

        # Update ancestors and descendants. Note, all clones already have themselves as ancestor and descendant.
        self.tree.create_node(str(self.next_mutation_index), self.next_mutation_index, parent=parent_idx)

        # Update the mutation_array
        self.raw_fitness_array[self.next_mutation_index] = new_fitness_array

        self.next_mutation_index += 1

    def _draw_mutations_for_single_cell(self, parent_idxs) -> None:
        """
        For the case where a single or multiple mutations are added to the same cell.
        If multiple, they must be added one at a time so that they combine fitness with each other correctly

        :param parent_idxs:
        :return:
        """
        for p in parent_idxs:
            self._draw_multiple_mutations_and_add_to_array([p])

    def _draw_multiple_mutations_and_add_to_array(self, parent_idxs) -> None:
        """Select a fitness for the new mutation and the cell in which the mutation occurs
        parent_idx = the id of the clone in which the mutation occurs

        For multiple mutations at once. Need the new mutation generator
        """
        selected_clones = self.clones_array[parent_idxs]
        new_types = selected_clones[:, self.label_idx]  # Are the new clones labelled or not
        old_fitnesses = selected_clones[:, self.fitness_idx]
        old_mutation_arrays = self.raw_fitness_array[parent_idxs]

        # Get a fitness value for the new clone.
        new_fitness_values, new_fitness_arrays, \
        synonymous, genes_mutated = self.fitness_calculator.get_new_fitnesses(old_fitnesses, old_mutation_arrays)

        mutation_indices = np.arange(self.next_mutation_index, self.next_mutation_index + len(parent_idxs))

        s = synonymous == 1
        ns = synonymous == 0
        self.s_muts.update(mutation_indices[s])
        self.ns_muts.update(mutation_indices[ns])

        # Add the new clone to the clone_array
        new_fitness_values = self._apply_treatment(new_fitness_values, new_fitness_arrays)
        new_array = np.array([mutation_indices, new_types, new_fitness_values,
                              np.full(len(parent_idxs), self.plot_idx), parent_idxs, genes_mutated]).T
        self.clones_array[self.next_mutation_index:self.next_mutation_index + len(parent_idxs)] = new_array

        # Update clone tree
        for m, p in zip(mutation_indices, parent_idxs):
            self.tree.create_node(str(m), m, parent=p)

        # Update the mutation_array
        self.raw_fitness_array[self.next_mutation_index:self.next_mutation_index + len(parent_idxs)] = new_fitness_arrays

        self.next_mutation_index += len(parent_idxs)

    def _store_any_extras(self, new_growth_rate, synonymous, gene_mutated, parent_idx):
        # A function to be used if more information needs storing after a mutation
        pass
