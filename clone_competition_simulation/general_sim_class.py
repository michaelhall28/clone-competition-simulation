"""
This is a super class for all of the simulations.
It contains the common function to setup, run and plot results from simulations.

The subclasses have to define the sim_step and any other functions required for the specific algorithm
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import bisect
from collections import Counter
import pickle
from clone_competition_simulation.useful_functions import mean_clone_size, mean_clone_size_fit, surviving_clones_fit, \
    incomplete_moment, add_incom_to_plot
from clone_competition_simulation.animator import NonSpatialToGridAnimator, HexAnimator, HexFitnessAnimator
import warnings
from scipy.sparse import lil_matrix, SparseEfficiencyWarning
import gzip
from treelib import Tree
warnings.simplefilter('ignore',SparseEfficiencyWarning)


class GeneralSimClass(object):
    """
    Common functions for all simulation algorithms.
    Functions for setting up simulations and for plotting results
    """
    def __init__(self, parameters):
        """

        :param parameters: A Parameters object.
        """
        # Get attributes from the parameters
        self.total_pop = parameters.initial_cells
        self.initial_size_array = parameters.initial_size_array
        self.initial_clones = len(self.initial_size_array)
        self.mutation_rates = parameters.mutation_rates
        self.mutation_generator = parameters.mutation_generator
        self.division_rate = parameters.division_rate
        self.max_time = parameters.max_time
        self.times = parameters.times
        # To make sure the floating point errors do not lead to incorrect times when searching adjust by small value.
        # Generally not used - finds the closest time instead.
        if len(self.times) > 1:
            min_diff = np.diff(self.times).min()
        else:
            min_diff = self.times[0]
        self._search_times = self.times - min_diff/100
        self.sample_points = parameters.sample_points
        self.non_zero_calc = parameters.non_zero_calc
        self.label_times = parameters.label_times
        self.label_frequencies = parameters.label_frequencies
        self.label_values = parameters.label_values
        self.label_fitness = parameters.label_fitness
        self.label_genes = parameters.label_genes

        self.label_times = self._adjust_raw_times(self.label_times)
        if self.label_times is not None:
            self.next_label_time = self.label_times[0]
        else:
            self.next_label_time = np.inf
        self.label_count = 0

        self.current_fitness_multiplier = None  # The effect of the current treatment
        self.treatment_count = -1
        if parameters.treatment_timings is None:
            # No treatment applied. But this set up means the initial fitness is set correctly then not changed.
            self.treatment_timings = [0, np.inf]
            self.treatment_effects = [1, 1]  # Always neutral
            self.next_treatment_time = 0
            self.treatment_replace_fitness = False
        else:
            self.treatment_timings = self._adjust_raw_times(parameters.treatment_timings)
            self.treatment_timings = list(self.treatment_timings) + [np.inf]
            self.treatment_effects = parameters.treatment_effects
            self.next_treatment_time = self.treatment_timings[0]  # First value will always be zero
            self.treatment_replace_fitness = parameters.treatment_replace_fitness

        self.parameters = parameters

        self.sim_length = len(self.times)

        self.raw_fitness_array = parameters.fitness_array
        self.clones_array = None  # Will store the information about the clones. One row per clone.
        # A clone here will contain exactly the same combination of mutations.
        self.population_array = None  # Will store the clone sizes. One row per clone. One column per sample.

        # Include indices here for later use. These are the columns of self.clones_array
        self.id_idx = 0  # Unique integer id for each clone. Int.
        self.label_idx = 1  # The type of the clone. Inherited label does not change. Int. Represents GFP or similar.
        self.fitness_idx = 2  # The fitness of the clone. Float.
        self.generation_born_idx = 3  # The sample the clone first appeared in.  Int.
        self.parent_idx = 4  # The id of the clone that this clone emerged from.  Int.
        self.gene_mutated_idx = 5  # The gene (index) the mutation appears in (or similar info). Int.
        # Could encode gene and/or nonsense/missense/... depending on how genes are defined

        self.s_muts = set()  # Synonymous mutations. Indices of the first clone they appear in
        self.ns_muts = set()  # Non-synonymous mutations. Indices of the first clone they appear in
        self.label_muts = set()  # Labelled clones. Indices of the clones that get a labeled (after init)

        self.plot_idx = 0  # Keeping track of x-coordinate of the plot

        self.new_mutation_count = 0

        # We can calculate the number of mutations added in each generation beforehand and make the arrays the correct
        # size. This should speed things up for long, mutation heavy simulations.
        self._precalculate_mutations()
        self.total_clone_count = self.initial_clones + self.new_mutation_count
        if parameters.progress:
            print(self.new_mutation_count, 'mutations to add', flush=True)

        # Make the arrays the correct size.
        self.tree = Tree()
        self.tree.create_node(str(-1), -1)  # Make a root node that isn't a clone.
        self.trimmed_tree = None  # Used for mutant clone arrays.
        self._init_arrays(parameters.label_array, parameters.gene_label_array)
        self.next_mutation_index = self.initial_clones  # Keeping track of how many mutations added

        # Details for plotting
        self.figsize = parameters.figsize
        self.descendant_counts = {}
        self.colourscales = parameters.colourscales
        self.progress = parameters.progress  # Prints update every n samples
        self.i = 0
        self.colours = None

        # Stores the sizes of clones containing particular mutants.
        self.mutant_clone_array = None

        self.tmp_store = parameters.tmp_store
        self.store_rotation = 0  # Alternates between two tmp stores (0, 1) in case error occurs during pickle dump.
        self.is_lil = True  # Is the population array stored in scipy.sparse.lil_matrix (True) or numpy array (False)
        self.finished = False
        self.random_state = None  # For storing the state of the random sequence for continuing

    ############ Functions for running simulations ############

    ##### Functions for setting up the simulations
    def _adjust_raw_times(self, array):
        """
        Takes an array of time points and converts to number of simulation steps
        This is for the Moran simulations. Overwrite for the other cases
        :param array: Numpy array or list of time points.
        """
        if array is not None:
            array = np.array(array) * self.division_rate * self.total_pop
        return array

    def _precalculate_mutations(self):
        """
        To be overwritten. Will calculate the number and timing of all mutations in the simulation
        :return:
        """
        self.new_mutation_count = 0

    def _init_arrays(self, labels_array, gene_label_array):
        """
        Defines self.clones_array, self.population_array and self.raw_fitness_array
        Fills the self.clones_array with any information given about the initial cells.
        """
        self.clones_array = np.zeros((self.total_clone_count, 6))
        self.clones_array[:, self.id_idx] = np.arange(len(self.clones_array))  # Give clone an identifier

        if labels_array is None:
            labels_array = 0
        self.clones_array[:self.initial_clones, self.label_idx] = labels_array  # Give each intial cell a type

        if gene_label_array is None:
            gene_label_array = -1
        # Give each initial cell mutation type. -1 if no mutation
        self.clones_array[:self.initial_clones, self.gene_mutated_idx] = gene_label_array

        self.clones_array[:self.initial_clones, self.generation_born_idx] = 0
        self.clones_array[:self.initial_clones, self.parent_idx] = -1

        # For each clone, need an raw fitness array as long as the number of genes
        # Needs to be dtype=float, which is the default of np.zeros
        if self.mutation_generator.multi_gene_array:
            num_cols_genes = len(self.mutation_generator.genes)+1
            if self.mutation_generator.epistatics is not None:
                num_cols = num_cols_genes + len(self.mutation_generator.epistatics)
            else:
                num_cols = num_cols_genes
            blank_fitness_array = np.full((self.total_clone_count, num_cols),
                                          np.nan, dtype=float)
            blank_fitness_array[:, 0] = self.parameters.default_fitness
            blank_fitness_array[:self.initial_clones, :num_cols_genes] = self.raw_fitness_array
            self.raw_fitness_array = blank_fitness_array
            # self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_arrays=self.raw_fitness_array)
        else:
            blank_fitness_array = np.full((self.total_clone_count, 1), self.parameters.default_fitness, dtype=float)
            blank_fitness_array[:self.initial_clones, 0] = self.raw_fitness_array
            self.raw_fitness_array = blank_fitness_array
            # self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_values=self.raw_fitness_array)

        self.population_array = lil_matrix((self.total_clone_count,
                                            self.sim_length))  # Will store the population counts

        # Start with the initial_quantities
        if self.times[0] == 0:
            self.population_array[:self.initial_clones, 0] = self.initial_size_array.reshape(
                len(self.initial_size_array), 1)
            self.plot_idx = 1

        # Make any initial clones roots of the clone tree
        for i in range(self.initial_clones):
            self.tree.create_node(str(i), i, parent=-1)  # Directly descended from the root node

    ##### Functions for running the simulation
    def run_sim(self, continue_sim=False):
        # Functions which runs any of the simulation types.
        # self.sim_step will include the differences between the methods.

        if self.i > 0:
            # Not the first time it has been run
            if self.finished:
                print('Simulation already run')
                return
            elif continue_sim:
                print('Continuing from step', self.i)
            else:
                print('Simulation already started but incomplete')
                return

        current_population = np.zeros(len(self.clones_array), dtype=int)
        current_population[:self.initial_clones] = self.initial_size_array
        if self.non_zero_calc:  # Faster for the non-spatial simulations to only track the current surviving clones
            non_zero_clones = np.where(current_population > 0)[0]
            current_population = current_population[non_zero_clones]
        else:
            non_zero_clones = None

        # Change treatment if required (can change fitness of clones)
        if self._check_treatment_time():
            self._change_treatment(initial=True)

        # Add a label (similar to a lineage tracing label) if requested
        if self._check_label_time():
            current_population, non_zero_clones = self._add_label(current_population,
                                                                  non_zero_clones,
                                                                  self.label_frequencies[self.label_count],
                                                                  self.label_values[self.label_count],
                                                                  self.label_fitness[self.label_count],
                                                                  self.label_genes[self.label_count])
        if self.progress:
            print('Steps completed:')

        while self.plot_idx < self.sim_length:
            # Run step of the simulation
            # Each step can be a generation (Wright-Fisher), a single birth-death-mutation event (Moran) or
            # a single birth or death event (Branching)
            current_population, non_zero_clones = self._sim_step(self.i, current_population,
                                                                 non_zero_clones)
            self.i += 1
            self._record_results(self.i, current_population, non_zero_clones)  # Record the current state

            # Add a label (similar to a lineage tracing label) if requested
            if self._check_label_time():
                current_population, non_zero_clones = self._add_label(current_population,
                                                                      non_zero_clones,
                                                                      self.label_frequencies[self.label_count],
                                                                      self.label_values[self.label_count],
                                                                      self.label_fitness[self.label_count],
                                                                      self.label_genes[self.label_count])

            # Change treatment if required (can change fitness of clones)
            if self._check_treatment_time():
                self._change_treatment()

        if self.progress:
            print('Finished', self.i, 'steps')

        # Clean up the results arrays
        self._finish_up()
        self.finished = True

    def continue_sim(self):
        if self.random_state is not None:
            np.random.set_state(self.random_state)
        self.run_sim(continue_sim=True)

    def _sim_step(self, i, current_population, non_zero_clones):  # Overwrite
        return current_population, non_zero_clones

    def _finish_up(self):
        """
        Some of the simulations may required some tidying up at the end,
        for example, removing unused rows in the arrays.
        :return:
        """
        pass

    ##### Functions for storing population counts.
    def _record_results(self, i, current_population, non_zero_clones):
        """
        Check if the current step is one of the sample points
        Record the results at the point the simulation is up to.
        Report progress if required
        :param i:
        :param current_population:
        :return:
        """
        if i == self.sample_points[self.plot_idx]:  # Regularly take a sample for the plot
            self._take_sample(current_population, non_zero_clones)

        if self.progress:
            if i % self.progress == 0:
                print(i, end=', ', flush=True)

    def _take_sample(self, current_population, non_zero_clones):
        if self.non_zero_calc:
            self.population_array[non_zero_clones, self.plot_idx] = current_population
        else:
            non_zero = np.where(current_population > 0)[0]
            self.population_array[non_zero, self.plot_idx] = current_population[non_zero]
        self.plot_idx += 1
        if self.tmp_store is not None:  # Store current state of the simulation.
            if self.store_rotation == 0:
                self.pickle_dump(self.tmp_store)
                self.store_rotation = 1
            else:
                self.pickle_dump(self.tmp_store + '1')
                self.store_rotation = 0

    def pickle_dump(self, filename):
        self.random_state = np.random.get_state()
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    ##### Functions for changing treatment (changes clone fitness)
    def _check_treatment_time(self):
        if self.i >= self.next_treatment_time:
            return True
        return False

    def _change_treatment(self, initial=False):
        self.treatment_count += 1
        self.current_fitness_multiplier = self.treatment_effects[self.treatment_count]
        self.next_treatment_time = self.treatment_timings[self.treatment_count+1]
        if initial:
            if self.mutation_generator.multi_gene_array:
                self.clones_array[:self.initial_clones, self.fitness_idx] = self._apply_treatment(fitness_arrays=self.raw_fitness_array[:self.initial_clones])
            else:
                self.clones_array[:self.initial_clones, self.fitness_idx] = self._apply_treatment(fitness_values=self.raw_fitness_array[:self.initial_clones])
        else:
            if self.mutation_generator.multi_gene_array:
                self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_arrays=self.raw_fitness_array)
            else:
                self.clones_array[:, self.fitness_idx] = self._apply_treatment(fitness_values=self.raw_fitness_array)

    def _apply_treatment(self, fitness_values=None, fitness_arrays=None):
        # Apply the treatment affects to an array of fitnesses
        # fitness_values is 1D array of overall fitness for each clone
        # fitness_arrays is a 2D array with one row per clone and one column per gene plus a column for wild type
        if self.mutation_generator.multi_gene_array:
            # Apply the treatment to the genes, then calculate the overall fitness.
            if not self.treatment_replace_fitness:  # Multiply the fitness by the treatment effect
                adjusted_fitness_array = fitness_arrays * self.current_fitness_multiplier
                combined_fitness_array, _ = self.mutation_generator.combine_vectors(adjusted_fitness_array)
            else:
                # Replace the fitness with the new fitness effect
                adjusted_fitness_array = np.tile(self.current_fitness_multiplier, (len(fitness_arrays), 1))
                adjusted_fitness_array[np.isnan(fitness_arrays)] = np.nan  # Leave all unmutated genes unmutated

                # Leave all untreated genes (nan multiplier) as they were.
                adjusted_fitness_array[:,
                    np.isnan(self.current_fitness_multiplier)] = fitness_arrays[:,
                                                                    np.isnan(self.current_fitness_multiplier)]
                combined_fitness_array, _ = self.mutation_generator.combine_vectors(adjusted_fitness_array)
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

    ##### Functions for adding labelled clones (similar to lineage tracing experiments)
    def _check_label_time(self):
        if self.i >= self.next_label_time:
            return True
        return False

    def _add_label(self, current_population, non_zero_clones, label_frequency, label, label_fitness, label_gene):
        """
        Add some labelling at the current label frequency.
        The labelling is not exact, so each cell has same chance.
        Use a Poisson distribution of events for each clone.
        """
        # Random draw for each clone base on clone size
        labels_per_clone = np.random.binomial(current_population, label_frequency)
        assert not np.any(labels_per_clone - current_population > 0), (labels_per_clone, current_population)

        num_labels = np.sum(labels_per_clone)
        self._extend_arrays_fixed_amount(num_labels)

        # Add the new clones to the current population. Cells will be removed from the old clones below.
        current_population = np.concatenate([current_population, np.ones(num_labels, dtype=int)])
        non_zero_clones = np.concatenate([non_zero_clones,
                                          np.arange(self.next_mutation_index,
                                                    self.next_mutation_index + num_labels)])

        for i, (c, n) in enumerate(zip(labels_per_clone, non_zero_clones)):
            for j in range(c):
                current_population[i] -= 1
                self._add_labelled_clone(n, label, label_fitness, label_gene)

        gr_z = np.where(current_population > 0)[0]  # The indices of clones alive at this point in the current pop
        non_zero_clones = non_zero_clones[gr_z]  # Convert to the original clone numbers
        current_population = current_population[gr_z]  # Only keep the currently alive clones in current pop

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        return current_population, non_zero_clones

    def _add_labelled_clone(self, parent_idx, label, label_fitness, label_gene):
        """Select a fitness for the new mutation and the cell in which the mutation occurs
        parent_idx = the id of the clone in which the mutation occurs
        """
        selected_clone = self.clones_array[parent_idx]
        old_fitness = selected_clone[self.fitness_idx]
        old_mutation_array = self.raw_fitness_array[parent_idx]
        new_fitness_array = old_mutation_array.copy()
        if label_gene is None:
            gene_mutated = -1  # Not a gene mutation. Any fitness change will be on wild type
            label_gene = 0
        else:
            gene_mutated = label_gene
        if label_fitness is not None:  # Fitness will replace what went before for that gene/wild type
            new_fitness_array[label_gene] = label_fitness
            new_fitness, self.raw_fitness_array[self.next_mutation_index] \
                = self.mutation_generator.combine_vectors(np.atleast_2d(new_fitness_array))
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

    def _extend_arrays_fixed_amount(self, extension):
        """
        Add new rows to the population and clones arrays. For when the labels are added.
        """
        s = self.population_array.shape[0]
        new_pop_array = lil_matrix((s + extension, self.sim_length))
        new_pop_array[:s] = self.population_array
        self.population_array = new_pop_array

        self.clones_array = np.concatenate([self.clones_array, np.zeros((extension, 6))], axis=0)

        self.raw_fitness_array = np.concatenate([self.raw_fitness_array,
                                                 np.full((extension, self.raw_fitness_array.shape[1]), np.nan)],
                                                axis=0)

    ##### Functions for adding mutations
    def _draw_mutations_for_single_cell(self, parent_idxs):
        """
        For the case where a single or multiple mutations are added to the same cell.
        If multiple, they must be added one at a time so that they combine fitness with each other correctly

        :param parent_idxs:
        :return:
        """
        for p in parent_idxs:
            self._draw_multiple_mutations_and_add_to_array([p])

    def _draw_multiple_mutations_and_add_to_array(self, parent_idxs):
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
        synonymous, genes_mutated = self.mutation_generator.get_new_fitnesses(old_fitnesses, old_mutation_arrays)

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

    ############ Functions for post-processing simulations ############
    def view_clone_info(self, include_raw_fitness=False):
        """
        This converts the clone_array into a more readable pandas dataframe
        :param include_raw_fitness: Add the raw_fitness_array data to the dataframe
        :return: pandas.DataFrame
        """
        df = pd.DataFrame({
            'clone id': pd.Series(self.clones_array[:, self.id_idx], dtype=int),
            'label': pd.Series(self.clones_array[:, self.label_idx], dtype=int),
            'fitness': pd.Series(self.clones_array[:, self.fitness_idx], dtype=int),
            'generation born': pd.Series(self.clones_array[:, self.generation_born_idx], dtype=int),
            'parent clone id': pd.Series(self.clones_array[:, self.parent_idx], dtype=int),
            'last gene mutated': pd.Series(
                [self.mutation_generator.get_gene_name(int(g)) for g in self.clones_array[:, self.gene_mutated_idx]],
                dtype=object),
        })

        if include_raw_fitness:
            cols = []
            if self.mutation_generator.multi_gene_array:
                cols += ['Initial clone fitness']
            cols += [g.name for g in self.mutation_generator.genes]
            if self.mutation_generator.epistatics is not None:
                cols += [e[0] for e in self.mutation_generator.epistatics]
            raw_df = pd.DataFrame(self.raw_fitness_array, columns=cols)
            df = pd.concat([df, raw_df], axis=1)

        return df

    def change_sparse_to_csr(self):
        """
        Converts to a different type of sparse matrix.
        Required for some of the post-processing and plotting functions.
        """
        if self.is_lil:
            self.population_array = self.population_array.tocsr()  # Convert to CSR matrix
        self.is_lil = False

    def _convert_time_to_index(self, t, nearest=True):
        if nearest:  # Find nearest point to the time of interest
            return self._find_nearest(t)
        else:  # Find the index at or just before the time of interest
            i = bisect.bisect_right(self._search_times, t)
            if i:
                return i - 1
            raise ValueError

    def _find_nearest(self, t):
        # From stackoverflow, Demitri, https://stackoverflow.com/a/26026189
        array = self.times
        idx = np.searchsorted(array, t, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(t - array[idx - 1]) < math.fabs(t - array[idx])):
            return idx - 1
        else:
            return idx

    def get_clone_sizes_array_for_non_mutation(self, t=None, index_given=False, label=None, exclude_zeros=True):
        """
        Gets array of all clone sizes.
        Clones here are defined by a unique set of mutations, not per mutation.
        Therefore this is only really suitable for a simulation without mutations, where we want to track the sizes of
        a number of initial clones.
        :param t: time or index of the sample to get the distribution for.
        :param index_given: True if t is the index
        :return:
        """
        if self.is_lil:
            self.change_sparse_to_csr()

        if t is None:
            index_given = True
            t = -1
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = t
        if label is not None:
            clones_to_select = np.where(self.clones_array[:, self.label_idx] == label)
            clones = self.population_array[clones_to_select, i]
        else:
            clones = self.population_array[:, i]

        clones = clones.toarray().astype(int).flatten()  # Must convert to 1D array to use bincount
        if exclude_zeros:
            clones = clones[clones > 0]

        return clones

    def get_clone_size_distribution_for_non_mutation(self, t=None, index_given=False, label=None, exclude_zeros=False):
        """
        Gets the clone size frequencies. Not normalised.
        Clones here are defined by a unique set of mutations, not per mutation.
        Therefore this is only really suitable for a simulation without mutations, where we want to track the sizes of
        a number of initial clones.
        :param t: time or index of the sample to get the distribution for.
        :param index_given: True if t is the index
        :return:
        """
        clones = self.get_clone_sizes_array_for_non_mutation(t=t, index_given=index_given, label=label,
                                                             exclude_zeros=exclude_zeros)
        counts = np.bincount(clones)
        return counts

    def get_surviving_clones_for_non_mutation(self, times=None, label=None):
        """
        Follows the surviving clones based on of each row in the clone array. This is a clone defined by a unique set of
        mutations, not be a particular mutation.
        Therefore, this function is only suitable for tracking the progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.
        """
        if times is None:
            times = self.times
        surviving_clones = []
        if label is not None:
            clones_to_select = np.where(self.clones_array[:, self.label_idx] == label)
            pop_array = self.population_array[clones_to_select]
        else:
            pop_array = self.population_array
        for t in times:
            idx = self._convert_time_to_index(t)
            surviving_clones.append(pop_array[:, idx].count_nonzero())

        return surviving_clones, times

    def get_clone_ancestors(self, clone_idx):
        """
        Return the clone ids of all ancestors of a given clone.
        :param clone_idx: int
        :return:
        """
        return [n for n in self.tree.rsearch(clone_idx)]

    def get_clone_descendants(self, clone_idx):
        """
        Return the clone ids of all descendants of a given clone.
        :param clone_idx: int
        :return:
        """
        return list(self.tree.subtree(clone_idx).nodes.keys())  # Might be better way to do this

    def _trim_tree(self):
        # Some clones may have appeared and died between sampling points.
        # These won't affect the results but can slow down the processing
        # Make new tree just from sampled clones.

        if self.trimmed_tree is None:

            non_zero_sampled_clones = np.unique((self.population_array.nonzero()[0]))
            sampled_clones_set = set()
            for clone in non_zero_sampled_clones[::-1]:
                if clone not in sampled_clones_set:
                    ancestors = self.get_clone_ancestors(clone)
                    sampled_clones_set.update(ancestors)

            sampled_clones_set.remove(-1)
            self.trimmed_tree = Tree()
            self.trimmed_tree.create_node("-1", -1)
            self.sampled_clones = sorted(sampled_clones_set)
            for n in sorted(self.sampled_clones):  # For every clone that is alive at a sampling time
                for n2 in self.tree.rsearch(n):  # Find the first ancestor that was sampled. This is the new parent.
                    if n != n2 and (n2 == -1 or n2 in sampled_clones_set):
                        self.trimmed_tree.create_node(str(n), n, parent=n2)
                        break

    def _get_clone_descendants_trimmed(self, clone_idx):
        """Must run trim tree first"""
        return list(self.trimmed_tree.subtree(clone_idx).nodes.keys())

    def track_mutations(self, selection='all'):
        """
        Get a dictionary of the clones which contain each mutation.
        :param selection: 'all', 'ns', 's'. All/non-synonymous only/synonymous only.
        :return: Dict. Key: mutation id (id of first clone which contains the mutation),
        value: set of clone ids which contain that mutation
        """
        if selection == 's':
            mutant_clones = {k: self.get_clone_descendants(k) for k in self.s_muts}
        elif selection == 'ns':
            mutant_clones = {k: self.get_clone_descendants(k) for k in self.ns_muts}
        elif selection == 'all':
            mutant_clones = {k: self.get_clone_descendants(k) for k in range(len(self.clones_array))}
        elif selection == 'label':
            mutant_clones = {k: self.get_clone_descendants(k) for k in self.label_muts}
        elif selection == 'mutations':
            mutant_clones = {k: self.get_clone_descendants(k) for k in self.ns_muts.union(self.s_muts)}
        elif selection == 'non_zero':
            self._trim_tree()
            mutant_clones = {k: self._get_clone_descendants_trimmed(k) for k in self.sampled_clones}
        else:
            print("Please select from 'all', 's', 'ns', 'label', 'mutations' or 'non_zero'")
            raise ValueError("Please select from 'all', 's', 'ns', 'label', 'mutations' or 'non_zero'")

        return mutant_clones

    def _create_mutant_clone_array(self):
        """
        Create an array with the clone sizes for each mutant across the entire simulation.
        The populations will usually add up to more than the total since many clones will have multiple mutations
        """
        mutant_clones = self.track_mutations(selection='non_zero')
        self.mutant_clone_array = lil_matrix(self.population_array.shape)
        for mutant in mutant_clones:
            self.mutant_clone_array[mutant] = self.population_array[mutant_clones[mutant]].sum(axis=0)

    def get_idx_of_gene_mutated(self, gene_mutated):
        """
        Returns a set of all clones with gene_mutated given
        :param gene_mutated: The name of the gene mutated.
        """
        gene_num = self.mutation_generator.get_gene_number(gene_mutated)
        return set(np.where(self.clones_array[:, self.gene_mutated_idx] == gene_num)[0])

    def get_mutant_clone_sizes(self, t=None, selection='mutations', index_given=False, gene_mutated=None, non_zero_only=False):
        """
        Get an array of mutant clone sizes at a particular time
        WARNING: This may not work exactly as expected if there were multiple initial clones!
        :param t: time/sample index
        :param selection: 'all', 'ns', 's'. All/non-synonymous only/synonymous only.
        :param index_given: True if t is an index of the sample, False if t is a time.
        :param gene_mutated: Gene name. Only return clone sizes for a particular additional label.
        For example to only get mutations for a single gene.
        :param non_zero_only: Only return mutants with a positive cell count.
        :return: np.array of ints
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = t
        if self.mutant_clone_array is None:
            # If the mutant clone array has not been created yet, create it.
            self._create_mutant_clone_array()
        # We now find all rows in the mutant clone array that we want to keep
        if selection == 'all':
            muts = set(range(self.initial_clones, len(self.clones_array)))  # Get all rows except the initial clones
        elif selection == 'mutations':
            muts = self.ns_muts.union(self.s_muts)
        elif selection == 'ns':
            muts = self.ns_muts
        elif selection == 's':
            muts = self.s_muts
        elif selection == 'label':
            muts = self.label_muts
        if gene_mutated is not None:
            muts = list(muts.intersection(self.get_idx_of_gene_mutated(gene_mutated)))
        else:
            muts = list(muts)

        mutant_clones = self.mutant_clone_array[muts][:, i].toarray().astype(int).flatten()

        if non_zero_only:
            return mutant_clones[mutant_clones > 0]
        else:
            return mutant_clones

    def get_mutant_clone_size_distribution(self, t=None, selection='mutations', index_given=False, gene_mutated=None):
        """
        Get the frequencies of mutant clone sizes. Not normalised.
        :param t: time/sample index
        :param selection: 'mutations', 'ns', 's'. All/non-synonymous only/synonymous only.
        :param index_given: True if t is an index of the sample, False if t is a time.
        :param gene_mutated: Int. Only return clone sizes for a particular additional label.
        For example to only get mutations for a single gene.
        :return: np.array of ints.
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = t
        if selection == 'mutations':
            if self.ns_muts and not self.s_muts:
                selection = 'ns'
            elif self.s_muts and not self.ns_muts:
                selection = 's'
            elif not self.s_muts and not self.ns_muts:
                print('No mutations at all')
                return None
        elif selection == 'ns' and not self.ns_muts:
            print('No non-synonymous mutations')
            return None
        elif selection == 's' and not self.s_muts:
            print('No synonymous mutations')
            return None

        clones = self.get_mutant_clone_sizes(i, selection=selection, index_given=True,
                                             gene_mutated=gene_mutated)

        counts = np.bincount(clones)
        return counts

    def get_dnds(self, t=None, min_size=1, gene=None):
        """
        Returns the dN/dS at a particular time.
        :param t: Time. If None, will be the end of the simulation.
        :param min_size: Int. The minimum size of clones to include.
        :param gene: Int. The type of the mutation. E.g. For getting dN/dS for a particular gene.
        :return:
        """
        if t is None:
            t = self.max_time
        ns_mut = self.get_mutant_clone_sizes(t, selection='ns', gene_mutated=gene)
        s_mut = self.get_mutant_clone_sizes(t, selection='s', gene_mutated=gene)
        ns_mut_measured = ns_mut[ns_mut >= min_size]
        total_ns = len(ns_mut_measured)
        s_mut_measured = s_mut[s_mut >= min_size]
        total_s = len(s_mut_measured)

        gene_num = self.mutation_generator.get_gene_number(gene)  # If gene is None, will get the overall ns
        expected_ns = total_s * (1 / self.mutation_generator.get_synonymous_proportion(gene_num) - 1)
        try:
            dnds = total_ns / expected_ns
            return dnds
        except ZeroDivisionError as e:
            return np.nan

    def get_labeled_population(self, label=None):
        """
        If label is None, will return the total population (not interesting for the fixed population models)
        :param label:
        :return: Array of population at all time points
        """
        if label is not None:
            clones_to_select = np.where(self.clones_array[:, self.label_idx] == label)
            pop = self.population_array[clones_to_select]
        else:
            pop = self.population_array

        return pop.toarray().sum(axis=0)

    def get_mean_clone_size(self, t=None, selection='mutations', index_given=False, gene_mutated=None):
        """
        Returns the mean mutant clone size.
        Each clone is defined as the total cells containing a mutation.
        :param t: time point. If index_given=True, it is the index of the time point required.
        :param selection: 'mutations' for all mutant clones. 'ns' for non-synonymous mutations only. 's' for synonymonus
        clones only.
        :param index_given: To use with t.
        :param gene_mutated: String. If given, will limit to clones of the gene given.
        :return:
        """
        clone_sizes = self.get_mutant_clone_sizes(t=t, selection=selection, index_given=index_given,
                                                  gene_mutated=gene_mutated)
        mean_ = clone_sizes[clone_sizes > 0].mean()
        return mean_

    def get_mean_clone_sizes_syn_and_non_syn(self, t=None, index_given=False, gene_mutated=None):
        """
        Convenient function to get mean size of both the synonymous and non-synonymous mutations.
        :param t:
        :param index_given:
        :param gene_mutated:
        :return: Tuple(float, float). Mean synonymous clone size, mean non-synonymous clone size.
        """
        mean_syn = self.get_mean_clone_size(t=t, selection='s', index_given=index_given, gene_mutated=gene_mutated)
        mean_non_syn = self.get_mean_clone_size(t=t, selection='ns', index_given=index_given, gene_mutated=gene_mutated)

        return mean_syn, mean_non_syn

    def get_average_fitness(self, t=None):
        """
        Get the average fitness of the entire population at the given time point.
        :param t: If None, will be the end of the simulation.
        :return: float
        """
        self.change_sparse_to_csr()
        if t is None:
            idx = -1
        else:
            idx = self._convert_time_to_index(t)

        fitnesses = self.clones_array[:, self.fitness_idx]
        weights = np.squeeze(self.population_array[:, idx].toarray()) * fitnesses
        global_average_fitness = weights.sum() / self.population_array[:, idx].sum()
        return global_average_fitness

    ############ Plotting functions ############
    def _assign_colour(self, scaled_fitness, clone_label, ns, initial, last_mutated_gene, genes_mutated):
        """
        Gets the colour for a clone to be plotted in.
        The colour will depend on the chosen colourscale and the attributes of the clone.
        :param scaled_fitness: The growth_rate/fitness of the clone
        :param clone_label: The type of the clone
        :param ns: Whether the (last) mutation is non-synonymous
        :param initial: Whether the clone is one from the start of the simuation (True), or whether it was created by
         a mutation during the simulation (False)
        :param last_mutated_gene:
        :param genes_mutated: all genes mutated in the clone
        :return:
        """
        if self.colourscales is not None:
            return self.colourscales._get_colour(scaled_fitness, clone_label, ns, initial, last_mutated_gene,
                                                 genes_mutated)
        else:
            return cm.YlOrBr(scaled_fitness / 2)  # Range of yellow/brown/orange

    def _get_colours(self, clones_array, force_regenerate=False):
        # Generate the colours for the clones plot. Colour depends on type (wild type/A), relative fitness and s/ns
        if not self.colours or force_regenerate:
            rates = clones_array[:, self.fitness_idx]
            min_ = rates.min() - 0.1
            max_ = rates.max()
            self.colours = {}
            for i, clone in enumerate(clones_array):
                scaled_fitness = (clone[self.fitness_idx] - min_) / (max_ - min_)
                if clone[self.id_idx] in self.ns_muts:
                    ns = True
                else:
                    ns = False
                if clone[self.generation_born_idx] == 0:
                    initial = True
                else:
                    initial = False
                self.colours[clone[self.id_idx]] = self._assign_colour(scaled_fitness, clone[self.label_idx], ns, initial,
                                                                       clone[self.gene_mutated_idx],
                                                                       tuple(np.where(~np.isnan(self.raw_fitness_array[i]))[0]))

    def get_colour(self, clone_id):
        """
        Return the colour for a clone_id.
        If the clone_id is not in the clones_array (can happen if manually adding values to a grid),
        create a random colour from the colourscale for it.
        :param clone_id: int.
        :return:
        """
        if clone_id not in self.colours:
            # Not in the colours dictionary.
            if clone_id in self.clones_array[self.id_idx]:
                # It is a clone generated during the simulation, so can generate all of the colours
                self._get_colours(force_regenerate=True)
            else:
                # A new clone_id not seen before. This is probably for some manually manipulation of grids for plotting.
                # Generate a new colour for this clone. Store for later.
                # This will ignore any complex rules for colouring. To do that, add a row to the clones_array and
                # generated the colours dictionary.
                if type(self.colourscales.colourmaps) == dict:
                    colourmap = self.colourscales.colourmaps[self.colourscales.colourmaps.keys()[0]]
                else:
                    colourmap = self.colourscales.colourmaps
                self.colours[clone_id] = colourmap(np.random.random())

        return self.colours[clone_id]

    def muller_plot(self, plot_file=None, plot_against_time=True, quick=False, min_size=1,
                    allow_y_extension=False, plot_order=None, figsize=None, force_new_colours=False, ax=None,
                    show_mutations_with_x=True):
        """
        Plots the results of the simulation over time.
        Mutations marked with X unless show_mutations_with_x=False.
        The clones will appear as growing and shrinking sideways tear drops.
        Sub-clones emerge from their parent clones
        :param plot_file: File name to save the plot. If none, the plot will be displayed.
        If a file name, include the file type, e.g. "output_plot.pdf"
        :param plot_against_time: Bool. Use the time from the simulation instead of index of the sample for x-axis
        :param quick: Bool. Runs a faster version of the plotting which looks worse
        :param min_size: Show only clones which reach this number of cells.
         Showing less clones speeds up the plotting and can make the plot clearer.
        :param allow_y_extension: If the population is not constant, allows the y-axis to extend beyond the initial pop
        :param plot_order: Manually list of the order of the clones to plot.
        :param figsize: Figure size.
        :param force_new_colours: Regenerate the colours of each clone.
        :param ax: Axes to plot on.
        :param show_mutations_with_x: If True, will place Xs on the plot to mark the origins of clones
        :return: ax
        """
        if self.is_lil:
            self.change_sparse_to_csr()

        self._get_colours(self.clones_array, force_new_colours)  # Get the colours used for the plots

        if min_size > 0:  # Have to keep as >0 as some algorithms (e.g. relative fitness may have fractional counts)
            # Removes clones too small to plot by absorbing them into their parent clones
            clone_array, populations = self._absorb_small_clones(min_size)
        else:
            clone_array, populations = self.clones_array, self.population_array

        # Break up the populations so subclones appear from their parent clone
        split_pops_for_plotting, plot_order = self._split_populations_for_muller_plot(clone_array,
                                                                                      populations, plot_order)
        if ax is None:
            if figsize is None:
                figsize = self.figsize
            fig, ax = plt.subplots(figsize=figsize)

        if quick:
            self._make_quick_stackplot(ax, split_pops_for_plotting, plot_order, plot_against_time)
        else:
            cumulative_array = np.cumsum(split_pops_for_plotting, axis=0)
            self._make_stackplot(ax, cumulative_array, plot_order, plot_against_time)  # Add the clone populations to the plots

        # Add the clone births to the plot as X's
        if show_mutations_with_x:
            x = []
            y = []
            c = []
            for clone in clone_array:
                gen = clone[self.generation_born_idx]
                if gen > 0:
                    plot_gen = int(gen - 1)  # Puts the mutation mark so it appears at the start of the clone region
                    pops = np.where(plot_order == clone[self.id_idx])[0][0]
                    if plot_against_time:
                        x.append(self.times[int(plot_gen)])
                    else:
                        x.append(plot_gen)
                    y.append(split_pops_for_plotting[:pops][:, plot_gen].sum())
                    if clone[self.id_idx] in self.ns_muts:
                        c.append('r')  # Plot non-synonymous mutations with a red X
                    elif clone[self.id_idx] in self.s_muts:
                        c.append('b') # Plot synonymous mutations with a blue X
                    else:
                        c.append('k')  # Plot a labelling event with a black X

            ax.scatter(x, y, c=c, marker='x')

        if allow_y_extension:
            plt.gca().set_ylim(bottom=0)
        else:
            plt.ylim([0, self.total_pop])
        if plot_against_time:
            plt.xlim([0, self.max_time])
        else:
            plt.xlim([0, self.sim_length - 1])

        if plot_file:
            plt.savefig('{0}'.format(plot_file))

        return ax

    def _absorb_small_clones(self, min_size=1):
        """Creates a new clones_array and population_array removing clones that never get larger than the
        minimum proportion min_prop.
        Clones which are too small are absorbed into their parent clone so the total population remains the same.
        """
        clones_to_remove = set()
        new_pop_array = self.population_array.copy()
        parent_set = Counter(self.clones_array[:, self.parent_idx])
        for i in range(len(self.clones_array) - 1, -1, -1):  # Start from the youngest clones
            if new_pop_array[i].max() < min_size:  # If clone is always smaller than minimum size
                if parent_set[i] == 0:  # If clone does not have any large descendants.
                    parent = int(self.clones_array[i, self.parent_idx])  # Find the parent of this small clone
                    new_pop_array[parent] += new_pop_array[i]  # Add the population of the small clone to the parent
                    new_pop_array[i] = 0  # Remove the population of the small clone
                    clones_to_remove.add(i)
                    parent_set[parent] -= 1
        clones_to_keep = sorted(set(range(len(self.clones_array))).difference(clones_to_remove))
        return self.clones_array[clones_to_keep], new_pop_array[clones_to_keep]

    def _get_children(self, clones_array, idx):
        # Return the ids of immediate subclones of the given clone idx
        return clones_array[clones_array[:, self.parent_idx] == idx][:, self.id_idx]

    def _get_descendants_for_muller_plot(self, clones_array, idx, order):
        # Find the subclones of the given clone. Runs iteratively until found all descendants
        # Adds clone ids to order list.
        # order will be used to make the stackplot so that the subclones appear from their parent clone
        # Uses the clones array rather than the tree since it may be filtered to remove small clones.
        order.append(idx)
        children = self._get_children(clones_array, idx)  # Immediate subclones of the clone idx
        np.random.shuffle(children)
        self.descendant_counts[idx] = len(children)
        for ch in children:  # Find the subclones of the subclones.
            if ch != idx:
                self._get_descendants_for_muller_plot(clones_array, ch, order)
                order.append(idx)

    def _split_populations_for_muller_plot(self, clones_array, population_array, plot_order=None):
        # Breaks up the populations so subclones appear from their parent clone

        original_clones = clones_array[clones_array[:, self.parent_idx] == -1]

        # Will put labelled clones together if plot_order given or if the labels are in the original clones.
        if plot_order is None:
            all_types = np.unique(original_clones[:, self.label_idx])
        else:
            all_types = plot_order

        orders = []
        for t in all_types:
            order_t = []
            originators = original_clones[original_clones[:, self.label_idx] == t]
            for orig in originators[:, self.id_idx]:
                self._get_descendants_for_muller_plot(clones_array, orig, order_t)
            orders.append(order_t)

        split_pops_for_plotting = np.concatenate([np.concatenate([
            population_array[clones_array[:, self.id_idx] == o].toarray() / (self.descendant_counts[o] + 1) \
            for o in order]) for order in orders], axis=0)

        plot_order = list(itertools.chain.from_iterable(orders))
        return split_pops_for_plotting, plot_order

    def _make_stackplot(self, ax, cumulative_array, plot_order, plot_against_time=True):
        # Make the stackplot using fill between. Prevents gaps in the plot that appear with using matplotlib stackplot
        for i in range(len(plot_order) - 1, -1, -1):  # Start from the end/top
            colour = self.get_colour(plot_order[i])
            array = cumulative_array[i]
            if i > 0:
                next_array = cumulative_array[i - 1]
            else:
                next_array = 0

            if plot_against_time:
                x = self.times
            else:
                x = list(range(self.sim_length))
            ax.fill_between(x, array, 0, where=array > next_array, facecolor=colour,
                            interpolate=True, linewidth=0)  # Fill all the way from the top of the clone to the x-axis

    def _make_quick_stackplot(self, ax, split_pops_for_plotting, plot_order, plot_against_time=True):
        # Make the stackplot using matplotlib stackplot
        if plot_against_time:
            x = self.times
        else:
            x = list(range(self.sim_length))
        ax.stackplot(x, split_pops_for_plotting, colors=[self.get_colour(i) for i in plot_order])

    def plot_incomplete_moment(self, t=None, selection='mutations', xlim=None, ylim=None, plt_file=None, sem=False,
                               show_fit=False, show_legend=True, fit_prop=1,
                               min_size=1, errorevery=1, clear_previous=True, show_plot=False, max_size=None,
                               fit_style='m--', label='InMo', ax=None):
        """
        Plots the incomplete moment
        :param t: The time to plot the incomplete moment for. If None, will use the end of the simulation
        :param selection: 'mutations', 'ns' or 's' for all mutations, non-synonymous only or synonymous only
        :param xlim: Tuple/list for the x-limits of the plot
        :param ylim: Tuple/list for the y-limits of the plot
        :param plt_file: File to output the plot - include the file type e.g. incom_plot.pdf.
        :param sem: Will display the SEM on the plot
        :param show_fit: Adds a straight line fit to the log plot. The intercept will be fixed at the (min_size, 1).
        Will be fitted to a proportion of the data specified by fit_prop
        :param show_legend: Shows a legend with the R^2 coefficient of the straight line fit.
        :param fit_prop: The proportion of the data to fit the straight line on.
        Starts from the smallest included sizes. Will be the clone sizes that together contain fit_prop proportion
        of the clones.
        :param min_size: The smallest clone size to include. All smaller clones will be ignored.
        :param errorevery: If showing the SEM, will only show the errorbar every errorevery points.
        :param clear_previous: If wanting to show more on the same plot, set to false and plot the other traces
        before running this function
        :param show_plot: If needing to show the plot rather than adding more traces after
        :return:
        """
        if t is None:
            t = self.max_time
        clone_size_dist = self.get_mutant_clone_size_distribution(t, selection)
        if clone_size_dist is not None:
            if min_size > 0:
                clone_size_dist[:min_size] = 0
            if max_size is not None:
                clone_size_dist = clone_size_dist[:max_size + 1]
            incom = incomplete_moment(clone_size_dist)
            if clear_previous and ax is None:
                plt.close('all')
                fig, ax = plt.subplots()
            if incom is not None:
                add_incom_to_plot(incom, clone_size_dist, sem=sem, show_fit=show_fit, fit_prop=fit_prop,
                                  min_size=min_size, label=label, errorevery=errorevery, fit_style=fit_style, ax=ax)

                ax.set_yscale("log")
                if xlim is not None:
                    ax.xlim(xlim)
                if ylim is not None:
                    ax.ylim(ylim)
                if show_legend:
                    ax.legend()

                ax.set_xlabel('Clone size (cells)')
                ax.set_ylabel('First incomplete moment')

                if plt_file is not None:
                    plt.savefig('{0}'.format(plt_file))
                elif show_plot:
                    plt.show()

    def _expected_incomplete_moment(self, t, max_n):
        """The expected incomplete moment if the simulation is neutral and all clones are measured accurately"""
        return np.exp(-np.arange(1, max_n + 1) / (self.division_rate * t))

    def plot_dnds(self, plt_file=None, min_size=1, gene=None, clear_previous=True, legend_label=None, ax=None):
        """
        Plot dN/dS ratio over time.
        :param plt_file: Output file if required. Include the output file type in the name, e.g. "out.pdf"
        :param min_size: Minimum size of clones to include.
        :param gene: Only include mutations in this gene.
        :param clear_previous: Clear previous plot.
        :param legend_label: Label for the line in the figure.
        :param ax: ax to plot on.
        :return: None
        """
        if clear_previous and ax is None:
            plt.close('all')
            fig, ax = plt.subplots()
        elif ax is None:
            ax = plt.gca()
        dndss = [self.get_dnds(t, min_size, gene) for t in self.times]
        ax.plot(self.times, dndss, label=legend_label)
        if plt_file is not None:
            plt.savefig('{0}'.format(plt_file))

    def plot_overall_population(self, label=None, legend_label=None, ax=None):
        """
        With no label, plots for simulations without a fixed total population
        (will also run for the fixed population, but will not be interesting)

        With a label, will plot the labelled population
        """
        if ax is None:
            fig, ax = plt.subplots()
        pop = self.get_labeled_population(label=label)
        ax.plot(self.times, pop, label=legend_label)
        ax.set_ylabel("Population")
        ax.set_xlabel("Time")

    def plot_average_fitness_over_time(self, legend_label=None, ax=None):
        """
        Plots the average fitness of the entire cell population.
        :param legend_label:
        :param ax:
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots()
        avg_fit = [self.get_average_fitness(t) for t in self.times]
        ax.plot(self.times, avg_fit, label=legend_label)
        ax.set_ylabel("Average fitness")
        ax.set_xlabel("Time")

    def animate(self, animation_file, grid_size=None, generations_per_frame=1, starting_clones=1,
                figsize=None, figxsize=5, bitrate=500, min_prop=0, external_call=False, dpi=100, fps=5,
                fitness=False, fitness_cmap=cm.Reds, min_fitness=0, fixed_label_text=None, fixed_label_loc=(0, 0),
                fixed_label_kwargs=None, show_time_label=False, time_label_units=None,
                time_label_decimal_places=0, time_label_loc=(0, 0), time_label_kwargs=None, equal_aspect=False):
        """
        Output an animation of the simulation on a 2D grid.

        For the 2D simulations, will plot the grid from the simulations.
        For the non-spatial simulations, will plot a 2D representation of the clone proportions. This is not very
        meaningful, but may help to visualise the simulation results.

        :param animation_file: Output file. Needs the file type included, e.g. 'out.mp4'
        :param grid_size: For non-spatial simulations only, size of the grid to plot on.
        :param generations_per_frame: For non-spatial simulations only.
        :param starting_clones: For non-spatial simulations only. Can split initial clone cell populations into
        separately placed clones.
        :param figsize: Figure size.
        :param figxsize: For 2D simulations only. If not using figsize, this gives the x-dimension of the video. The
        y-dimension will be calculated based on the grid dimensions.
        :param bitrate: Bitrate of the video.
        :param min_prop: For non-spatial simulations only. Hides clones which occupy less than this proportion of the
        total tissue. Helps to speed up animation.
        :param external_call: For 2D simulations only. Will run a version which is cruder and may run faster
        :param dpi: DPI of the video.
        :param fps: Frames per second.
        :param fitness: Boolean. For 2D simulations only. Colour cells by their fitness instead of their clone_id.
        :param fitness_cmap: For 2D simulations only. Colourmap for the fitness.
        :param min_fitness: The lower limit for the colourbar in the fitness animation.
        :param fixed_label_text: For 2D simulations only. Text to add as a label over the video.
        :param fixed_label_loc: Tuple. For 2D simulations only. The location for the fixed_label_text.
        :param fixed_label_kwargs: Dictionary. For 2D simulations only. Any kwargs to pass to ax.text for the fixed_label_text.
        :param show_time_label: For 2D simulations only. If True, will show the time of each frame overlaid on the video.
        The time will be based on the times from the simulation (which may not be the frame number).
        :param time_label_units: String, the units for the time label. For 2D simulations only.
        Will not adjust the values, is just a string to follow the number. E.g. 'days', 'weeks', 'years'.
        :param time_label_decimal_places: For 2D simulations only. Number of decimal places to show for the time label.
        :param time_label_loc: Tuple. For 2D simulations only. Location of the time label.
        :param time_label_kwargs: Dictionary. For 2D simulations only. Any kwargs to pass to ax.text for the time label.
        :param equal_aspect: For 2D simulations only.  If True, will force the aspect ratio of the x and y axes to have the same scale.
        However, this will not look equal aspect in terms of the number of cells per unit due to the tesselation of the hexagons.
        :return:
        """
        if self.is_lil:
            self.change_sparse_to_csr()

        if self.parameters.algorithm in self.parameters.spatial_algorithms:
            if fitness:
                animator = HexFitnessAnimator(self, cmap=fitness_cmap, min_fitness=min_fitness,
                                              figxsize=figxsize, figsize=figsize, dpi=dpi,
                                              bitrate=bitrate, fps=fps)
            else:
                animator = HexAnimator(self, figxsize=figxsize, figsize=figsize, dpi=dpi, bitrate=bitrate,
                                       fps=fps, external_call=external_call, fixed_label_text=fixed_label_text,
                                       fixed_label_loc=fixed_label_loc, fixed_label_kwargs=fixed_label_kwargs,
                                       show_time_label=show_time_label, time_label_units=time_label_units,
                                       time_label_decimal_places=time_label_decimal_places,
                                       time_label_loc=time_label_loc, time_label_kwargs=time_label_kwargs,
                                       equal_aspect=equal_aspect)

        else:
            if fitness:
                print('Cannot currently animate fitness for non-spatial simulations')
            animator = NonSpatialToGridAnimator(self, grid_size=grid_size, generations_per_frame=generations_per_frame,
                                starting_clones=starting_clones, figsize=figsize, bitrate=bitrate, min_prop=min_prop,
                                dpi=dpi, fps=fps)

        animator.animate(animation_file)

    ## Plots for lineage tracing experiments
    # These assume no mutations occurred during the simulation,
    # but all mutations (or labelled clones) are induced at the start.
    def plot_mean_clone_size_graph_for_non_mutation(self, times=None, label=None, show_spm_fit=True, spm_fit_rate=None,
                                                    legend_label=None, legend_label_fit=None, ax=None,
                                                    plot_kwargs=None, fit_plot_kwargs=None):
        """
        Follows the mean clone sizes of each row in the clone array. This is a clone defined by a unique set of
        mutations, not be a particular mutation.
        Therefore, this function is only suitable for tracking the progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.
        """
        if times is None:
            times = self.times

        means = []
        for t in times:
            means.append(mean_clone_size(self.get_clone_size_distribution_for_non_mutation(t, label=label)))

        if ax is None:
            fig, ax = plt.subplots()
        if show_spm_fit:
            if spm_fit_rate is None:
                spm_fit_rate = self.division_rate
            # Plot the theoretical mean clone size from the single progenitor model
            if fit_plot_kwargs is None:
                fit_plot_kwargs = {}
            ax.plot(times, mean_clone_size_fit(times, spm_fit_rate), label=legend_label_fit, **fit_plot_kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean clone size of surviving clones')
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.plot(times, means, label=legend_label, **plot_kwargs)

    def plot_surviving_clones_for_non_mutation(self, times=None, ax=None, label=None, show_spm_fit=False,
                                           spm_fit_rate=None, plot_kwargs=None, legend_label=None):
        """
        Follows the surviving clones based on of each row in the clone array. This is a clone defined by a unique set of
        mutations, not be a particular mutation.
        Therefore, this function is only suitable for tracking the progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.
        """
        surviving_clones, times = self.get_surviving_clones_for_non_mutation(times=times, label=label)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot the theoretical number of surviving clones from the single progenitor model
        if show_spm_fit:   # Assumes the Moran model. Timing will be wrong for the WF models.
            if spm_fit_rate is None:
                spm_fit_rate = self.division_rate
            ax.plot(times, surviving_clones_fit(times, spm_fit_rate,
                                                self.get_surviving_clones_for_non_mutation(times=[0], label=label)),
                    'r--')
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.plot(times, surviving_clones, label=legend_label, **plot_kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Surviving clones')
        ax.set_yscale("log")

    def plot_clone_size_distribution_for_non_mutation(self, t=None, label=None, legend_label=None, ax=None,
                                                      as_bar=False):
        """
        Plots the clone size distribution, with the clones defined by the clones_array - i.e. not one clone per
        mutation, one clone per unique set of mutations.
        WARNING - Only really suitable for the case of no mutations, where we want to track the growth of a number of
        initial clones over time.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if t is None:
            t = self.max_time
        csd = self.get_clone_size_distribution_for_non_mutation(t, label=label)
        csd = csd / csd[1:].sum()
        if as_bar:
            ax.bar(range(1, len(csd)), csd[1:], label=legend_label)
        else:
            ax.scatter(range(1, len(csd)), csd[1:], label=legend_label)
        ax.set_ylim([0, csd[1:].max() * 1.1])

    def plot_clone_size_scaling_for_non_mutation(self, times, markersize=2, label=None, legend_label="", ax=None):
        """Mostly useful for simulations without any mutations. For comparing to single progenitor model."""
        if ax is None:
            fig, ax = plt.subplots()
        for t in times:
            csd = self.get_clone_size_distribution_for_non_mutation(t, label=label)
            mean_ = mean_clone_size(csd)
            csd = csd / csd[1:].sum()
            revcumsum = np.cumsum(csd[::-1])[::-1]
            x = np.arange(1, len(csd)) / mean_
            ax.scatter(x, revcumsum[1:], alpha=0.5, s=markersize, label=legend_label + str(t))
        ax.legend()


def pickle_load(filename, change_sparse_to_csr=True):
    """
    Load a simulation from a gzipped pickle
    :param filename:
    :return:
    """
    with gzip.open(filename, 'rb') as f:
        sim = pickle.load(f)

    if change_sparse_to_csr:
        sim.change_sparse_to_csr()

    return sim
