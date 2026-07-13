from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import lil_matrix
from treelib import Tree

if TYPE_CHECKING:
    from ..parameters import Parameters

class SimulationSetupMixin:
    """Functions to set up simulations
    """
    def _calculate_search_times(self) -> None:
        """Create adjusted version of times to prevent floating point errors
        when searching

        Calculates slightly adjusted times so floating point errors do 
        not lead to incorrect time point selection
        To fix the cases where we are searching for the time before 
        or equal to t and we exclude t.0000000001. 
        The times are adjusted by a small value to account 
        for this floating point error. 
        
        Generally not used - by default the closest time 
        is found instead
        """
        if len(self.times) > 1:
            min_diff = np.diff(self.times).min()
        else:
            min_diff = self.times[0]
        self._search_times = self.times - min_diff / 100

    def _setup_label_times(self) -> None:
        """Convert the input label times to simulation steps 

        Also sets the next_label_time attribute to the first label time.
        """
        self.label_times = self._adjust_raw_times(self.label_times)
        if self.label_times is not None:
            self.next_label_time = self.label_times[0]
        else:
            self.next_label_time = np.inf

    def _setup_treatment(self, parameters: "Parameters") -> None:
        """Convert input treatment times to simulation steps

        Also sets the next_treatment_time attribute to the first treatment time.

        Parameters
        ----------
        parameters : Parameters
            Parameters object used to create the simulation
        """
        self.treatment_count = -1
        if parameters.treatment.treatment_timings is None:
            # No treatment applied. But this set up means the initial fitness is set correctly then not changed
            self.treatment_timings = [0, np.inf]
            self.treatment_effects = [1, 1]  # Always neutral
            self.next_treatment_time = 0
            self.treatment_replace_fitness = False
        else:
            self.treatment_timings = self._adjust_raw_times(parameters.treatment.treatment_timings)
            self.treatment_timings = list(self.treatment_timings) + [np.inf]
            self.treatment_effects = parameters.treatment.treatment_effects
            self.next_treatment_time = self.treatment_timings[0]   # First value will always be zero
            self.treatment_replace_fitness = parameters.treatment.treatment_replace_fitness

    def _setup_clone_tree(self) -> None:
        """Sets up the tree that tracks the ancestry of mutant clones."""
        self.tree = Tree()
        self.tree.create_node(str(-1), -1)  # Make a root node that isn't a clone.
    
    @abstractmethod
    def _adjust_raw_times(self, array: ArrayLike) \
            -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Converts times to number of simulation steps

        Varies depending on the algorithm, so is defined in the subclasses. 

        Parameters
        ----------
        array : ArrayLike
            Time points

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float64]]
            Array of simulation step indices
        """
        raise NotImplementedError()

    @abstractmethod
    def _precalculate_mutations(self) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.int_]]]:
        """Calculate the number of mutations to add at each simulation step

        Returns
        -------
        tuple[int, np.ndarray[tuple[int], np.dtype[np.int_]]]
            the total new_mutation_count, 
            array of the number of mutations to add at each simulation step

        """
        raise NotImplementedError()

    def _init_arrays(self, labels_array: ArrayLike | None, 
                     initial_mutant_gene_array: ArrayLike | None, 
                     input_fitness_array: ArrayLike) -> None:
        """Defines self.clones_array, self.population_array and self.raw_fitness_array

        Fills self.clones_array with any information given about the initial cells.

        Parameters
        ----------
        labels_array : ArrayLike | None
            Labels array for initial cloens
        initial_mutant_gene_array : ArrayLike | None
            Mutant genes for initial clones
        input_fitness_array : ArrayLike
            Fitness for initial clones
        """
        self.clones_array = np.zeros((self.total_clone_count, 6))
        self.clones_array[:, self.id_idx] = np.arange(len(self.clones_array))  # Give clones an identifier

        if labels_array is None:
            labels_array = 0
        self.clones_array[:self.initial_clones, self.label_idx] = labels_array

        if initial_mutant_gene_array is None:
            initial_mutant_gene_array = np.nan
        # Give each initial cell mutation type. np.nan if no mutation
        self.clones_array[:self.initial_clones, self.gene_mutated_idx] = initial_mutant_gene_array

        self.clones_array[:self.initial_clones, self.generation_born_idx] = 0
        self.clones_array[:self.initial_clones, self.parent_idx] = -1  

        # For each clone, need an raw fitness array as long as the number of genes
        # Needs to be dtype=float, which is the default of np.zeros
        if self.fitness_calculator and self.fitness_calculator.multi_gene_array:
            blank_fitness_array = np.full(
                (self.total_clone_count, self.fitness_calculator.n_cols),
                np.nan, dtype=float)
            blank_fitness_array[:, 0] = self.parameters.fitness._wt_fitness
            blank_fitness_array[:self.initial_clones, :self.fitness_calculator.n_cols] = input_fitness_array
            self.raw_fitness_array = blank_fitness_array
        else:
            blank_fitness_array = np.full((self.total_clone_count, 1), self.parameters.fitness._wt_fitness,
                                          dtype=float)
            blank_fitness_array[:self.initial_clones, 0] = input_fitness_array
            self.raw_fitness_array = blank_fitness_array

        # Create the population array that will store the clone sizes at each time point.
        self.population_array = lil_matrix((self.total_clone_count, self.sim_length))
        # Start with the initial_quantities
        if self.times[0] == 0:
            self.population_array[:self.initial_clones, 0] = self.initial_size_array.reshape(
                len(self.initial_size_array), 1)
            self.plot_idx = 1

        # Add the initial clones to the clone tree
        for i in range(self.initial_clones):
            self.tree.create_node(str(i), i, parent=-1)  # Directly descended from the root node

    def _extend_arrays_fixed_amount(self, extension: int) -> None:
        """Add new rows to the population and clones arrays. 
        
        For when labels are added.

        Parameters
        ----------
        extension : int
            The number of rows to add
        """
        s = self.population_array.shape[0]
        new_pop_array = lil_matrix((s + extension, self.sim_length))
        new_pop_array[:s] = self.population_array
        self.population_array = new_pop_array

        self.clones_array = np.concatenate([self.clones_array, 
                                            np.zeros((extension, 6))], axis=0)

        self.raw_fitness_array = np.concatenate(
            [
                self.raw_fitness_array,
                np.full((extension, self.raw_fitness_array.shape[1]), 
                        np.nan)
            ],
        axis=0)
