import bisect
import math
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from treelib import Tree

if TYPE_CHECKING:
    from .base_sim_class import BaseSimClass


class SimulationAnalysisMixin:
    """Functions for analysing simulation results
    """
    def view_clone_info(self, include_raw_fitness: bool=False) -> pd.DataFrame:
        """Get a Pandas data from of information a about the clones

        Converts the clones array into dataframe form. 

        :param include_raw_fitness: Add the raw_fitness_array data to the dataframe. 
         This will show the fitness applied by mutations in each gene. 

        Parameters
        ----------
        include_raw_fitness : bool, optional
            If True, adds the raw_fitness_array data to the dataframe. 
            This will show the fitness applied by mutations in each gene.
            By default False

        Returns
        -------
        pd.DataFrame
            Dataframe containing information about each clone
        """
        df = pd.DataFrame({
            'clone id': pd.Series(self.clones_array[:, self.id_idx], dtype=int),
            'label': pd.Series(self.clones_array[:, self.label_idx], dtype=int),
            'fitness': pd.Series(self.clones_array[:, self.fitness_idx], dtype=float),
            'generation born': pd.Series(self.clones_array[:, self.generation_born_idx], dtype=int),
            'parent clone id': pd.Series(self.clones_array[:, self.parent_idx], dtype=int),
        })
        if self.fitness_calculator is not None:
            df['last gene mutated'] = pd.Series(
                [self.fitness_calculator.get_gene_name(g) for g in self.clones_array[:, self.gene_mutated_idx]],
                dtype=object)

        if include_raw_fitness and self.fitness_calculator is not None:
            cols = []
            if self.fitness_calculator.multi_gene_array:
                cols += ['Initial clone fitness']
            cols += [g.name for g in self.fitness_calculator.genes]
            if self.fitness_calculator.epistatics is not None:
                cols += [e.name for e in self.fitness_calculator.epistatics]
            raw_df = pd.DataFrame(self.raw_fitness_array, columns=cols)
            df = pd.concat([df, raw_df], axis=1)

        return df

    def change_sparse_to_csr(self) -> None:
        """Converts the population array to CSR.

        Required for some of the post-processing and plotting functions.
        """
        if self.is_lil:
            self.population_array = self.population_array.tocsr()
        self.is_lil = False

    def _convert_time_to_index(self, t: float, nearest: bool=True) -> int:
        """Converts a time to an index of a sample point

        Parameters
        ----------
        t : float
            Time
        nearest : bool, optional
            If True, will find the nearest sample point to the time. 
            If False, will find the index before or at the time.
            By default True

        Returns
        -------
        int
            The index of a sample point

        Raises
        ------
        ValueError
            If the bisect method doesn't find a sample at or 
            before the time
        """
        if nearest:   
            # Find nearest point to the time of interest
            return self._find_nearest(t)
        else:  
            # Find the index at or just before the time of interest
            i = bisect.bisect_right(self._search_times, t)
            if i:
                return i - 1
            raise ValueError(f"Cannot find matching index for t={t}")

    def _find_nearest(self, t: float) -> int:
        """Function to find the closest sample point to t

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        int
            The index of the closest sample point
        """
        # From stackoverflow, Demitri, https://stackoverflow.com/a/26026189
        array = self.times
        idx = np.searchsorted(array, t, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(t - array[idx - 1]) < math.fabs(t - array[idx])):
            return idx - 1
        else:
            return idx

    def get_clone_sizes_array_for_non_mutation(self, t: float | int | None=None,
                                               index_given: bool=False, label: int | None=None,
                                               exclude_zeros: bool=True) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Gets array of all clone sizes.

        Clones here are defined by a unique set of mutations, not per mutation.
        Therefore this is only really suitable for a simulation without 
        mutations, where we want to track the sizes of the initial clones.

        :param t: time or index of the sample to get the distribution for.
        :param index_given: True if t is the index
        :param label: label of the clones to include. Will return all clones if None.
        :param exclude_zeros: Remove any "dead" clones
        :return: 1D-array of clones sizes.

        Parameters
        ----------
        t : float | int | None, optional
            time or index of the sample to get the distribution for.
            By default None and the final time point is used
        index_given : bool, optional
            True if t is the index, by default False
        label : int | None, optional
            label of the clones to include. By default None and all 
            clones will be included
        exclude_zeros : bool, optional
            Remove any "dead" clones, by default True

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of clone sizes
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

        clones = clones.toarray().astype(int).flatten()
        if exclude_zeros:
            clones = clones[clones > 0]

        return clones

    def get_clone_size_distribution_for_non_mutation(
            self, t: float | int | None=None, index_given: bool=False,
            label: int | None=None, exclude_zeros: bool=True) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Gets the clone size frequencies (a NumPy bincount of clone sizes)
        
        Not normalised.

        Clones here are defined by a unique set of mutations, not per mutation.
        Therefore this is only really suitable for a simulation without 
        mutations, where we want to track the sizes of the initial clones.

        Parameters
        ----------
        t : float | int | None, optional
            time or index of the sample to get the distribution for.
            By default None and the final time point is used
        index_given : bool, optional
            True if t is the index, by default False
        label : int | None, optional
            label of the clones to include. By default None and all 
            clones will be included
        exclude_zeros : bool, optional
            Remove any "dead" clones, by default True

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Count of clones of each size. The index of the array is the 
            clone size, so the value at index 1 is the number of clones 
            of size 1, etc.
        """
        clones = self.get_clone_sizes_array_for_non_mutation(
            t=t, index_given=index_given, label=label,
            exclude_zeros=exclude_zeros)
        counts = np.bincount(clones)
        return counts

    def get_surviving_clones_for_non_mutation(
            self, times: Iterable[float] | None=None, label: int | None=None) \
                -> tuple[np.ndarray[tuple[int], np.dtype[np.int_]], Iterable[float]]:
        """Gets the surviving clone counts over time

        Follows the surviving clones based on of each row in the clone 
        array. This is a clone defined by a unique set of mutations.
        Therefore, this function is only suitable for tracking the 
        progress of clones growing without any new mutations.
        For comparing to single progenitor model in lineage tracing experiments.

        :param times: Iterable of time points to get the number of surviving clones at. If None, will use all time points.
        :param label: label of the clones to include. Will return all clones if None.
        :return: Tuple of (array of number of surviving clones at each time point, array of time points). 

        Parameters
        ----------
        times : Iterable[float] | None, optional
            Iterable of time points to get the number of surviving 
            clones at. By default None, and will use all time points.
        label : int | None, optional
            label of the clones to include. By default None and all 
            clones will be included.

        Returns
        -------
        tuple[np.ndarray[tuple[int], np.dtype[np.int_]], Iterable[float]]
            Tuple of 
            array of number of surviving clones at each time point, 
            array of time points 
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

    def get_clone_ancestors(self, clone_idx: int) -> list[int]:
        """Get the clone ids of all ancestors of a given clone.

        Parameters
        ----------
        clone_idx : int
            ID of a clone

        Returns
        -------
        list[int]
            clone ids of all ancestors of the given clone.
        """
        return [n for n in self.tree.rsearch(clone_idx)]

    def get_clone_descendants(self, clone_idx: int) -> list[int]:
        """Get the clone ids of all descendants of a given clone.

        Parameters
        ----------
        clone_idx : int
            ID of a clone

        Returns
        -------
        list[int]
            clone ids of all descendants of the given clone.
        """
        return list(self.tree.subtree(clone_idx).nodes.keys())

    def _trim_tree(self) -> tuple[Tree, list[int]]:
        """Creates a clone tree excluding unobserved clones
        
        Creates a clone family tree excluding any clones with are never
        observed and do not have any observed descendants. 
        Can greatly reduce the number of clones we have to consider 
        in an analysis. 

        Returns
        -------
        tuple[Tree, list[int]]
            The trimmed tree and list of the clones in it. 
        """
        if self.trimmed_tree is None:
            # Some clones may have appeared and died between sampling points.
            # These won't affect the results but can slow down the processing
            # Make new tree just from sampled clones.
            non_zero_sampled_clones = np.unique((self.population_array.nonzero()[0]))
            sampled_clones_set = set()
            for clone in non_zero_sampled_clones[::-1]:
                if clone not in sampled_clones_set:
                    ancestors = self.get_clone_ancestors(clone)
                    sampled_clones_set.update(ancestors)

            sampled_clones_set.remove(-1)
            trimmed_tree = Tree()
            trimmed_tree.create_node("-1", -1)
            sampled_clones = sorted(sampled_clones_set)
            for n in sorted(sampled_clones):  # For every clone that is alive at a sampling time
                for n2 in self.tree.rsearch(n):  # Find the first ancestor that was sampled. This is the new parent.
                    if n != n2 and (n2 == -1 or n2 in sampled_clones_set):
                        trimmed_tree.create_node(str(n), n, parent=n2)
                        break
            self.trimmed_tree = trimmed_tree
            self.sampled_clones = sampled_clones

        return self.trimmed_tree, self.sampled_clones

    def _get_clone_descendants_trimmed(self, clone_idx: int) -> list[int]:
        """Gets the clone descendants using the trimmed tree

        Will exclude any clones which are not observed and have no 
        observed descendants.

        Parameters
        ----------
        clone_idx : int
            ID of a clone

        Returns
        -------
        list[int]
            clone ids of descendants of the given clone.
        """
        if not self.trimmed_tree:
            self._trim_tree()
        return list(self.trimmed_tree.subtree(clone_idx).nodes.keys())

    def _track_mutations(
            self, selection: Literal['all', 's', 'ns', 'label', 'mutations', 'non_zero']='all')\
                  -> dict[int, list[int]]:
        """Get a dictionary of the clones which contain each mutation.

        :param selection: 'all', 'ns', 's', 'label', 'mutations', 'non_zero'. 'all: All clones. 
         'ns': non-synonymous clones only. 's': synonymous clones only. 
         'label': clones from a labelling event. 'mutations': clones from mutants. 
         'non_zero': clones which are observed at sample points and their ancestors.

        :return: Dict. Key: mutation id (id of first clone which contains the mutation),
                       Value: set of clone ids which contain that mutation

        Parameters
        ----------
        selection : Literal['all', 's', 'ns', 'label', 'mutations', 'non_zero'], optional
            Which clones to include, by default 'all'
            'all': All clones. 
            'ns': non-synonymous clones only. 
            's': synonymous clones only. 
            'label': clones from a labelling event. 
            'mutations': clones from mutants. 
            'non_zero': clones which are observed at sample points and their ancestors.

        Returns
        -------
        dict[int, list[int]]
            Key: mutation id (id of first clone which contains the mutation)
            Value: set of clone ids which contain that mutation

        Raises
        ------
        ValueError
            _description_
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
            # Only show observed clones. 
            self._trim_tree()  # Calcutes the clone tree with only observed clones and their ancestors.
            self._trim_tree()
            mutant_clones = {k: self._get_clone_descendants_trimmed(k) for k in self.sampled_clones}
        else:
            raise ValueError("Please select from 'all', 's', 'ns', 'label', 'mutations' or 'non_zero'")

        return mutant_clones

    def _create_mutant_clone_array(self) -> None:
        """Create an array with the clone sizes for each mutant

        The populations will usually add up to more than the total since 
        many clones will have multiple mutations
        """
        mutant_clones = self._track_mutations(selection='non_zero')
        self.mutant_clone_array = lil_matrix(self.population_array.shape)
        for mutant in mutant_clones:
            self.mutant_clone_array[mutant] = self.population_array[mutant_clones[mutant]].sum(axis=0)

    def get_idx_of_gene_mutated(self, gene_mutated: str) -> set[int]:
        """Returns a set of all clones with gene_mutated given

        Parameters
        ----------
        gene_mutated : str
            The name of the gene

        Returns
        -------
        set[int]
            The clones which contain mutations in the given gene
        """
        gene_num = self.fitness_calculator.get_gene_number(gene_mutated)
        return set(np.where(self.clones_array[:, self.gene_mutated_idx] == gene_num)[0])

    def get_mutant_clone_sizes(self, t: float | int | None=None,
                               selection: Literal['mutations', 'all', 'ns', 's', 'label']='mutations',
                               index_given: bool=False, gene_mutated: str | None=None,
                               non_zero_only: bool=False) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Get an array of mutant clone sizes at a particular time

        WARNING: This may not work exactly as expected if there were 
        multiple initial clones!

        Parameters
        ----------
        t : float | int | None, optional
            time/sample index, by default None and the final time will be used
        selection : Literal['mutations', 'all', 'ns', 's', 'label'], optional
            Which clones to include, by default 'mutations'.
            'all': All clones, including initial clones and mutants. 
            'ns': non-synonymous clones only. 
            's': synonymous clones only. 
            'label': clones from a labelling event. 
            'mutations': clones from mutants. 
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time. 
            By default False
        gene_mutated : str | None, optional
            Gene name. Only return clone sizes for mutants in this gene.
            By default None and all genes will be included
        non_zero_only : bool, optional
            If True, only return mutants with a positive cell count. 
            By default False

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of mutant clones sizes. 
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = int(t)
        if self.mutant_clone_array is None:
            # If the mutant clone array has not been created yet, create it.
            self._create_mutant_clone_array()
        
        # We now find all rows in the mutant clone array that we want to keep
        if selection == 'all':
            muts = set(range(self.initial_clones, len(self.clones_array)))   # Get all rows except the initial clones
        elif selection == 'mutations':
            muts = self.ns_muts.union(self.s_muts)
        elif selection == 'ns':
            muts = self.ns_muts
        elif selection == 's':
            muts = self.s_muts
        elif selection == 'label':
            muts = self.label_muts
        else:
            muts = set()
        if gene_mutated is not None:
            muts = list(muts.intersection(self.get_idx_of_gene_mutated(gene_mutated)))
        else:
            muts = list(muts)

        mutant_clones = self.mutant_clone_array[muts][:, i].toarray().astype(int).flatten()

        if non_zero_only:
            return mutant_clones[mutant_clones > 0]
        else:
            return mutant_clones

    def get_mutant_clone_size_distribution(self, t: float | int | None=None,
                                           selection: Literal['mutations', 'ns', 's']='mutations',
                                           index_given: bool=False, gene_mutated: str | None=None) \
                                            -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Get the frequencies of mutant clone sizes. 
        
        Not normalised.

        Parameters
        ----------
        t : float | int | None, optional
            time/sample index, by default None and the final time will be used
        selection : Literal['mutations', 'ns', 's'], optional
            Which clones to include, by default 'mutations'.
            'ns': non-synonymous clones only. 
            's': synonymous clones only. 
            'mutations': clones from all mutants. 
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time. 
            By default False
        gene_mutated : str | None, optional
            Gene name. Only return clone sizes for mutants in this gene.
            By default None and all genes will be included

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Count of clones of each size. The index of the array is the 
            clone size, so the value at index 1 is the number of clones 
            of size 1, etc.
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = int(t)
        if selection == 'mutations':
            if self.ns_muts and not self.s_muts:
                selection = 'ns'
            elif self.s_muts and not self.ns_muts:
                selection = 's'
            elif not self.s_muts and not self.ns_muts:
                return np.array([], dtype=int)
        elif selection == 'ns' and not self.ns_muts:
            return np.array([], dtype=int)
        elif selection == 's' and not self.s_muts:
            return np.array([], dtype=int)

        clones = self.get_mutant_clone_sizes(i, selection=selection, index_given=True,
                                             gene_mutated=gene_mutated)
        counts = np.bincount(clones)
        return counts

    def get_dnds(self, t: float | int | None=None, min_size: int=1, gene: str | None=None) -> float:
        """Returns the dN/dS ratio at a particular time.

        :param t: Time. If None, will be the end of the simulation.
        :param min_size: Int. The minimum size of clones to include.
        :param gene: str. The name of a gene. E.g. For getting dN/dS for a particular gene.
        :return: float. The dN/dS ratio.

        Parameters
        ----------
        t : float | int | None, optional
            Time. By default None and the final time will be used.
        min_size : int, optional
            The minimum size of clones to include, by default 1
        gene : str | None, optional
            Gene name. If given will return the dN/dS for that gene only. 
            By default None and the overall dN/dS will be return

        Returns
        -------
        float
            dN/dS ratio
        """
        if t is None:
            t = self.max_time
        ns_mut = self.get_mutant_clone_sizes(t, selection='ns', gene_mutated=gene)
        s_mut = self.get_mutant_clone_sizes(t, selection='s', gene_mutated=gene)
        ns_mut_measured = ns_mut[ns_mut >= min_size]
        total_ns = len(ns_mut_measured)
        s_mut_measured = s_mut[s_mut >= min_size]
        total_s = len(s_mut_measured)

        # If gene is None, will get the overall ns
        if gene is not None:
            # Convert gene name to gene number for the fitness calculator
            gene = self.fitness_calculator.get_gene_number(gene)
        expected_ns = total_s * (1 / self.fitness_calculator.get_synonymous_proportion(gene) - 1)
        try:
            return total_ns / expected_ns
        except ZeroDivisionError:
            return np.nan

    def get_labeled_population(self, label: int | None=None) -> NDArray:
        """Get the cell population with a particular label

        If label is None, will return the total population 
        (not interesting for the fixed population models)

        Parameters
        ----------
        label : int | None, optional
            The label of the clones to include, by default None

        Returns
        -------
        NDArray
            Array of labelled population at all time points
        """
        if label is not None:
            clones_to_select = np.where(self.clones_array[:, self.label_idx] == label)
            pop = self.population_array[clones_to_select]
        else:
            pop = self.population_array

        return pop.toarray().sum(axis=0)

    def get_mean_clone_size(self, t: float | int| None=None,
                            selection: Literal['mutations', 'all', 'ns', 's', 'label']='mutations',
                            index_given: bool=False, gene_mutated: str | None=None) -> float:
        """Get the mean mutant clone size

        Each clone is defined as the total cells containing a mutation.

        :param t: time point. If index_given=True, it is the index of the time point required.
        :param selection: 'mutations' for all mutant clones. 'ns' for non-synonymous mutations only. 's' for synonymonus
        clones only.
        :param index_given: To use with t.
        :param gene_mutated: String. If given, will limit to clones of the gene given.
        :return:

        Parameters
        ----------
        t : float | int | None, optional
            time/sample index, by default None and the final time will be used
        selection : Literal['mutations', 'all', 'ns', 's', 'label'], optional
            Which clones to include, by default 'mutations'. 
            'all': All clones, including initial clones and mutants. 
            'ns': non-synonymous clones only. 
            's': synonymous clones only. 
            'label': clones from a labelling event. 
            'mutations': clones from mutants. 
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time. 
            By default False
        gene_mutated : str | None, optional
            If given, will limit to clones of the gene given.
            By default None and all genes will be included

        Returns
        -------
        float
            The mean mutant clone size
        """
        clone_sizes = self.get_mutant_clone_sizes(t=t, selection=selection, index_given=index_given,
                                                  gene_mutated=gene_mutated)
        return clone_sizes[clone_sizes > 0].mean()

    def get_mean_clone_sizes_syn_and_non_syn(self, t: float | int | None=None, index_given: bool=False,
                                             gene_mutated: str | None=None) -> tuple[float, float]:
        """Get the mean clone sizes for the synonymous and non-synonymous mutations.

        Parameters
        ----------
        t : float | int | None, optional
            time/sample index, by default None and the final time will be used
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time. 
            By default False
        gene_mutated : str | None, optional
            If given, will limit to clones of the gene given.
            By default None and all genes will be included

        Returns
        -------
        tuple[float, float]
            Mean synonymous clone size, mean non-synonymous clone size.
        """
        mean_syn = self.get_mean_clone_size(
            t=t, selection='s', index_given=index_given, 
            gene_mutated=gene_mutated)
        mean_non_syn = self.get_mean_clone_size(
            t=t, selection='ns', index_given=index_given, 
            gene_mutated=gene_mutated)
        return mean_syn, mean_non_syn

    def get_average_fitness(self, t: float | None=None) -> float:
        """Get the average fitness of the entire population at the given time point.

        Parameters
        ----------
        t : float | None, optional
            Time. By default None and the final time will be used. 

        Returns
        -------
        float
            Average fitness value
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
