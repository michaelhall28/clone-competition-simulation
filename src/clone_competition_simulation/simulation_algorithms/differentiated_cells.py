"""
Classes to simulate differentiated cells along with proliferating cells in the basal layer.

It is assumed that competition only occurs between the proliferating cells. The differentiated cells do not affect the
behaviour of the proliferating cells and are not added to the grids in the 2D simulations.

Uses Cython code (diff_cell_functions.pyx) to increase speed of the differentiated cell simulations.

Not used or tested extensively, and not all functions will work well with these simulations.
"""
from dataclasses import dataclass
from typing import Self, Literal

from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import lil_matrix

try:
    import diff_cell_functions
except ImportError:
    diff_cell_functions = None

from ..analysis.analysis import mean_clone_size, mean_clone_size_fit
from ..parameters.algorithm_validation import AlgorithmClass
from ..utils import find_ge
from .base_2D_class import SpatialCurrentData
from .base_sim_class import BaseSimClass
from .branching_process import Branching
from .current_data import NonSpatialCurrentData
from .moran import Moran
from .moran2D import Moran2D
from loguru import logger


@dataclass
class DiffNonSpatialCurrentData(NonSpatialCurrentData):
    current_diff_cell_population: np.ndarray[tuple[int], np.dtype[np.int_]]

    @classmethod
    def from_sim(cls, sim: BaseSimClass) -> Self:
        """Create DiffNonSpatialCurrentData object from simulation

        Parameters
        ----------
        sim : BaseSimClass
            A non-spatial differentiated cell simulation instance

        Returns
        -------
        Self
            DiffNonSpatialCurrentData, with the initial clone population
            Starts with zero differentiated cells
        """
        current_population = np.zeros(len(sim.clones_array), dtype=int)
        current_population[:sim.initial_clones] = sim.initial_size_array
        
        non_zero_clones = np.where(current_population > 0)[0]
        current_population = current_population[non_zero_clones]
        current_diff_cell_population = np.zeros_like(current_population)

        return cls(
            current_population=current_population, 
            non_zero_clones=non_zero_clones, 
            current_diff_cell_population=current_diff_cell_population
        )

    def update(self, current_population: np.ndarray[tuple[int], np.dtype[np.int_]], 
               non_zero_clones: np.ndarray[tuple[int], np.dtype[np.int_]], 
               current_diff_cell_population: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        """Update the current data

        Parameters
        ----------
        current_population : np.ndarray[tuple[int], np.dtype[np.int_]]
            New progenitor cell count for each clone
        non_zero_clones : np.ndarray[tuple[int], np.dtype[np.int_]]
            New array of ids of surviving clones
        current_diff_cell_population : np.ndarray[tuple[int], np.dtype[np.int_]]
            New differentiated cell count for each clone
        """
        super().update(current_population=current_population, non_zero_clones=non_zero_clones)
        self.current_diff_cell_population = current_diff_cell_population

    def update_diff_cell_population_array(
            self, diff_cell_population_array: lil_matrix, plot_idx: int) -> None:
        """Update the simulation differentiated cell population array 
        with the current differentiated cell counts

        Parameters
        ----------
        diff_cell_population_array : lil_matrix
            Array storing differentiated cell counts for each clone at 
            each sample point
        plot_idx : int
            The index of the column to update
        """
        diff_cell_population_array[
            self.non_zero_clones, plot_idx] = self.current_diff_cell_population


@dataclass
class DiffSpatialCurrentData(SpatialCurrentData):
    current_diff_cell_population: np.ndarray[tuple[int], np.dtype[np.int_]]

    @classmethod
    def from_sim(cls, sim: BaseSimClass) -> Self:
        """Create DiffSpatialCurrentData object from simulation

        Parameters
        ----------
        sim : BaseSimClass
            A spatial differentiated cell simulation instance

        Returns
        -------
        Self
            DiffSpatialCurrentData, with the initial clone population
            Starts with zero differentiated cells
        """
        grid_array = np.ravel(sim.parameters.population.initial_grid)

        current_diff_cell_population = np.zeros(
            shape=len(sim.clones_array), dtype=np.int_)

        return cls(
            grid_array=grid_array, 
            current_diff_cell_population=current_diff_cell_population
        )

    def update(self, grid_array: np.ndarray[tuple[int], np.dtype[np.int_]], 
               current_diff_cell_population: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        """Update the current data

        Parameters
        ----------
        grid_array : np.ndarray[tuple[int], np.dtype[np.int_]]
            NEw clone grid array
        current_diff_cell_population : np.ndarray[tuple[int], np.dtype[np.int_]]
            New differentiated cell count for each clone
        """
        super().update(grid_array=grid_array)
        self.current_diff_cell_population = current_diff_cell_population

    @property
    def current_population(self) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Convert the grid array into an array of cell counts per clone

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of cell counts per clone
        """
        return np.bincount(self.grid_array)
    
    def update_diff_cell_population_array(
            self, diff_cell_population_array: lil_matrix, plot_idx: int) -> None:
        """Update the simulation differentiated cell population array 
        with the current differentiated cell counts

        Parameters
        ----------
        diff_cell_population_array : lil_matrix
            Array storing differentiated cell counts for each clone at 
            each sample point
        plot_idx : int
            The index of the column to update
        """
        non_zero = np.where(self.current_diff_cell_population > 0)[0]
        diff_cell_population_array[
            non_zero, plot_idx] = self.current_diff_cell_population[non_zero]


class BaseSimDiffCells(BaseSimClass):
    """
    In the single progenitor model proposed in Clayton et al 2007, 
    there are differentiated cells that remain in the basal layer for 
    a short period before differentiating
    It is assumed here that they do not affect clonal dynamics beyond 
    adjusting clone sizes. i.e. they do not affect what any other cells do.
    They are therefore simulated in a non-spatial manner and it is 
    assumed that the cells simulated normally are all progenitor cells.

    This class replaces a few functions to allow for simulation of 
    differentiated cells in the basal layer
    """

    def __init__(self, parameters):

        if diff_cell_functions is None:
            raise ImportError(
                'Cython functions for simulating differentiated cells not found. ' \
                'Please compile diff_cell_functions.pyx to use this class.')

        # r and gamma from the single progenitor model in Clayton et al
        self.r = parameters.differentiated_cells.r  # The proportion of symmetric divisions
        self.gamma = parameters.differentiated_cells.gamma   # The differentiation rate
        super().__init__(parameters)

        self.diff_cell_population = lil_matrix((self.total_clone_count,
                                                self.sim_length))  # Will store the population counts
        self.current_diff_cell_population = None
        self.diff_cell_mutant_clone_array = None
        self.basal_cell_mutant_clone_array = None

        if parameters.algorithm.algorithm_class == AlgorithmClass.MORAN:
            # This might be marginally wrong if number of simulation steps does not end exactly at desired time point
            # Will be negligible for almost all cases.
            self.time_step = self.times[-1] / parameters.times.simulation_steps  # Only used for the Moran style simulations
        self.asym_div_rate = self.division_rate / self.r * (1 - 2 * self.r)
        self.diff_born_rate = self.total_pop * self.asym_div_rate

        # If stratification occurs on a timescale much faster than the time between sample points,
        # then many of the differentiated cells will be created and stratify before being sampled,
        # and will have no effect on anything else (aside from using a little computation)
        # To speed up simulations, option to simulate differentiated cells only for periods prior to sampling
        # points where the differentiated cells have a non-negligible chance to be sampled.
        self.diff_cell_sim_switches = parameters.differentiated_cells.diff_cell_sim_switches
        if self.diff_cell_sim_switches[0] == 0:
            # B cells simulated from start
            self.sim_diff_cells = True
            self.diff_cell_switch_idx = 1
        else:
            self.sim_diff_cells = False
            self.diff_cell_switch_idx = 0
        self.next_diff_cell_switch = self.diff_cell_sim_switches[self.diff_cell_switch_idx]

    ############ Functions for running simulations ############

    def _take_sample(self, current_data: DiffNonSpatialCurrentData | DiffSpatialCurrentData) -> None:
        """Store the current state of the simulation in the population arrays.  
        
        If storing partially completed simulation states, dump the 
        simulation to a pickle. 

        Parameters
        ----------
        current_data : DiffNonSpatialCurrentData | DiffSpatialCurrentData
            Current state of the simulation
        """
        current_data.update_population_array(self.population_array, self.plot_idx)
        current_data.update_diff_cell_population_array(self.diff_cell_population, self.plot_idx)

        self.plot_idx += 1
        if self._tmp_store is not None:  # Store current state of the simulation.
            if self._store_rotation == 0:
                self.pickle_dump(self._tmp_store, current_data)
                self._store_rotation = 1
            else:
                self.pickle_dump(self._tmp_store + '1', current_data)
                self._store_rotation = 0

    def _switch_diff_cell_simulations_on_off(
            self, current_population: np.ndarray[tuple[int], np.dtype[np.int_]], 
            current_diff_population: np.ndarray[tuple[int], np.dtype[np.int_]]) \
                -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Switch between simulating differentiated cells and not. 

        If using stratification_sim_proportion<1, we only need to 
        simulate the differentiated cells that are most likely to 
        survive to a sampling time. 
        If near a sampling time, turn the diff cell simulations on.  
        If after a sampling time and not close to the next one, 
        turn the diff cell simulations off.  

        The times to switch have been calculated in advance and are 
        stored in self.diff_cell_sim_switches.

        Parameters
        ----------
        current_population : np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of current clone sizes (progenitor cells only)
        current_diff_population : np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of current clone sizes (diff cells only)

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of current clone sizes (diff cells only)
        """
        self.diff_cell_switch_idx += 1
        self.next_diff_cell_switch = self.diff_cell_sim_switches[self.diff_cell_switch_idx]
        self.sim_diff_cells = not self.sim_diff_cells
        if self.sim_diff_cells:
            current_diff_population = np.zeros_like(current_population)
        return current_diff_population

    ############ Functions for post-processing simulations ############
    def change_sparse_to_csr(self):
        """Convert lil arrays to CSR format for post-processing
        """
        if self._is_lil:
            self.population_array = self.population_array.tocsr()  
            self.diff_cell_population = self.diff_cell_population.tocsr() 
        self._is_lil = False

    def get_clone_size_distribution_for_non_mutation(
            self, t: float | None=None, index_given: bool=False, 
            label: int| None=None, include_diff_cells: bool=False) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Gets the clone size frequencies. Not normalised.

        Clones here are defined by a unique set of mutations, not per mutation.
        Therefore this is only really suitable for a simulation without 
        mutations, where we want to track the sizes of the initial clones.

        :param t: time or index of the sample to get the distribution for.
        :param index_given: True if t is the index
        :return:

        Parameters
        ----------
        t : float | None, optional
            time or index of the sample to get the distribution for. 
            By default None and the final time point is used. 
        index_given : bool, optional
            True if t is the index, not the time. By default False
        label : int | None, optional
            If not None, only include clones with this label. 
            By default None
        include_diff_cells : bool, optional
            If True, include differentiated cells in the clone sizes. 
            By default False

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Count of clones of each size. The index of the array is the 
            clone size, so the value at index 1 is the number of clones 
            of size 1, etc.
        """
        if self._is_lil:
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

        if include_diff_cells:
            if label is not None:
                b_clones = self.diff_cell_population[clones_to_select, i]
            else:
                b_clones = self.diff_cell_population[:, i]
            clones = clones + b_clones

        clones = clones.toarray().astype(int).flatten()

        counts = np.bincount(clones)
        return counts

    def _create_mutant_clone_array_diff_cell_only(self):
        """Create an array with the b cell clone sizes for each mutant 
        across the entire simulation.

        The populations will usually add up to more than the total since
        many clones will have multiple mutations.
        """
        mutant_clones = self._track_mutations(selection='non_zero')
        self.diff_cell_mutant_clone_array = lil_matrix(self.diff_cell_population.shape)
        for mutant in mutant_clones:
            self.diff_cell_mutant_clone_array[mutant] = self.diff_cell_population[mutant_clones[mutant]].sum(axis=0)

    def _create_mutant_clone_array_basal_cells(self):
        """Create an array with the basal cell clone sizes for each 
        mutant across the entire simulation.

        Includes noth the progenitor and the differentiated cells.
        The populations will usually add up to more than the total 
        since many clones will have multiple mutations
        """
        if self.mutant_clone_array is None:
            self._create_mutant_clone_array()
        if self.diff_cell_mutant_clone_array is None:
            self._create_mutant_clone_array_diff_cell_only()
        self.basal_cell_mutant_clone_array = self.mutant_clone_array + self.diff_cell_mutant_clone_array

    def get_mutant_clone_sizes(self, t: float | None=None, 
                               selection: Literal['all', 'ns', 's']='all', 
                               index_given: bool=False, 
                               gene_mutated: str | None=None,
                               include_diff_cells: bool=False, 
                               non_zero_only: bool=False) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Get an array of mutant clone sizes at a particular time

        WARNING: This may not work exactly as expected if there were multiple initial clones!

        Parameters
        ----------
        t : float | None, optional
            time/sample index, by default None and the final time will be used
        selection : Literal['all', 'ns', 's'], optional
            Set to ns or s to include only non-synonmous or synonymous 
            clones respectively. By default 'all'.
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time.
            By default False
        gene_mutated : str | None, optional
            Only return clones of this gene. Must match a gene name. 
            By default None and mutations from all genes will be included.
        include_diff_cells : bool, optional
            If True, add the differentiated cell counts to the 
            proliferative cell counts. By default False
        non_zero_only : bool, optional
            If True, only return mutants with a positive cell count.
            By default False

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Array of mutant clone sizes
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = t
        if not include_diff_cells:
            if self.mutant_clone_array is None:
                # If the mutant clone array has not been created yet, create it.
                self._create_mutant_clone_array()
            mut_array = self.mutant_clone_array
        elif include_diff_cells:
            if self.basal_cell_mutant_clone_array is None:
                self._create_mutant_clone_array_basal_cells()
            mut_array = self.basal_cell_mutant_clone_array
        # We now find all rows in the mutant clone array that we want to keep
        if selection == 'all':
            muts = set(range(self.initial_clones, len(self.clones_array)))  # Get all rows except the initial clones
        elif selection == 'ns':
            muts = self.ns_muts
        elif selection == 's':
            muts = self.s_muts
        if gene_mutated is not None:
            muts = list(muts.intersection(self.get_idx_of_gene_mutated(gene_mutated)))
        else:
            muts = list(muts)

        mutant_clones = mut_array[muts][:, i].toarray().astype(int).flatten()

        if non_zero_only:
            return mutant_clones[mutant_clones > 0]
        else:
            return mutant_clones

    def get_mutant_clone_size_distribution(
            self, t: float | None=None, 
            selection: Literal['all', 'ns', 's']='all', 
            index_given: bool=False, gene_mutated: str | None=None,
            include_diff_cells: bool=False) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Get the frequencies of mutant clone sizes. Not normalised.

        Parameters
        ----------
        t : float | None, optional
            time/sample index, by default None and the final time will be used
        selection : Literal['all', 'ns', 's'], optional
            Set to ns or s to include only non-synonmous or synonymous 
            clones respectively. By default 'all'.
        index_given : bool, optional
            True if t is an index of the sample, False if t is a time.
            By default False
        gene_mutated : str | None, optional
            Only return clones of this gene. Must match a gene name. 
            By default None and mutations from all genes will be included.
        include_diff_cells : bool, optional
            If True, add the differentiated cell counts to the 
            proliferative cell counts. By default False

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.int_]]
            Count of mutant clones of each size. The index of the array is the 
            clone size, so the value at index 1 is the number of clones 
            of size 1, etc.
        """
        if t is None:
            t = self.max_time
            index_given = False
        if not index_given:
            i = self._convert_time_to_index(t)
        else:
            i = t
        if selection == 'all':
            if self.ns_muts and not self.s_muts:
                selection = 'ns'
            elif self.s_muts and not self.ns_muts:
                selection = 's'
            elif not self.s_muts and not self.ns_muts:
                logger.debug('No mutations at all')
                return np.array([])
        elif selection == 'ns' and not self.ns_muts:
            logger.debug('No non-synonymous mutations')
            return np.array([])
        elif selection == 's' and not self.s_muts:
            logger.debug('No synonymous mutations')
            return np.array([])

        clones = self.get_mutant_clone_sizes(i, selection=selection, index_given=True,
                                                 gene_mutated=gene_mutated, include_diff_cells=include_diff_cells)

        counts = np.bincount(clones)
        return counts

    ############ Plotting functions ############
    def plot_clone_size_distribution_for_non_mutation(
            self, t: float | None=None, label: int| None=None, 
            include_diff_cells: bool=False) -> None:
        """
        Plots the clone size distribution, with the clones defined by 
        the clones_array i.e. one clone per unique set of mutations.

        WARNING - Only really suitable for the case of no mutations, 
        where we want to track the growth of a number of
        initial clones over time.

        Parameters
        ----------
        t : float | None, optional
            time or index of the sample to get the distribution for. 
            By default None and the final time point is used. 
        label : int | None, optional
            If not None, only include clones with this label. 
            By default None
        include_diff_cells : bool, optional
            If True, include differentiated cells in the clone sizes. 
            By default False
        """
        if t is None:
            t = self.max_time
        csd = self.get_clone_size_distribution_for_non_mutation(
            t, label=label, include_diff_cells=include_diff_cells
        )
        csd = csd / csd[1:].sum()
        plt.scatter(range(1, len(csd)), csd[1:])

    def plot_mean_clone_size_graph_for_non_mutation(
            self, times: ArrayLike | None=None, 
            label: int| None=None, show_spm_fit: bool=True, 
            spm_fit_rate: float | None=None, 
            legend_label: str | None=None, 
            legend_label_fit: str | None=None, 
            include_diff_cells: bool=False,
            plot_kwargs: dict | None=None, ax: axes.Axes | None=None) -> None:
        """Plot the mean clone size over time

        Follows the mean clone sizes of each row in the clone array. 
        This is a clone defined by a unique set of mutations.
        Therefore, this function is only suitable for tracking the 
        progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing.

        Parameters
        ----------
        times : ArrayLike | None, optional
            List of array of times to plot. By default None and the 
            times from the simulation will be used. 
        label : int | None, optional
            If not None, only include clones with this label. 
            By default None
        show_spm_fit : bool, optional
            Show the expected mean clone size for the single progenitor 
            model, by default True
        spm_fit_rate : float | None, optional
            The slope of the expected mean clone size. 
            By default None, and the division rate will be used. 
        legend_label : str | None, optional
            Legend label for the simulated mean clone sizes. 
            By default None
        legend_label_fit : str | None, optional
            Legend label for the expected mean clone size, by default None
        include_diff_cells : bool, optional
            If True, include differentiated cells in the clone sizes. 
            By default False
        plot_kwargs : dict | None, optional
            Any additional arguments to pass to the MatPlotLib scatter 
            function. By default None
        ax : axes.Axes | None, optional
            Axes to add the traces to. By default None and a new figure 
            will be created. 
        """
        if times is None:
            times = self.times

        means = []
        for t in times:
            means.append(mean_clone_size(
                self.get_clone_size_distribution_for_non_mutation(
                    t, label=label,include_diff_cells=include_diff_cells
                )
            ))

        if ax is None:
            fig, ax = plt.subplots()
        if show_spm_fit:
            if spm_fit_rate is None:
                spm_fit_rate = self.division_rate
            # Plot the theoretical mean clone size from the single progenitor model
            ax.plot(times, mean_clone_size_fit(times, spm_fit_rate), 
                    'r--', label=legend_label_fit)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean clone size of surviving clones')
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.scatter(times, means, label=legend_label, **plot_kwargs)


class MoranWithDiffCells(Moran, BaseSimDiffCells):
    """
    The fixed population only applies to the number of progenitor cells.
    The number of differentiated cells is allowed to vary and 
    does not effect the dynamics of the progenitor population.
    Allows for values of rho and r other than 0.5 and 0.25

    Assume we always start without any differentiated cells.
    """
    current_data_cls = DiffNonSpatialCurrentData

    def _sim_step(self, i: int, current_data: DiffNonSpatialCurrentData) \
        -> DiffNonSpatialCurrentData:
        """Run a single step fo the Moran simulation

        One cell is selected to die at random. 
        Another cell is selected to replicate and replace the dead cell
        with its offspring. 
        The replicating cell is selected in proportion with 
        its relative fitness

        Parameters
        ----------
        i : int
            The step number
        current_data : DiffNonSpatialCurrentData
            The current state of the simulation (the clone sizes)

        Returns
        -------
        DiffNonSpatialCurrentData
            The updated state of the simulation
        """

        current_population, non_zero_clones, current_diff_cell_population = (
            current_data.current_population, 
            current_data.non_zero_clones, 
            current_data.current_diff_cell_population
        )

        if i > self.next_diff_cell_switch:
            current_diff_cell_population = \
                self._switch_diff_cell_simulations_on_off(
                    current_population, current_diff_cell_population)
        if self.sim_diff_cells:
            current_diff_cell_population = \
                diff_cell_functions.bcell_cy(
                    current_population, current_diff_cell_population,
                    self.time_step, self.asym_div_rate, self.gamma
                )

        # Select population to replicate cell
        # Select random number to select which population
        birth_selector = np.random.random()
        # make cumulative list of the fitnesses
        fitness_cumsum = np.cumsum(
            current_population * self.clones_array[non_zero_clones, self.fitness_idx], 
            axis=0)
        # Pick out the selected population
        # birth_idx is the index for the current population. 
        # The clone number is non_zero_clones[birth_idx]
        birth_idx = find_ge(fitness_cumsum, birth_selector * fitness_cumsum[-1])

        # Select replaced population
        # death_idx is the index for the current population. 
        # The clone number is non_zero_clones[death_idx]
        death_selector = np.random.random()
        cumsum = np.cumsum(current_population, axis=0)
        death_idx = find_ge(cumsum, death_selector * cumsum[-1])

        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned a mutation
            new_muts = np.concatenate([[non_zero_clones[birth_idx]],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])
            # New mutation means extending the current_population.
            # Only have to add one clone to the current population. The rest with not be non-zero clones.
            current_population = np.concatenate([current_population, [1]])
            current_diff_cell_population = np.concatenate([current_diff_cell_population, [0]])
            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation
            non_zero_clones = np.concatenate([non_zero_clones, [self.next_mutation_index - 1]])
        else:
            current_population[birth_idx] += 1
        current_population[death_idx] -= 1

        if self.sim_diff_cells:
            current_diff_cell_population[death_idx] += 2  # Two differentiated cells created if progenitor cell 'dies'
        full_clone_sizes = current_population + current_diff_cell_population

        gr_z = np.nonzero(full_clone_sizes > 0)[0]  # The indices of clones alive at this point in the current pop
        non_zero_clones = non_zero_clones[gr_z]  # Convert to the original clone numbers
        current_population = current_population[gr_z]  # Only keep the currently alive clones in current pop
        current_diff_cell_population = current_diff_cell_population[gr_z]

        current_data.update(
            current_population=current_population, 
            non_zero_clones=non_zero_clones, 
            current_diff_cell_population=current_diff_cell_population
        )

        return current_data


class Moran2DWithDiffcells(Moran2D, BaseSimDiffCells):
    """
    The fixed population only applies to the number of progenitor cells.
    The number of differentiated cells is allowed to vary and 
    does not effect the dynamics of the progenitor population.
    Allows for values of rho and r other than 0.5 and 0.25

    Assume we always start without any differentiated cells.
    """

    current_data_cls = DiffSpatialCurrentData

    def _sim_step(self, i: int, current_data: DiffSpatialCurrentData) \
        -> DiffSpatialCurrentData:
        """Run a single step fo the Moran simulation

        One cell is selected to die at random. 
        Another cell is selected to replicate and replace the dead cell
        with its offspring. 
        The replicating cell is selected in proportion with 
        its relative fitness

        Parameters
        ----------
        i : int
            The step number
        current_data : DiffSpatialCurrentData
            The current state of the simulation (the cell grid)

        Returns
        -------
        DiffSpatialCurrentData
            The updated state of the simulation
        """

        grid_array, current_diff_cell_population = (
            current_data.grid_array, 
            current_data.current_diff_cell_population
        )
        current_population = current_data.current_population

        if i > self.next_diff_cell_switch:
            current_diff_cell_population = \
                self._switch_diff_cell_simulations_on_off(
                    current_population, current_diff_cell_population)
        if self.sim_diff_cells:
            current_diff_cell_population = \
                diff_cell_functions.bcell_cy(
                    current_population, current_diff_cell_population,
                    self.time_step, self.asym_div_rate, self.gamma)

        coord = self.get_differentiating_cell(i, current_data)
        death_idx = grid_array[coord]
        birth_idx = self.get_dividing_cell(coord, current_data)
        if self.mutations_to_add[i] > 0:  # If True, this division has been assigned a mutation
            new_muts = np.concatenate([[birth_idx],
                                       np.arange(self.next_mutation_index,
                                                 self.next_mutation_index + self.mutations_to_add[i] - 1)])

            self._draw_mutations_for_single_cell(new_muts)

            # Only add the last mutation
            new_cell = self.next_mutation_index - 1
        else:
            new_cell = birth_idx

        current_diff_cell_population[death_idx] += 2  # Two differentiated cells created if progenitor cell 'dies'
        grid_array[coord] = new_cell

        if i == self.sample_points[self.plot_idx] - 1:  # Must compare to -1 since increment is after this function
            grid = np.reshape(grid_array, self.grid_shape)
            self.grid_results.append(grid.copy())

        current_data.update(
            grid_array=grid_array,
            current_diff_cell_population=current_diff_cell_population
        )

        return current_data


class BranchingWithDiffCells(Branching, BaseSimDiffCells):

    def _run_for_clone(self, clone_id: int, start_time: float) -> None:
        """Simulated one  clone

        Parameters
        ----------
        clone_id : int
            Id of the clone to simulate
        start_time : float
            Birth time of the clone
        """
        self._reset_to_start(start_time)
        self._reset_diff_cell_params()
        if clone_id < len(self.initial_size_array):
            current_population = self.initial_size_array[clone_id]  # One of the initial cells
            clone_sizes = [current_population]
            clone_sizes_diff = [0]
            clone_times = []
        else:
            current_population = 1  # New mutations always start from one cell
            clone_sizes = [0, current_population]
            clone_sizes_diff = [0, 0]
            clone_times = [start_time]

        self.next_sample_time = -1
        self.time_idx = -1
        if self.time > self.max_time:
            return
        while self.time >= self.next_sample_time:
            self.time_idx += 1
            self.next_sample_time = self.times[self.time_idx]

        current_diff_cell_population = 0

        while self.time < self.max_time:
            previous_current_pop = current_population
            current_population, current_diff_cell_population, \
            intermediate_times, intermediate_b_cell_pops = self._sim_step(clone_id,  # Run step of the simulation
                                                                          current_population,
                                                                          current_diff_cell_population)
            for t, b in zip(intermediate_times, intermediate_b_cell_pops):
                clone_sizes.append(previous_current_pop)
                clone_sizes_diff.append(b)
                clone_times.append(t)

            clone_sizes.append(current_population)
            clone_sizes_diff.append(current_diff_cell_population)
            clone_times.append(self.time)

            while self._check_label_time():
                current_population = self._add_label(clone_id, current_population,
                                                     self.label_frequencies[self.label_count],
                                                     self.label_values[self.label_count])

            while self._check_treatment_time():
                self._change_treatment()
            if current_population == 0 and self.time < self.max_time:
                # The population can go extinct in this simulation. Must then stop the sim.
                last_time = self.time
                while self.max_time >= self.next_sample_time:

                    current_diff_cell_population = self._diff_cell_check_and_run(current_population,
                                                                                 current_diff_cell_population,
                                                                                 self.next_sample_time, last_time)
                    last_time = self.next_sample_time
                    self.time_idx += 1
                    if self.time_idx < len(self.times):
                        self.next_sample_time = self.times[self.time_idx]
                    else:
                        self.next_sample_time = np.inf

                    clone_times.append(last_time)
                    clone_sizes_diff.append(current_diff_cell_population)
                    clone_sizes.append(current_population)

                self.time = np.inf  # So loop ends
                clone_times.append(np.inf)

        self._record_results(clone_id, clone_sizes, clone_sizes_diff, clone_times)

    def _reset_diff_cell_params(self) -> None:
        """Reset the differentiate cell simulation state 
        """
        if self.diff_cell_sim_switches[0] == 0:
            # Differentiated cells simulated from start
            self.sim_diff_cells = True
            self.diff_cell_switch_idx = 1
        else:
            self.sim_diff_cells = False
            self.diff_cell_switch_idx = 0
        self.next_diff_cell_switch = self.diff_cell_sim_switches[self.diff_cell_switch_idx]

    def _diff_cell_check_and_run(
            self, current_population: int, 
            current_diff_cell_population: int, 
            current_time: float, last_time: float) -> int:
        """Check if differentiated cells need simulating, and if so, do so

        Parameters
        ----------
        current_population : int
            Current progenitor cell population of the clone
        current_diff_cell_population : int
            Current diff cell population of the clone
        current_time : float
            Current time. The time to simulate up to. 
        last_time : float
            The time to simulate from

        Returns
        -------
        int
            The updated number of differentiated cells in the clone
        """
        while current_time > self.next_diff_cell_switch:
            current_diff_cell_population = \
                self._switch_diff_cell_simulations_on_off(
                    current_population, current_diff_cell_population)
            
        if self.sim_diff_cells:
            current_diff_cell_population = \
                diff_cell_functions.single_gillespie_cy_with_check(
                    current_population, current_diff_cell_population,
                    time_step=current_time - last_time,
                    asym_div_rate=self.asym_div_rate,
                    gamma=self.gamma)

        return current_diff_cell_population

    def _sim_step(self, clone_id: int, current_population: int, 
                  current_diff_cell_population: int) -> tuple[int, int, list[float], list[float]]:
        """Run one simulation step for a single clone


        Parameters
        ----------
        clone_id : int
            Id of the clone
        current_population : int
            Current progenitor cell population of the clone
        current_diff_cell_population : int
            Current differentiated cell population of the clone

        Returns
        -------
        tuple[int, int, list[float], list[float]]
            Updated progenitor cell population, 
            updated diff cell population, 
            sample times passed during the step, 
            diff cell populations at each of those sample time
        """

        # Division rate is taken as r*lambda.
        # The rate of either a symmetric AA or BB division is then 2*r*lambda = 2*division_rate
        # This then matches with the Moran model.
        # This branching model required twice as many simulations steps as the Moran as the divisions and deaths
        # happen in different steps
        last_time = self.time
        self.time += np.random.exponential(1 / (current_population * 2 * self.division_rate))

        # simulate any B cells since the last symmetric division
        # All events are independent so this will not affect the A cell results.
        intermediate_times = []
        intermediate_diff_cell_pops = []
        while self.time > self.next_sample_time:
            current_diff_cell_population = self._diff_cell_check_and_run(current_population,
                                                                         current_diff_cell_population,
                                                                         self.next_sample_time, last_time)
            intermediate_times.append(self.next_sample_time)
            intermediate_diff_cell_pops.append(current_diff_cell_population)
            last_time = self.next_sample_time
            self.time_idx += 1
            if self.time_idx < len(self.times):
                self.next_sample_time = self.times[self.time_idx]
            else:
                self.next_sample_time = np.inf

        current_diff_cell_population = self._diff_cell_check_and_run(current_population, current_diff_cell_population,
                                                                     self.time, last_time)

        if self.time > self.next_rate_change_time:
            self.mutation_rate_idx += 1
            self.current_mutation_rate = self.mutation_rates[self.mutation_rate_idx][1]
            if self.mutation_rate_idx + 1 < len(self.mutation_rates):
                self.next_rate_change_time = self.mutation_rates[self.mutation_rate_idx + 1][0]
            else:
                self.next_rate_change_time = np.inf

        # Fitness=1 is balanced.
        # If a random draw from [0,2) is taken.
        # If the draw is above the fitness, the cell will die.
        # If the draw is below the fitness, the cell will divide (and possibly mutate).
        # This means the higher the fitness, the more the clone will proliferate.
        # Fitnesses above 2 are essentially infinite, the clone will not die.
        if np.random.uniform(0, 2) <= self.clones_array[clone_id, self.fitness_idx]:
            if self.current_mutation_rate > 0:
                new_muts = np.random.poisson(self.current_mutation_rate)
            else:
                new_muts = 0
            if new_muts > 0:
                if self.next_mutation_index + new_muts >= self.population_array.shape[0]:
                    self._extend_arrays(current_population, min_extension=self.population_array.shape[
                                                                             0] - self.next_mutation_index + new_muts)

                # Current population does not change. Record the existence of a new clone at this timepoint
                # and simulate later.
                parents = np.concatenate([[clone_id], np.arange(self.next_mutation_index,
                                                                self.next_mutation_index + new_muts - 1)])

                self.plot_idx = np.searchsorted(self.times, self.time)  # Make sure the "generation_born" is correct
                self._draw_mutations_for_single_cell(parents)

                # Add the last mutation for simulation later
                self.new_mutations[self.next_mutation_index - 1] = self.time
            else:
                current_population += 1
        else:
            current_population -= 1
            current_diff_cell_population += 2

        return current_population, current_diff_cell_population, \
            intermediate_times, intermediate_diff_cell_pops

    def _switch_diff_cell_simulations_on_off(
            self, current_population: int, current_diff_population: int) -> int:
        """Switch between simulating differentiated cells and not. 

        If using stratification_sim_proportion<1, we only need to 
        simulate the differentiated cells that are most likely to 
        survive to a sampling time. 
        If near a sampling time, turn the diff cell simulations on.  
        If after a sampling time and not close to the next one, 
        turn the diff cell simulations off.  

        The times to switch have been calculated in advance and are 
        stored in self.diff_cell_sim_switches.

        Parameters
        ----------
        current_population : int
            Current progenitor cell count
        current_diff_population : int
            Current diff cell count

        Returns
        -------
        int
            New diff cell count (0)
        """
        self.diff_cell_switch_idx += 1
        self.next_diff_cell_switch = self.diff_cell_sim_switches[self.diff_cell_switch_idx]
        self.sim_diff_cells = not self.sim_diff_cells
        return 0

    def _extend_arrays(self, clone_id: int, min_extension: int=1) -> None:
        """Add more rows to the progenitor population, diff cell population, 
        clones and raw fitness arrays
        
        We cannot pre-calculate the number of mutations (and therefore 
        clones) for this algorithm as the population is not fixed, 
        so we must extend the arrays once they get full.

        Base the extension size on the initial population, 
        the mutation rate and the number of remaining clones to simulate.

        Parameters
        ----------
        clone_id : int
            ID of the clone
        min_extension : int, optional
            Minimum number of rows to add, by default 1
        """
        remaining_clones = max(len(self.initial_size_array) - clone_id, 0) + len(self.new_mutations)
        starting_clones = len(self.initial_size_array)
        proportion_finished = remaining_clones / starting_clones

        chunk_increase = max(int(self.current_mutation_rate * self.division_rate *
                                 self.total_pop * proportion_finished + 1), min_extension)

        s = self.population_array.shape[0]
        new_pop_array = lil_matrix((s + chunk_increase, self.sim_length))
        new_pop_array[:s] = self.population_array
        self.population_array = new_pop_array

        self.clones_array = np.concatenate([self.clones_array, np.zeros((chunk_increase, 6))], axis=0)

        self.raw_fitness_array = np.concatenate([self.raw_fitness_array,
                                                 np.full((chunk_increase, self.raw_fitness_array.shape[1]), np.nan)],
                                                axis=0)

        new_diff_pop_array = lil_matrix((s + chunk_increase, self.sim_length))
        new_diff_pop_array[:s] = self.diff_cell_population
        self.diff_cell_population = new_diff_pop_array

    def _finish_up(self) -> None:
        """Remove unused rows from arrays
        
        Some of the plotting/post processing steps assume that all rows 
        in the arrays are used in the simulation, so remove rows that 
        have not been used
        """
        self.clones_array = self.clones_array[:self.next_mutation_index]
        self.population_array = self.population_array[:self.next_mutation_index]
        self.raw_fitness_array = self.raw_fitness_array[:self.next_mutation_index]
        self.diff_cell_population = self.diff_cell_population[:self.next_mutation_index]

    def _record_results(self, clone_id: int, clone_sizes: list[int], 
                        clone_sizes_diff: list[int], 
                        clone_times: list[float]) -> None:
        """Record the results at the point the simulation is up to.

        Parameters
        ----------
        clone_id : int
            Clone id
        clone_sizes : list[int]
            Progenitor cell counts at each of the clone_times
        clone_sizes_diff : list[int]
            Diff cell counts at each of the clone_times
        clone_times : list[float]
            List of times the clone was alive for. 
        """
        j = 0
        a = []
        b = []
        for i, t in enumerate(self.times):
            while t >= clone_times[j]:
                j += 1
            a.append(clone_sizes[j])
            b.append(clone_sizes_diff[j])

        self.population_array[clone_id] = a
        self.diff_cell_population[clone_id] = b


def set_gsl_random_seed(s: int) -> None:
    """Set the random seed for GSL random functions 
    
    Used for diff cell simulation

    Parameters
    ----------
    s : int
        Seed
    """
    diff_cell_functions.set_random_seed(s)
