import itertools
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Literal

import matplotlib.pyplot as plt
from matplotlib import axes
import numpy as np
from numpy.typing import NDArray

from ..analysis.analysis import (add_incom_to_plot, incomplete_moment,
                                 mean_clone_size, mean_clone_size_fit,
                                 surviving_clones_fit)
from ..plotting.animator import NonSpatialToGridAnimator

if TYPE_CHECKING:
    from .base_sim_class import BaseSimClass


class SimulationPlottingMixin:
    """Functions for plotting simulation results
    """
    def _get_colours(self, clones_array: np.ndarray, force_regenerate=False) -> None:
        """Generate the colours for the clones plot. 
        
        Colour depends on type (wild type/A), relative fitness and s/ns

        Args:
            clones_array (np.ndarray): The array of clones for which to generate colours.
            force_regenerate (bool, optional): Whether to force regeneration of colours. Defaults to False.

        Parameters
        ----------
        clones_array : np.ndarray
             The array of clones for which to generate colours.
        force_regenerate : bool, optional
            Whether to force regeneration of colours. Defaults to False.
        """
        if not self.colours or force_regenerate:
            rates = clones_array[:, self.fitness_idx]
            min_ = rates.min() - 0.1
            max_ = rates.max()
            self.colours = {}
            for i, clone in enumerate(clones_array):
                scaled_fitness = (clone[self.fitness_idx] - min_) / (max_ - min_)
                ns = clone[self.id_idx] in self.ns_muts
                initial = clone[self.generation_born_idx] == 0
                if self.fitness_calculator is not None:
                    last_mutated_gene = self.fitness_calculator.get_gene_name(clone[self.gene_mutated_idx])
                else:
                    last_mutated_gene = None
                self.colours[clone[self.id_idx]] = self.plot_colour_maps._get_colour(
                    fitness=scaled_fitness, label=clone[self.label_idx],
                    ns=ns, initial=initial,
                    last_mutated_gene=last_mutated_gene,
                    genes_mutated=self._get_mutated_gene_names(i)
                )

    def _get_mutated_gene_names(self, clone_id: int) -> set[str]:
        """Get the mutated genes in a clone and convert to the gene names

        Args:
            clone_id (int): Id of the clone

        Returns:
            set[str]: Set of names of the mutated genes. 

        Parameters
        ----------
        clone_id : int
            Id of the clone

        Returns
        -------
        set[str]
            Set of names of the mutated genes. 
        """
        if self.fitness_calculator is None:
            return set()
        
        # Get the non-nan entries in the raw fitness array for this clone. 
        # Skip the first column (the WT fitness). 
        # Then the index from np.where equals the gene number in the Mutation generator and 
        # we can get the gene name
        mutated_gene_numbers = np.where(~np.isnan(self.raw_fitness_array[clone_id, 1:]))[0]
        gene_names = {self.fitness_calculator.get_gene_name(gene_number) for gene_number in mutated_gene_numbers}
        gene_names.discard(None)
        return gene_names

    def get_colour(self, clone_id: int):# -> Any:
        """Get the colour for a clone_id.

        If the clone_id is not in the clones_array (can happen if 
        manually adding values to a grid), 
        create a random colour from the colourscale for it.

        Parameters
        ----------
        clone_id : int
            Id of the clone

        Returns
        -------
        Any
            Colour for the clone
        """
        if clone_id not in self.colours:
            # Not in the colours dictionary.
            if clone_id in self.clones_array[:, self.id_idx]:
                # It is a clone generated during the simulation, so can generate all of the colours
                self._get_colours(self.clones_array, force_regenerate=True)
            else:
                # A new clone_id not seen before. 
                # This is probably for some manually manipulation of grids for plotting.
                # Generate a new colour for this clone. Store for later.
                # This will ignore any complex rules for colouring. 
                # To do that, add a row to the clones_array and
                # generated the colours dictionary. 
                # Arbitrarily use the first colourmap in the plot_colour_maps for these new clones.
                colourmap = self.plot_colour_maps.colour_rules[0].colourmap
                self.colours[clone_id] = colourmap(np.random.random())

        return self.colours[clone_id]

    def muller_plot(self, plot_file: str | None=None, plot_against_time: bool=True, quick: bool=False,
                    min_size: int=1, allow_y_extension: bool=False, plot_order: list[int] | None=None,
                    figsize: tuple[int, int] | None=None, force_new_colours: bool=False, ax: axes.Axes | None=None,
                    show_mutations_with_x=True) -> axes.Axes:
        """Plots the results of the simulation over time.

        Mutations marked with X unless show_mutations_with_x=False.
        The clones will appear as growing and shrinking sideways tear drops.
        Sub-clones emerge from their parent clones

        Parameters
        ----------
        plot_file : str | None, optional
            File name to save the plot. If none, the plot will be displayed.
            If giving a file name, include the file type, e.g. "plot.png".
            By default None
        plot_against_time : bool, optional
            Set False to use the index of the sample points for x-axis 
            labelling instead of the time. By default True
        quick : bool, optional
            Runs a faster version of the plotting. Can look worse.
            By default False
        min_size : int, optional
            Show only clones which reach this number of cells.
            Showing fewer clones speeds up the plotting and can make 
            the plot cleaner. By default 1
        allow_y_extension : bool, optional
            If the population is not constant, allows the y-axis to 
            extend beyond the initial pop. 
            Only helpful for the Branching algorithms. 
            By default False
        plot_order : list[int] | None, optional
            Manually list the order of the clones in the plot. 
            By default None
        figsize : tuple[int, int] | None, optional
            Figure size for matplotlib. By default None and will use 
            the default figsize. Only relevant if ax is None.
        force_new_colours : bool, optional
            Regenerate the colours of each clone. 
            May be useful if the colours are randomly generated.
            By default False
        ax : axes.Axes | None, optional
            Axes to plot on. If None, will create a new figure and axes. 
            If given, figsize will be ignored. By default None
        show_mutations_with_x : bool, optional
            If True, will place Xs on the plot to mark the origins of clones. 
            Non-synonymous mutations will be red, 
            synonymous mutations will be blue and labelling events will be black.
            By default True

        Returns
        -------
        axes.Axes
            Axes with the plot
        """
        if self.is_lil:
            self.change_sparse_to_csr()

        # Get the colours used for the plots
        self._get_colours(self.clones_array, force_new_colours)

        if min_size > 0: 
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
            self._make_stackplot(ax, cumulative_array, plot_order, plot_against_time) 

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
                        c.append('b')  # Plot synonymous mutations with a blue X
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
            plt.savefig(plot_file)

        return ax

    def _absorb_small_clones(self, min_size=1) -> tuple[NDArray, NDArray]:
        """Absord small clones into their parents
        
        Creates a new clones_array and population_array removing 
        clones that never get larger than the minimum size.
        Clones which are too small are absorbed into their parent 
        clone so the total population remains the same.

        Parameters
        ----------
        min_size : int, optional
            Minimum clone size, by default 1. Smaller clones than this
            will be absorbed into their parent

        Returns
        -------
        tuple[NDArray, NDArray]
            Clones array, population array
        """
        clones_to_remove = set()
        new_pop_array = self.population_array.copy()
        parent_set = Counter(self.clones_array[:, self.parent_idx])
        for i in range(len(self.clones_array) - 1, -1, -1):    # Start from the youngest clones
            if new_pop_array[i].max() < min_size:
                if parent_set[i] == 0:  # If clone does not have any large descendants.
                    parent = int(self.clones_array[i, self.parent_idx])   # Find the parent of this small clone
                    new_pop_array[parent] += new_pop_array[i]  # Add the population of the small clone to the parent
                    new_pop_array[i] = 0  # Remove the population of the small clone
                    clones_to_remove.add(i)
                    parent_set[parent] -= 1
        clones_to_keep = sorted(set(range(len(self.clones_array))).difference(clones_to_remove))
        return self.clones_array[clones_to_keep], new_pop_array[clones_to_keep]

    def _get_children(self, clones_array: NDArray, idx: int) -> NDArray:
        """Return the ids of immediate subclones of given clone

        Parameters
        ----------
        clones_array : NDArray
            The clones array to use
        idx : int
            Clone ID

        Returns
        -------
        NDArray
            Immediate children of the clone
        """
        return clones_array[clones_array[:, self.parent_idx] == idx][:, self.id_idx]

    def _get_descendants_for_muller_plot(self, clones_array: NDArray, 
                                         idx: int, order: list[int]) -> None:
        """Find the subclones of the given clone. 
        
        Runs recursively until all descendants found.
        Adds clone ids to order list.
        order will be used to make the stackplot so that 
        the subclones appear from their parent clone
        Uses the clones array rather than the tree since it may 
        be filtered to remove small clones.

        Parameters
        ----------
        clones_array : NDArray
            Clones array
        idx : int
            Clone ID
        order : list[int]
            Order of clones for plotting on a Muller plot
        """
        order.append(idx)
        children = self._get_children(clones_array, idx)  # Immediate subclones of the clone idx
        np.random.shuffle(children)
        self.descendant_counts[idx] = len(children)
        for ch in children:   # Find the subclones of the subclones.
            if ch != idx:
                self._get_descendants_for_muller_plot(clones_array, ch, order)
                order.append(idx)

    def _split_populations_for_muller_plot(
            self, clones_array: NDArray, population_array: NDArray, 
            plot_order: list[int] | None=None) -> tuple[NDArray, list[int]]:
        """Breaks up the clone populations 
        so subclones appear from their parent clone in the Muller plot

        Parameters
        ----------
        clones_array : NDArray
            Clones array
        population_array : NDArray
            Population array
        plot_order : list[int] | None, optional
            Order of clones for the Muller plot, by default None

        Returns
        -------
        tuple[NDArray, list[int]]
            Clone populations and clone order for Muller plot
        """
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
            population_array[clones_array[:, self.id_idx] == o].toarray() / (self.descendant_counts[o] + 1)
            for o in order]) for order in orders], axis=0)

        plot_order = list(itertools.chain.from_iterable(orders))
        return split_pops_for_plotting, plot_order

    def _make_stackplot(self, ax: axes.Axes, 
                        cumulative_array: NDArray, 
                        plot_order: list[int], 
                        plot_against_time: bool=True) -> None:
        """Make the Muller plot stackplot using fill between

        Prevents gaps in the plot that appear when using matplotlib stackplot

        Parameters
        ----------
        ax : axes.Axes
            Axes to plot on
        cumulative_array : NDArray
            Cumulative clone populations for plotting
        plot_order : list[int]
            Clone ID order
        plot_against_time : bool, optional
            Set False to use the index of the sample points for x-axis 
            labelling instead of the time. By default True
        """
        for i in range(len(plot_order) - 1, -1, -1):   # Start from the end/top
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
                            interpolate=True, linewidth=0)

    def _make_quick_stackplot(self, ax: axes.Axes, 
                              split_pops_for_plotting: NDArray, 
                              plot_order: list[int], 
                              plot_against_time: bool=True) -> None:
        """Make the stackplot using matplotlib stackplot

        Faster, but may not look as nice as _make_stackplot

        Parameters
        ----------
        ax : axes.Axes
            Axes to plot on
        split_pops_for_plotting : NDArray
            Clone populations for plotting
        plot_order : list[int]
            Clone ID order
        plot_against_time : bool, optional
            Set False to use the index of the sample points for x-axis 
            labelling instead of the time. By default True
        """
        # Make the stackplot using matplotlib stackplot
        if plot_against_time:
            x = self.times
        else:
            x = list(range(self.sim_length))
        ax.stackplot(x, split_pops_for_plotting, colors=[self.get_colour(i) for i in plot_order])

    def plot_incomplete_moment(self, t: float | int | None=None, 
                               selection: Literal['mutations', 'ns', 's']='mutations', 
                               xlim: tuple[float, float] | None=None, 
                               ylim: tuple[float, float] | None=None, 
                               plt_file: str | None=None, sem: bool=False,
                               show_fit: bool=False, show_legend: bool=True, fit_prop: float=1,
                               min_size: int=1, errorevery: int=1, 
                               clear_previous: bool=True, show_plot: bool=False, 
                               max_size: int | None=None,
                               fit_style: str='m--', label: str='InMo', 
                               ax: axes.Axes | None=None) -> None:
        """Plot the incomplete moment

        Parameters
        ----------
        t : float | int | None, optional
            The time to plot the incomplete moment for. 
            By default None and will use the end of the simulation. 
        selection : Literal['mutations', 'ns', 's'], optional
            Which clones to include, by default 'mutations'.
            'ns': non-synonymous clones only. 
            's': synonymous clones only. 
            'mutations': clones from all mutants. 
        xlim : tuple[float, float] | None, optional
            X-limits for the plot, by default None
        ylim : tuple[float, float] | None, optional
            Y-limits for the plot, by default None
        plt_file : str | None, optional
            File to output the plot - include the file type e.g. incom_plot.pdf.
            By default None.
        sem : bool, optional
            Show the SEM on the plot, by default False
        show_fit : bool, optional
            Add a straight line fit to the log plot. 
            The intercept will be fixed at (min_size, 1).
            Will be fitted to a proportion of the data specified by fit_prop
            By default False
        show_legend : bool, optional
            Show a legend with the R^2 coefficient of the straight line fit.
            By default True
        fit_prop : float, optional
            The proportion of the data to fit the straight line on. 
            Starts from the smallest included sizes. 
            Will be the clone sizes that together contain fit_prop proportion
            of the clones.
            By default 1
        min_size : int, optional
            The smallest clone size to include. 
            All smaller clones will be ignored. 
            By default 1
        errorevery : int, optional
            If showing the SEM, will only show the errorbar 
            every `errorevery` points.
            By default 1
        clear_previous : bool, optional
            If wanting to show more on the same plot, 
            set to false and plot the other traces before running this function.
            By default True
        show_plot : bool, optional
            If needing to show the plot rather than adding more traces after.
            By default False
        max_size : int | None, optional
            Maximum clone size to include. 
            Larger clones will be excluded, not truncated.
            By default None and all clones will be included.
        fit_style : str, optional
            Style for the fit line, by default 'm--'
        label : str, optional
            Legend label for the incomplete moment trace, by default 'InMo'
        ax : axes.Axes | None, optional
            Axes to plot on. By default None and a new figure will be created. 
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
            if len(incom) > 0:
                add_incom_to_plot(incom, clone_size_dist, sem=sem, 
                                  show_fit=show_fit, fit_prop=fit_prop,
                                  min_size=min_size, label=label, 
                                  errorevery=errorevery, fit_style=fit_style, ax=ax)

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

    def _expected_incomplete_moment(self, t: float, max_n: int) -> np.ndarray:
        """The expected incomplete moment if the simulation is neutral 
        and all clones are measured accurately

        Parameters
        ----------
        t : float
            Time
        max_n : int
            Max clone size

        Returns
        -------
        np.ndarray
            Incomplete moment from clone sizes 1 to max_n
        """
        return np.exp(-np.arange(1, max_n + 1) / (self.division_rate * t))

    def plot_dnds(self, plt_file: str | None=None, min_size: int=1, gene: str | None=None, 
                  clear_previous: bool=True, legend_label: str | None=None, 
                  ax: axes.Axes | None=None) -> None:
        """Plot dN/dS ratio over time.

        :param plt_file: Output file if required. Include the output file type in the name, e.g. "out.pdf"
        :param min_size: Minimum size of clones to include.
        :param gene: Gene name. Only include mutations in this gene.
        :param clear_previous: Clear previous plot.
        :param legend_label: Label for the line in the figure.
        :param ax: ax to plot on.
        :return: None

        Parameters
        ----------
        plt_file : str | None, optional
            Output file if required. 
            Include the output file type in the name, e.g. "out.pdf", 
            By default None
        min_size : int, optional
            Minimum size of clones to include, by default 1
        gene : str | None, optional
            Only include mutations in this gene. By default None and 
            all genes will be included.
        clear_previous : bool, optional
            Clear previous plot first. By default True
        legend_label : str | None, optional
            Label for the line in the figure. By default None
        ax : axes.Axes | None, optional
            Axes to plot on. 
            By default None and a new figure will be created.
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

    def plot_overall_population(self, label: str | None=None, 
                                legend_label: str | None=None, 
                                ax: axes.Axes | None=None) -> None:
        """Plot the total or labelled cell population

        With no label, plot the total population. 
        Intended for simulations without a fixed total population
        (will also run for the fixed population, but will not be interesting)

        With a label, will plot the labelled population

        Parameters
        ----------
        label : str | None, optional
            Label. By default None and all cells will be counted
        legend_label : str | None, optional
            Text for the legend, by default None
        ax : axes.Axes | None, optional
            Axes to plot on. By default None and a new figure will 
            be created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        pop = self.get_labeled_population(label=label)
        ax.plot(self.times, pop, label=legend_label)
        ax.set_ylabel("Population")
        ax.set_xlabel("Time")

    def plot_average_fitness_over_time(
            self, legend_label: str | None=None, 
            ax: axes.Axes | None=None) -> None:
        """Plot the average fitness of the entire cell population

        Parameters
        ----------
        legend_label : str | None, optional
            Text for the legend, by default None
        ax : axes.Axes | None, optional
            Axes to plot on. By default None and a new figure will 
            be created.
        """
        if ax is None:
            fig, ax = plt.subplots()
        avg_fit = [self.get_average_fitness(t) for t in self.times]
        ax.plot(self.times, avg_fit, label=legend_label)
        ax.set_ylabel("Average fitness")
        ax.set_xlabel("Time")

    def animate(self, animation_file: str, grid_size: tuple[int, int], 
                generations_per_frame: int=1,  
                starting_clones: int=1, 
                figsize: tuple[float, float] | None=None, 
                bitrate: int=500, 
                min_prop: float=0, dpi: int=100, fps: int=5) -> None:
        """Output an animation of the simulation on a 2D grid.

        For the non-spatial simulations, will plot a 2D representation 
        of the clone proportions. This is not very
        meaningful, but may help to visualise the simulation results.

        The 2D simulations overwrite this function to plot the actual 
        spatial distribution of the clones.

        Parameters
        ----------
        animation_file : str
            Name of the output file. Needs the file type included, 
            e.g. 'video.mp4'
        grid_size : tuple[int, int]
            Size of the grid to plot on.
        generations_per_frame : int, optional
            Number of cell generations to show per video frame. 
            By default 1
        starting_clones : int, optional
            Will split initial clone cell populations into separately placed clones.
            By default 1
        figsize : tuple[float, float] | None, optional
            If None, will use  the default figsize.
        bitrate : int, optional
            Bitrate of the video, by default 500
        min_prop : float, optional
            Hides clones which occupy less than this proportion of the
            total tissue. Helps to speed up animation. By default 0
        dpi : int, optional
            DPI of the video, by default 100
        fps : int, optional
            Frames per second, by default 5
        """
        if self.is_lil:
            self.change_sparse_to_csr()

        animator = NonSpatialToGridAnimator(self, grid_size=grid_size, generations_per_frame=generations_per_frame,
                        starting_clones=starting_clones, figsize=figsize, bitrate=bitrate, min_prop=min_prop,
                        dpi=dpi, fps=fps)

        animator.animate(animation_file)

    ## Plots for lineage tracing experiments
    # These assume no mutations occurred during the simulation,
    # but all mutations (or labelled clones) are induced at the start.
    def plot_mean_clone_size_graph_for_non_mutation(
            self, times: Iterable[float] | None=None, label: int | None=None, 
            show_spm_fit: bool=True, spm_fit_rate: float | None=None,
            legend_label: str | None=None, legend_label_fit: str | None=None, 
            ax: axes.Axes | None=None, plot_kwargs: dict | None=None, 
            fit_plot_kwargs: dict | None=None) -> None:
        """Plot mean clone size over time

        Follows the mean clone sizes of each row in the clone array. 
        This is a clone defined by a unique set of mutations.
        Therefore, this function is only suitable for tracking the 
        progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.

        Parameters
        ----------
        times : Iterable[float] | None, optional
            Iterable of times to plot the mean clone size at. 
            If None, will plot for all time points.
        label : int | None, optional
            If given, will only include clones with this label. 
            By default None and all clones will be included.
        show_spm_fit : bool, optional
            Whether to show the theoretical mean clone size from the 
            single progenitor model. By default True
        spm_fit_rate : float | None, optional
            The division rate to use for the single progenitor model fit. 
            If None, will use the division rate of the simulation. 
            By default None
        legend_label : str | None, optional
            Label for the mean clone size line in the figure. By default None
        legend_label_fit : str | None, optional
            Label for the single progenitor model fit line in the 
            figure, by default None
        ax : axes.Axes | None, optional
            Axes to plot on. If None, will create a new figure and axes.
        plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the mean clone 
            size plotting function. By default None
        fit_plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the plot of the 
            fit line. By default None
        """
        if times is None:
            times = self.times

        means = []
        for t in times:
            means.append(mean_clone_size(self.get_clone_size_distribution_for_non_mutation(t, label=label)))

        if ax is None:
            fig, ax = plt.subplots()
        if show_spm_fit:
            # Plot the theoretical mean clone size from the single progenitor model
            if spm_fit_rate is None:
                spm_fit_rate = self.division_rate
            if fit_plot_kwargs is None:
                fit_plot_kwargs = {}
            ax.plot(times, mean_clone_size_fit(times, spm_fit_rate), label=legend_label_fit, **fit_plot_kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean clone size of surviving clones')
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.plot(times, means, label=legend_label, **plot_kwargs)

    def plot_surviving_clones_for_non_mutation(
            self, times: Iterable[float] | None=None, 
            ax: axes.Axes | None=None, label: int | None=None, 
            show_spm_fit: bool=False, spm_fit_rate: float | None=None, 
            plot_kwargs: dict | None=None, 
            legend_label: str | None=None) -> None:
        """Plots the number of surviving clones over time

        Follows the surviving clones based on of each row in the 
        clone array. 
        This is a clone defined by a unique set of mutations. 
        Therefore, this function is only suitable for tracking the 
        progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.

        Parameters
        ----------
        times : Iterable[float] | None, optional
            Iterable of times to plot the mean clone size at. 
            If None, will plot for all time points.
        ax : axes.Axes | None, optional
            Axes to plot on. If None, will create a new figure and axes.
        label : int | None, optional
            If given, will only include clones with this label.
        show_spm_fit : bool, optional
            Whether to show the theoretical mean clone size from 
            the single progenitor model, by default False
        spm_fit_rate : float | None, optional
            The division rate to use for the single progenitor model fit. 
            If None, will use the division rate of the simulation. 
            By default None. 
        plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the mean clone size
            plotting function. By default None
        legend_label : str | None, optional
            Label for the mean clone size line in the figure, 
            by default None
        """
        surviving_clones, times = self.get_surviving_clones_for_non_mutation(times=times, label=label)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot the theoretical number of surviving clones from the single progenitor model
        if show_spm_fit:   
            if spm_fit_rate is None:
                spm_fit_rate = self.division_rate  # Assumes the Moran model. Timing will be wrong for the WF models.
            ax.plot(times, surviving_clones_fit(times, spm_fit_rate,
                                                self.get_surviving_clones_for_non_mutation(times=[0], label=label)),
                    'r--')
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.plot(times, surviving_clones, label=legend_label, **plot_kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Surviving clones')
        ax.set_yscale("log")

    def plot_clone_size_distribution_for_non_mutation(
            self, t: float | None=None, label: int | None=None, 
            legend_label: str | None=None, ax: axes.Axes | None=None,
            as_bar: bool=False) -> None:
        """Plot the clone size distribution

        Clones here are defined by the clones_array 
        - i.e. one clone per unique set of mutations.
        WARNING - Only really suitable for the case of no mutations, 
        where we want to track the growth of the initial clones over time.

        Parameters
        ----------
        t : float | None, optional
            Time point. If None, will plot for the final time point.
        label : int | None, optional
            If given, will only include clones with this label. 
            By default None
        legend_label : str | None, optional
            Label for the mean clone size line in the figure. 
            By default None
        ax : axes.Axes | None, optional
            Axes to plot on. If None, will create a new figure and axes.
        as_bar : bool, optional
            If True, will create a bar plot, otherwise will 
            create a scatter plot. By default False (scatter)
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

    def plot_clone_size_scaling_for_non_mutation(
            self, times: Iterable[float], markersize: int=2, 
            label: int | None=None, legend_label: str ="", 
            ax: axes.Axes | None=None) -> None:
        """Plot the cumulative clone size distribution, with the axis 
        scaled by the inverse mean clone size

        Plots the cumulative clone size distribution at multiple time 
        points, with the axis scaled by 1/(mean clone size). 
        Mostly useful for simulations without any mutations. 
        For comparing to the single progenitor model.

        Parameters
        ----------
        times : Iterable[float]
            Iterable of times to plot the clone size distribution for.
        markersize : int, optional
            Size of the markers for the scatter plot, by default 2
        label : int | None, optional
            If given, will only include clones with this label, 
            by default None
        legend_label : str, optional
            Prefix for the clone size distribution labels in the legend.  
            The time will be appended. 
            By default "". 
        ax : axes.Axes | None, optional
            Axes to plot on. If None, will create a new figure and axes.
        """
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
