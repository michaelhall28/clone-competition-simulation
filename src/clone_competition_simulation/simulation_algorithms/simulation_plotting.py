import itertools
from collections import Counter
from typing import TYPE_CHECKING, Literal, Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.sparse import lil_matrix

from ..analysis.analysis import mean_clone_size, mean_clone_size_fit, surviving_clones_fit, incomplete_moment, add_incom_to_plot
from ..plotting.animator import NonSpatialToGridAnimator, HexAnimator, HexFitnessAnimator

if TYPE_CHECKING:
    from .base_sim_class import BaseSimClass


class SimulationPlottingMixin:
    def _get_colours(self, clones_array: np.ndarray, force_regenerate=False) -> None:
        """Generate the colours for the clones plot. Colour depends on type (wild type/A), relative fitness and s/ns

        Args:
            clones_array (np.ndarray): The array of clones for which to generate colours.
            force_regenerate (bool, optional): Whether to force regeneration of colours. Defaults to False.
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

    def get_colour(self, clone_id: int):
        """
        Return the colour for a clone_id.
        If the clone_id is not in the clones_array (can happen if manually adding values to a grid),
        create a random colour from the colourscale for it.

        :param clone_id: int.
        :return:
        """
        if clone_id not in self.colours:
            # Not in the colours dictionary.
            if clone_id in self.clones_array[:, self.id_idx]:
                # It is a clone generated during the simulation, so can generate all of the colours
                self._get_colours(self.clones_array, force_regenerate=True)
            else:
                # A new clone_id not seen before. This is probably for some manually manipulation of grids for plotting.
                # Generate a new colour for this clone. Store for later.
                # This will ignore any complex rules for colouring. To do that, add a row to the clones_array and
                # generated the colours dictionary. 
                # Arbitrarily use the first colourmap in the plot_colour_maps for these new clones.
                colourmap = self.plot_colour_maps.colour_rules[0].colourmap
                self.colours[clone_id] = colourmap(np.random.random())

        return self.colours[clone_id]

    def muller_plot(self, plot_file: str | None=None, plot_against_time: bool=True, quick: bool=False,
                    min_size: int=1, allow_y_extension: bool=False, plot_order: list[int] | None=None,
                    figsize: tuple[int, int] | None=None, force_new_colours: bool=False, ax: plt.Axes | None=None,
                    show_mutations_with_x=True) -> plt.Axes:
        """
        Plots the results of the simulation over time.
        Mutations marked with X unless show_mutations_with_x=False.
        The clones will appear as growing and shrinking sideways tear drops.
        Sub-clones emerge from their parent clones

        :param plot_file: File name to save the plot. If none, the plot will be displayed.
        If a file name, include the file type, e.g. "output_plot.png"
        :param plot_against_time: Bool. Set False to use the index of the sample points for x-axis labelling instead of the time. 
        :param quick: Bool. Runs a faster version of the plotting which can look worse
        :param min_size: Show only clones which reach this number of cells.
         Showing fewer clones speeds up the plotting and can make the plot clearer.
        :param allow_y_extension: If the population is not constant, allows the y-axis to extend beyond the initial pop. 
         Only relevant for the Branching algorithms. 
        :param plot_order: Manually list the order of the clones in the plot. 
        :param figsize: Figure size for matplotlib. If None, will use the default figsize. Only relevant if ax is None.
        :param force_new_colours: Regenerate the colours of each clone. May be useful if the colours are randomly generated.
        :param ax: Axes to plot on. If None, will create a new figure and axes. If given, figsize will be ignored.
        :param show_mutations_with_x: If True, will place Xs on the plot to mark the origins of clones. 
            Non-synonymous mutations will be red, synonymous mutations will be blue and labelling events will be black.

        :return: ax: matplotlib.axes.Axes
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

    def _absorb_small_clones(self, min_size=1):
        """Creates a new clones_array and population_array removing clones that never get larger than the
        minimum proportion min_prop.
        Clones which are too small are absorbed into their parent clone so the total population remains the same.
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

    def _get_children(self, clones_array, idx):
        """Return the ids of immediate subclones of the given clone idx"""
        return clones_array[clones_array[:, self.parent_idx] == idx][:, self.id_idx]

    def _get_descendants_for_muller_plot(self, clones_array, idx, order):
        """
        Find the subclones of the given clone. Runs recursively until all descendants found.
        Adds clone ids to order list.
        order will be used to make the stackplot so that the subclones appear from their parent clone
        Uses the clones array rather than the tree since it may be filtered to remove small clones.
        """
        order.append(idx)
        children = self._get_children(clones_array, idx)  # Immediate subclones of the clone idx
        np.random.shuffle(children)
        self.descendant_counts[idx] = len(children)
        for ch in children:   # Find the subclones of the subclones.
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
            population_array[clones_array[:, self.id_idx] == o].toarray() / (self.descendant_counts[o] + 1)
            for o in order]) for order in orders], axis=0)

        plot_order = list(itertools.chain.from_iterable(orders))
        return split_pops_for_plotting, plot_order

    def _make_stackplot(self, ax, cumulative_array, plot_order, plot_against_time=True):
        # Make the stackplot using fill between. Prevents gaps in the plot that appear with using matplotlib stackplot
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

    def _make_quick_stackplot(self, ax, split_pops_for_plotting, plot_order, plot_against_time=True):
        # Make the stackplot using matplotlib stackplot
        if plot_against_time:
            x = self.times
        else:
            x = list(range(self.sim_length))
        ax.stackplot(x, split_pops_for_plotting, colors=[self.get_colour(i) for i in plot_order])

    def plot_incomplete_moment(self, t: float | int | None=None, selection: Literal['mutations', 'ns', 's']='mutations', 
                               xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, 
                               plt_file: str | None=None, sem: bool=False,
                               show_fit: bool=False, show_legend: bool=True, fit_prop: float=1,
                               min_size: int=1, errorevery: int=1, clear_previous: bool=True, show_plot: bool=False, 
                               max_size: int | None=None,
                               fit_style: str='m--', label: str='InMo', ax: plt.Axes | None=None) -> None:
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

    def _expected_incomplete_moment(self, t: float, max_n: int) -> np.ndarray:
        """The expected incomplete moment if the simulation is neutral and all clones are measured accurately"""
        return np.exp(-np.arange(1, max_n + 1) / (self.division_rate * t))

    def plot_dnds(self, plt_file: str | None=None, min_size: int=1, gene: str | None=None, 
                  clear_previous: bool=True, legend_label: str | None=None, 
                  ax: plt.Axes | None=None) -> None:
        """
        Plot dN/dS ratio over time.

        :param plt_file: Output file if required. Include the output file type in the name, e.g. "out.pdf"
        :param min_size: Minimum size of clones to include.
        :param gene: Gene name. Only include mutations in this gene.
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

    def plot_overall_population(self, label: str | None=None, legend_label: str | None=None, ax: plt.Axes | None=None) -> None:
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

    def plot_average_fitness_over_time(self, legend_label: str | None=None, ax: plt.Axes | None=None) -> None:
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

    def animate(self, animation_file: str, grid_size: tuple[int, int], generations_per_frame: int=1,  
                starting_clones: int=1, figsize: tuple[float, float] | None=None, bitrate: int=500, 
                min_prop: float=0, dpi: int=100, fps: int=5) -> None:
        """
        Output an animation of the simulation on a 2D grid.

        For the non-spatial simulations, will plot a 2D representation of the clone proportions. This is not very
        meaningful, but may help to visualise the simulation results.
        The 2D simulations overwrite this function to plot the actual spatial distribution of the clones.

        :param animation_file: Name of the output file. Needs the file type included, e.g. 'out.mp4'
        :param grid_size: tuple[int, int] - Size of the grid to plot on.
        :param generations_per_frame: Int. Number of cell generations to show per video frame. 
        :param starting_clones: Int. Can split initial clone cell populations into separately placed clones.
        :param figsize: Figure size. If None, will use  the default figsize. 
        :param bitrate: Bitrate of the video.
        :param min_prop: Hides clones which occupy less than this proportion of the
         total tissue. Helps to speed up animation.
        :param dpi: DPI of the video.
        :param fps: Frames per second.
        :return:
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
    def plot_mean_clone_size_graph_for_non_mutation(self, times: Iterable[float] | None=None, label: int | None=None, 
                                                    show_spm_fit: bool=True, spm_fit_rate: float | None=None,
                                                    legend_label: str | None=None, legend_label_fit: str | None=None, 
                                                    ax: plt.Axes | None=None,
                                                    plot_kwargs: dict | None=None, fit_plot_kwargs: dict | None=None):
        """
        Follows the mean clone sizes of each row in the clone array. This is a clone defined by a unique set of
        mutations, not be a particular mutation.
        Therefore, this function is only suitable for tracking the progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.
        :param times: Iterable of times to plot the mean clone size at. If None, will plot for all time points.
        :param label: Int. If given, will only include clones with this label.
        :param show_spm_fit: Bool. Whether to show the theoretical mean clone size from the single progenitor model.
        :param spm_fit_rate: Float. The division rate to use for the single progenitor model fit. If None, will use 
         the division rate of the simulation. 
        :param legend_label: Label for the mean clone size line in the figure.
        :param legend_label_fit: Label for the single progenitor model fit line in the figure.
        :param ax: Axes to plot on. If None, will create a new figure and axes.
        :param plot_kwargs: Dict. Additional keyword arguments to pass to the mean clone size plotting function.
        :param fit_plot_kwargs: Dict. Additional keyword arguments to pass to the plot of the fit line. 

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

    def plot_surviving_clones_for_non_mutation(self, times: Iterable[float] | None=None, 
                                               ax: plt.Axes | None=None, 
                                               label: int | None=None, show_spm_fit: bool=False,
                                               spm_fit_rate: float | None=None, 
                                               plot_kwargs: dict | None=None, legend_label: str | None=None) -> None:
        """
        Follows the surviving clones based on of each row in the clone array. This is a clone defined by a unique set of
        mutations, not be a particular mutation.
        Therefore, this function is only suitable for tracking the progress of clones growing without any mutations.
        For comparing to single progenitor model in lineage tracing experiments.
        :param times: Iterable of times to plot the mean clone size at. If None, will plot for all time points.
        :param label: Int. If given, will only include clones with this label.
        :param show_spm_fit: Bool. Whether to show the theoretical mean clone size from the single progenitor model.
        :param spm_fit_rate: Float. The division rate to use for the single progenitor model fit. If None, will use 
         the division rate of the simulation. 
        :param legend_label: Label for the mean clone size line in the figure.
        :param ax: Axes to plot on. If None, will create a new figure and axes.
        :param plot_kwargs: Dict. Additional keyword arguments to pass to the mean clone size plotting function.
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

    def plot_clone_size_distribution_for_non_mutation(self, t: float | None=None, 
                                                      label: int | None=None, legend_label: str | None=None, 
                                                      ax: plt.Axes | None=None,
                                                      as_bar: bool=False) -> None:
        """
        Plots the clone size distribution, with the clones defined by the clones_array - i.e. not one clone per
        mutation, one clone per unique set of mutations.
        WARNING - Only really suitable for the case of no mutations, where we want to track the growth of a number of
        initial clones over time.

        :param t: Float. Time point.If None, will plot for the final time point.
        :param label: Int. If given, will only include clones with this label.
        :param legend_label: Label for the mean clone size line in the figure.
        :param ax: Axes to plot on. If None, will create a new figure and axes.
        :param as_bar: Bool. Whether to use a bar plot (True) or scatter plot (False, default). 
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

    def plot_clone_size_scaling_for_non_mutation(self, times: Iterable[float], markersize: int=2, 
                                                 label: int | None=None, legend_label: str ="", 
                                                 ax: plt.Axes | None=None) -> None:
        """
        Plots the cumulative clone size distribution at multiple time points, with the axis scaled by 
        the mean clone size. 
        Mostly useful for simulations without any mutations. For comparing to single progenitor model.
        
        :param times: Iterable of times to plot the clone size distribution for.
        :param markersize: Int. Size of the markers for the scatter plot.
        :param label: Int. If given, will only include clones with this label.
        :param legend_label: Label for the mean clone size line in the figure.
        :param ax: Axes to plot on. If None, will create a new figure and axes.
        :param as_bar: Bool. Whether to use a bar plot (True) or scatter plot (False, default). 
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
