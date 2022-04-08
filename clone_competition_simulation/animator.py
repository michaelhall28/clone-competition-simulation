"""
Classes for making videos from the simulations. Can also be used to plot a single grid of a 2D simulation.

Intended for the 2D simulations on hexagonal grids. Also includes a class for representing the clone proportions from
a non-spatial simulation on a 2D grid. This is unreliable and unmaintained, but is left for those who are interested.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections, transforms
import subprocess
import math
import random

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # For the non-spatial 2D animation


class HexAnimator:
    """
    Turns the grids of the 2D simulations into a video.
    Colours based on the clones each cell belong to and the colourscale chosen.
    Also can be used to plot individual grids as images.
    """
    def __init__(self, sim, figxsize=5, figsize=None, dpi=100, bitrate=500, fps=5, external_call=False,
                 fixed_label_text=None, fixed_label_loc=(0, 0), fixed_label_kwargs=None,
                 show_time_label=False, time_label_units=None, time_label_decimal_places=0,
                 time_label_loc=(0, 0), time_label_kwargs=None, equal_aspect=False
                 ):
        """
        To set up,
        > h = Hexanimator(sim)

        To make a video, run
        > h.animate("outfile.mp4")
        This will make a video of the entire simulation passed when creating the Hexanimator object.

        To plot an individual grid (e.g. grid=sim.grid_results[10] to get the 10th grid from a simulation), run
        > h.plot_grid(grid)

        Alternatively, you can run
        > sim.animate(outfile)
        or
        > sim.plot_grid()
        which will create the Hexanimator object for you.


        :param sim: The simulation object (i.e. the object returned by Parameters.get_simulator()).
        :param figxsize: The x-dimension of the video/figure. If figsize not given, the y-dimension will be picked
        automatically based on the dimensions of the grid. If figsize is also defined, then figxsize will be ignored.
        :param figsize: Tuple. The figure size to use. If given, will ignore figsize.
        :param dpi: DPI for the video
        :param bitrate: Bitrate for the videp
        :param fps: Frames per second for the video.
        :param external_call: If True, will use subprocess to call ffmpeg and make the video instead of using
        matplotlib.animation.FuncAnimation. May sometimes be quicker.
        :param fixed_label_text: Text to add as a label over the video.
        :param fixed_label_loc: Tuple. The location for the fixed_label_text.
        :param fixed_label_kwargs: Any kwargs to pass to ax.text for the fixed_label_text.
        :param show_time_label: If True, will show the time of each frame overlaid on the video. The time will be based
        on the times from the simulation (which may not be the frame number).
        :param time_label_units: String, the units for the time label. Will not adjust the values, is just a string to
        follow the number. E.g. 'days', 'weeks', 'years'.
        :param time_label_decimal_places: Number of decimal places to show for the time label.
        :param time_label_loc: Tuple. Location of the time label.
        :param time_label_kwargs: Any kwargs to pass to ax.text for the time label.
        :param equal_aspect: If True, will force the aspect ratio of the x and y axes to have the same scale. However,
        this will not look equal aspect in terms of the number of cells per unit due to the tesselation of the hexagons.
        """
        self.sim = sim
        if self.sim.colours is None:
            self.sim._get_colours(self.sim.clones_array)
        self.figsize = figsize
        if self.figsize is None and figxsize is None:
            self.figxsize = 5
        else:
            self.figxsize = figxsize  # X-dim of figure. Y-dim will be automatically calculated. Alternative to figsize.
        if equal_aspect:
            self.aspect = 'equal'  # Force equal aspect on the plot axes
        else:
            self.aspect = 'auto'

        self.dpi = dpi
        self.bitrate = bitrate
        self.fps = fps
        self.external_call = external_call

        # parameters for text overlaying the video
        self.fixed_label_text = fixed_label_text
        self.fixed_label_loc = fixed_label_loc
        if fixed_label_kwargs is None:
            self.fixed_label_kwargs = {}
        else:
            self.fixed_label_kwargs = fixed_label_kwargs

        self.show_time_label = show_time_label
        self.time_label_units = time_label_units
        self.time_label_loc = time_label_loc
        self.time_label_decimal_places = time_label_decimal_places
        if time_label_kwargs is None:
            self.time_label_kwargs = {}
        else:
            self.time_label_kwargs = time_label_kwargs

    def animate(self, animation_file):
        """
        Create a video from the grid_results of a simulation.
        :param animation_file: String. Must end with the file format. E.g. "outfile.mp4"
        :return: None
        """
        if self.external_call:
            self._animate_2d_hex_external_call(animation_file)
        else:
            self._animate_2d_hex(animation_file)

    def plot_grid(self, grid, ax=None):
        """
        This function is not used in the animation but can be used to plot a grid
        :param grid: A grid from a simulation. This is designed to work for grids from the simulation used when creating
        the Hexanimator object. Other grids can be plotted, but they must have the same grid dimensions. It will also
        fail if the colourscale of the simulation cannot produce a colour for all the clone numbers in the new grid.
        :param ax: Axis to plot onto.
        :return:
        """
        self._setup_frame(ax=ax)

        # Make a list of colours
        colours = []
        for i in range(self.x):
            for j in range(self.y):
                colours.append(self.sim.colours[grid[i, j]])

        self.col.set_facecolor(colours)  # Sets the colours for each polygon

    def _get_figsize(self):
        """
        Sets up the figure size for the video/plot.
        Makes sure that there is an even number of pixels on each axis, which is needed for ffmpeg.
        :return:
        """
        if self.figsize is None:
            if self.figxsize is not None:
                self.figsize = self.figxsize, self.figxsize * self.y / self.x / 2
            else:
                raise ValueError('Either figsize or figxsize must be defined for the animator')

        self._adjust_figsize_for_dpi()

    def _adjust_figsize_for_dpi(self):
        """
        This adjusts the dpi so that the number of pixels in each axis is even (needed for ffmpeg)
        Increases the figsize slightly if needed.
        :return:
        """
        pixx = self.figsize[0]*self.dpi
        pixy = self.figsize[1]*self.dpi
        self.figsize = (
            math.ceil(pixx/2) * 2 / self.dpi, math.ceil(pixy/2) * 2 / self.dpi
        )

    def _setup_polygons_and_frame(self):
        """
        Makes all of the hexagons in the grid. The animation then works by changing the colours of the hexagons.
        Based on the matplotlib code for the hexbin plots.
        This will need updating because the "offset_position" argmument of PolyCollection is being deprecated.
        :return:
        """

        xmin, xmax = 0, self.x
        ymin, ymax = 0, self.y

        # Adding padding on the x axis
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / self.x
        sy = (ymax - ymin) / self.y

        """
        # x-position: 0, 1, 2... for even rows, 0.5, 1.5... for odd rows
        # y-position: 0, 1, 2... for even *rows*, 0.5, 1.5 ... for odd *rows*

        # first column of offsets is x coord, second column is y coord

        # e.g. for 4 x 4 grid:

           x   x   x   x
         x   x   x   x
           x   x   x   x
         x   x   x   x

        offsets = [
            [0, 0],
            [0.5, 0.5],
            [0, 1],
            [0.5, 1.5],
            ...
        ]

        """
        one_offest_col = np.array([np.tile([0, 0.5], int(self.y / 2)), np.arange(0, int(self.y / 2), 0.5)]).T
        offsets = np.concatenate([one_offest_col + [a, 0] for a in range(self.x)])

        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin

        polygon = np.zeros((6, 2), float)
        polygon[:, 0] = sx * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
        polygon[:, 1] = sy * np.array([-0.5, 0.5, 1.0, 0.5, -0.5, -1.0]) / 3.0

        self.col = collections.PolyCollection(
            [polygon],
            linewidths=[0],
            offsets=offsets,
            transOffset=transforms.IdentityTransform(),
            offset_position="data"
        )

        # Adjust the corners
        xmin = -0.5
        ymin = -0.5
        xmax = xmax
        ymax = ymax / 2
        corners = ((xmin, ymin), (xmax, ymax))
        self.ax.update_datalim(corners)
        self.col.sticky_edges.x[:] = [xmin, xmax]
        self.col.sticky_edges.y[:] = [ymin, ymax]

        plt.margins(0, 0)
        self.ax.autoscale_view(tight=True)

        # add the collection last
        self.ax.add_collection(self.col, autolim=False)

    def _add_fixed_label(self):
        self.ax.text(*self.fixed_label_loc, self.fixed_label_text, **self.fixed_label_kwargs)

    def _add_time_label(self):
        self.time_label = self.ax.text(*self.time_label_loc, "", **self.time_label_kwargs)

    def _update_time_label(self, frame_number):
        # Get the time for the frame
        t = self.sim.times[frame_number]

        # Get the right number of decimal places, add the time unit, and update the plot.
        text = "{:.{dp}f}".format(t, dp=self.time_label_decimal_places)
        if self.time_label_units is not None:
            text += " " + self.time_label_units
        self.time_label.set_text(text)

    def _setup_frame(self, ax=None):
        """
        Initial setup of the figure.
        Sets the figure size, adds the hexagons and any text labels.
        :param ax: Axis to plot onto.
        :return:
        """

        self.x, self.y = self.sim.grid.shape
        if ax is None:
            ax_given = False
            self._get_figsize()

            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.ax = self.fig.add_axes([0, 0, 1, 1], aspect=self.aspect)
        else:
            ax_given = True
            self.ax = ax

        self._setup_polygons_and_frame()
        if self.show_time_label:
            self._add_time_label()
        if self.fixed_label_text is not None:
            self._add_fixed_label()

        if not ax_given:
            plt.axis('off')

    def _first_frame(self):
        return self._update(0)

    def _animate_2d_hex(self, animation_file):
        self._setup_frame()

        ani = animation.FuncAnimation(self.fig, self._update, blit=True, init_func=self._first_frame,
                                      frames=len(self.sim.grid_results))
        ani.save(animation_file, fps=self.fps, bitrate=self.bitrate, codec="libx264", dpi=self.dpi,
                 extra_args=['-pix_fmt','yuv420p'])
        plt.close('all')

    def _animate_2d_hex_external_call(self, animation_file):
        """
        Use subprocess to call ffmpeg rather than using FuncAnimation
        Generally seems to be quicker.
        """
        self._setup_frame()

        width, height = self.fig.canvas.get_width_height()

        cmd = (
            'ffmpeg',
            '-y', '-framerate', '5',
            '-s', '{0}x{1}'.format(width, height),
            '-pix_fmt', 'yuv420p',
            '-f', 'rawvideo', '-i', '-',
            '-c:v', 'libx264',
            animation_file
        )

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        for frame in range(len(self.sim.grid_results)):
            self._update(frame)
            self.fig.canvas.draw()
            string = self.fig.canvas.tostring_argb()
            p.stdin.write(string)

        p.communicate()
        plt.close('all')

    def _update_first(self, frame_number):
        """
        Creates the first frame of the simulation.
        :param frame_number: This is going to be 0 for the videos. Allowing other values so other frames could be plotted
        with this function if needed.
        :return:
        """
        grid = self.sim.grid_results[frame_number]

        # Make a list of colours
        colours = []
        for i in range(self.x):
            for j in range(self.y):
                colours.append(self.sim.colours[grid[i, j]])

        self.col.set_facecolor(colours)  # Sets the colours for each polygon

        if self.show_time_label:
            self._update_time_label(frame_number)
            return self.col, self.time_label,
        else:
            return self.col,

    def _update(self, frame_number):
        """
        Updates the plot for the next frame of the video.
        :param frame_number: Int.
        :return:
        """
        if frame_number == 0:
            return self._update_first(frame_number)
        else:
            # Only update colours which have changed.
            grid = self.sim.grid_results[frame_number]
            grid_old = self.sim.grid_results[frame_number - 1]
            idx = 0
            for i in range(self.x):
                for j in range(self.y):
                    new_grid_val = grid[i, j]
                    if grid_old[i, j] != new_grid_val:
                        new_col = self.sim.colours[new_grid_val]
                        self.col._facecolors[idx] = new_col
                    idx += 1

            if self.show_time_label:
                self._update_time_label(frame_number)
                return self.col, self.time_label,
            else:
                return self.col,


class HexFitnessAnimator(HexAnimator):
    """
    Creates an animation of the grid showing the fitness of each cell.
    Essentially the same as HexAnimator, but uses the fitness value instead of the clone_id, has a colourmap to map
    fitness to a colour, and adds a colourbar.
    """

    def __init__(self, sim, cmap, figxsize=5, figsize=None, dpi=100, bitrate=500, fps=5, min_fitness=0):
        """

        :param sim: The simulation object (i.e. the object returned by Parameters.get_simulator()).
        :param cmap: Colourmap for mapping fitness to a colour.
        :param figxsize: The x-dimension of the video/figure. If figsize not given, the y-dimension will be picked
        automatically based on the dimensions of the grid. If figsize is also defined, then figxsize will be ignored.
        :param figsize: Tuple. The figure size to use. If given, will ignore figsize.
        :param dpi: DPI for the video
        :param bitrate: Bitrate for the videp
        :param fps: Frames per second for the video.
        :param min_fitness: Minimum fitness to use for the scaling of the colours.
        """
        self.sim = sim
        if self.sim.colours is None:
            self.sim._get_colours(self.sim.clones_array)
        self.cmap = cmap
        self.figsize = figsize
        if self.figsize is None and figxsize is None:
            self.figxsize = 5
        else:
            self.figxsize = figxsize  # X-dim of figure. Y-dim will be automatically calculated. Alternative to figsize.
        self.bitrate = bitrate
        self.dpi = dpi
        self.fps = fps
        self.min_fitness = min_fitness

    def plot_grid(self, grid, ax=None):
        """
        This function is not used in the animation but can be used to plot a grid
        :param grid: A grid of fitness values.
        :param ax: Axis to plot onto.
        :return:
        """
        self._setup_frame(ax=ax)

        # Make a list of colours
        colours = []
        for i in range(self.x):
            for j in range(self.y):
                colours.append(grid[i, j])

        self.col.set_array(np.array(colours))
        cb = self.fig.colorbar(self.col, cax=self.cbax)
        cb.set_label('Fitness')

    def plot_fitness_grid_from_clones_grid(self, clones_grid):
        """
        Converts a grid from a simulation (where each entry is a clone_id) to a grid of fitness values, then plots.
        :param clones_grid: 2D array of clone_ids. E.g. from simulation.grid_results
        :return:
        """
        fitness_grid = self._get_fitness_grid(clones_grid)
        self.plot_grid(fitness_grid)

    def _setup_frame(self, ax=None):
        """
        Initial setup of the figure.
        Sets the figure size, adds the hexagons and colourbar.
        :param ax: Axis to plot onto.
        :return:
        """
        self.x, self.y = self.sim.grid.shape
        if ax is None:
            ax_given = False
            self._get_figsize()

            self.fig, (self.ax, self.cbax) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi,
                                                          gridspec_kw={"width_ratios": [10, 1]})
        else:
            ax_given = True
            self.ax = ax

        self._setup_polygons_and_frame()

        self.col.set_cmap(self.cmap)
        self.col.set_clim([self.min_fitness, self.sim.clones_array[:, self.sim.fitness_idx].max()])

        if not ax_given:
            self.ax.axis('off')

    def _get_fitness_grid(self, clones_grid):
        """
        Converts a grid of clone_ids into a grid of cell fitness values based on the simulation (self.comp).
        :param clones_grid:
        :return:
        """
        grid_shape = clones_grid.shape
        fitness_grid = np.empty(grid_shape, dtype=float)
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                clone = clones_grid[i, j]
                fitness_grid[i, j] = self.sim.clones_array[clone, self.sim.fitness_idx]
        return fitness_grid

    def _first_frame(self):
        return self._update(0)

    def _update(self, frame_number):
        """
        Updates the plot for the next frame of the video.
        :param frame_number: Int.
        :return:
        """
        grid = self._get_fitness_grid(self.sim.grid_results[frame_number])

        # Make a list of colours
        colours = []
        for i in range(self.x):
            for j in range(self.y):
                colours.append(grid[i, j])

        self.col.set_array(np.array(colours))
        cb = self.fig.colorbar(self.col, cax=self.cbax,
                               orientation='vertical')
        cb.set_label('Fitness')
        return self.col,

    def animate(self, animation_file):
        self._setup_frame()

        ani = animation.FuncAnimation(self.fig, self._update, blit=True, init_func=self._first_frame,
                                      frames=len(self.sim.grid_results))
        ani.save(animation_file, fps=self.fps, bitrate=self.bitrate, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
        plt.close('all')


class NonSpatialToGridAnimator:
    """
    The animation has absolutely no biological meaning!
    It just takes the proportions from the non-spatial simulation and sticks them on a grid.
    The logic is rough and unreliable, so it can fail for some animations.

    It starts from the previous frame and adds and removes cells from clones until the proportions match with the
    simulated clone proportions.
    It tries to put cells from the same clone next to each other to create contiguous clones.
    It doesn't always find a solution within a reasonable time and gives up.
    """

    def __init__(self, sim, grid_size=None, generations_per_frame=1, starting_clones=1,
                 figsize=(3, 3), bitrate=500, min_prop=0, dpi=100, fps=5,
                 show_progress=False):
        self.sim = sim
        if min_prop > 0:
            self.clones_array, self.proportional_populations = absorb_small_clones_and_replace_parents(self.sim,
                                                                                                       min_prop)
        else:
            self.clones_array = self.sim.clones_array
            self.proportional_populations = self.sim.population_array / self.sim.population_array.sum(axis=0)
        self.grid_size = grid_size
        self.generations_per_frame = generations_per_frame
        self.starting_clones = starting_clones
        self.figsize = figsize
        self.bitrate = bitrate
        self.dpi = dpi
        self.fps = fps
        if self.sim.colours is None:
            self.sim._get_colours(self.sim.clones_array)  # Generate colours to ensure all are available
        colourer = lambda x: self.sim.colours[x]  # Function to convert clone id to a colour
        self.colourerfunc = np.vectorize(colourer)  # Vectorize the function
        self.show_progress = show_progress

    def animate(self, animation_file):
        # Generate a 2D visualisation of the clone populations
        if self.grid_size is None:
            print('Must provide a grid size')
            return
        plot_generations = np.arange(0, self.sim.sim_length, self.generations_per_frame)
        size = self.grid_size
        total_cells = size ** 2
        grid = np.zeros((size, size))
        self.ims = []  # List of the images for the animation
        fig = plt.figure(figsize=self.figsize)
        plt.axis('off')

        pop_dict = {0: [total_cells, 0]}  # Start with all WT

        initial_clones = self.clones_array[self.clones_array[:, self.sim.parent_idx] == -1]
        initial_clones_rows = np.argwhere(self.clones_array[:, self.sim.parent_idx] == -1)[:, 0]

        # Initial population of first clone
        start_0_pop = int(round(total_cells * self.proportional_populations[0, 0]))
        new_pop_dict = {0: [start_0_pop, 0]}  # For initial placement
        if len(initial_clones) == 2:
            start_1_pop = total_cells - start_0_pop  # population for the second clone (e.g. mutation A)
            for i in range(self.starting_clones):
                if i < start_1_pop % self.starting_clones:
                    new_pop_dict[1 + i] = [int(start_1_pop / self.starting_clones) + 1, 0]
                else:
                    new_pop_dict[1 + i] = [int(start_1_pop / self.starting_clones), 0]
            grid = self._place_clones(grid, new_pop_dict, pop_dict)  # Place the initial clones on the grid
            # Replace the temporary values with the second clone id
            other_starter_id = int(initial_clones[1, self.sim.id_idx])
            grid[grid >= 1] = other_starter_id
            pop_dict = {0: [start_0_pop, 0],
                        other_starter_id: [start_1_pop, 0]}  # For the next round
        else:
            total_placed = start_0_pop
            for row_idx, clone in zip(initial_clones_rows[1:], initial_clones[1:]):
                clone_id = clone[self.sim.id_idx]
                clone_pop = int(round(total_cells * self.proportional_populations[row_idx, 0]))
                if total_placed + clone_pop > total_cells:
                    clone_pop = total_cells - total_placed
                total_placed += clone_pop
                new_pop_dict[clone_id] = [clone_pop, 0]
            grid = self._place_clones(grid, new_pop_dict, pop_dict)
            pop_dict = new_pop_dict

        image = self._convert_grid_to_image(grid)  # Convert clone ids to colours for plotting

        im2 = plt.imshow(image, interpolation='nearest')  # Create image in matplotlib
        self.ims.append([im2])  # Add the frame to the animation list. Just the 2D

        if self.show_progress:
            print('Generation', end=" ")
        for gen in plot_generations[1:]:
            if self.show_progress:
                print(gen, end=' ', flush=True)
            new_pop_dict = self._get_clones_dict(gen, total_cells)
            grid = self._place_clones(grid, new_pop_dict, pop_dict)
            pop_dict = new_pop_dict
            image = self._convert_grid_to_image(grid)
            im2 = plt.imshow(image, interpolation='nearest')  # Create image in matplotlib
            self.ims.append([im2])  # Add the frame to the animation list. Just the 2D
        if self.show_progress:
            print()

        ani = animation.ArtistAnimation(fig, self.ims, interval=500, blit=True)
        ani.save(animation_file, fps=5, bitrate=self.bitrate)
        plt.close('all')

    def _convert_grid_to_image(self, grid):
        image = self.colourerfunc(grid)  # Apply the vectorized function
        image = np.array(image)  # Convert from tuple to np.array
        image = np.rollaxis(image, 0, 3)  # Move the axes to match the required dimensions for the image
        return image

    def _get_raw_clones_dict(self, generation):
        clones_dict = {}
        for i in range(len(self.clones_array)):
            clone = self.clones_array[i]
            pop = self.proportional_populations[i, generation]
            parent = clone[self.sim.parent_idx]
            if parent < 0:
                parent = 0
            clones_dict[clone[self.sim.id_idx]] = [pop, parent]
        return clones_dict

    def _get_clones_dict(self, generation, total_cells):
        clones_dict = self._get_raw_clones_dict(generation)  # Has population in proportions
        cells_dict = {}  # Will store populations in terms of cells to draw on grid
        for c, (pop, parent) in clones_dict.items():
            new_pop = int(round(pop * total_cells))
            cells_dict[c] = [new_pop, parent]

        # Since we need to round numbers to fit to grid, it is possible the sums will not add to the total
        # Adjust the counts to fill the grid
        total_cells_check = sum([v[0] for v in cells_dict.values()])
        if total_cells_check != total_cells:
            # The rounding to whole cells means it does not sum to one.
            # Add or remove cells until summing to one
            # Add/remove at random in proportion to size
            net_cell_diff = total_cells_check - total_cells
            adjusted_proportions = np.array([v[0] for v in clones_dict.values()]) / sum(
                [v[0] for v in clones_dict.values()])
            cumulative_orig_proportions = np.cumsum(adjusted_proportions)
            if net_cell_diff > 0:
                for i in range(net_cell_diff):
                    r = random.random()
                    for v, k in zip(cumulative_orig_proportions, clones_dict.keys()):
                        if r < v:
                            cells_dict[k][0] -= 1
                            break
            else:
                for i in range(-net_cell_diff):
                    r = random.random()
                    for v, k in zip(cumulative_orig_proportions, clones_dict.keys()):
                        if r < v:
                            cells_dict[k][0] += 1
                            break

        total_cells_check = sum([v[0] for v in cells_dict.values()])
        assert total_cells_check == total_cells, (total_cells_check, cells_dict)

        return cells_dict

    def _add_direction(self, coord, direction):
        return coord[0] + direction[0], coord[1] + direction[1]

    def _get_clones_to_reduce(self, cells_in_image, clones):
        ret = []
        total_cells = 0
        for c, (num, _) in clones.items():
            if cells_in_image[c] > num:
                ret.append(c)
                total_cells += cells_in_image[c]
        return ret, total_cells

    def _randomly_select_cell(self, grid, clone_cells, clones_to_change, clone_id, size):
        if len(clones_to_change) > 0:
            for i in np.random.permutation(len(clone_cells)):
                cell = clone_cells[i]
                if cell[0] > 0:
                    if grid[cell[0] - 1, cell[1]] in clones_to_change:
                        return cell
                if cell[0] < size - 1:
                    if grid[cell[0] + 1, cell[1]] in clones_to_change:
                        return cell
                if cell[1] > 0:
                    if grid[cell[0], cell[1] - 1] in clones_to_change:
                        return cell
                if cell[1] < size - 1:
                    if grid[cell[0], cell[1] + 1] in clones_to_change:
                        return cell
        else:
            for i in np.random.permutation(len(clone_cells)):
                cell = clone_cells[i]
                if cell[0] > 0:
                    if grid[cell[0] - 1, cell[1]] != clone_id:
                        return cell
                if cell[0] < size - 1:
                    if grid[cell[0] + 1, cell[1]] != clone_id:
                        return cell
                if cell[1] > 0:
                    if grid[cell[0], cell[1] - 1] != clone_id:
                        return cell
                if cell[1] < size - 1:
                    if grid[cell[0], cell[1] + 1] != clone_id:
                        return cell
        return None

    def _place_clones(self, grid, clones, last_image_clones):
        """
        1. Pick population that needs to add a cell
        Try:
            2. Pick a random cell from that population and check it is adjacent to a clone type to be reduced
            3. Randomly select one of those adjacent cells to replace. Go to 1.
        If 2 did not find anything:
            4. Pick a random cell from population and check it is adjacent to another clone type
            5. Replace cell if it not the last of a clone which needs to survive. Go to 1.

        param grid: the image from the previously generated round
        param clones: dictionary { id : [size, colour, parent_colour]}
         size = Total number of cells for the clone
         parent_colour used for placing first cell
        """
        size = grid.shape[0]
        cells_in_image = {c: last_image_clones[c][0] for c in last_image_clones}
        for c in clones:
            if c not in cells_in_image:
                cells_in_image[c] = 0

        i = 0
        j = 0
        last_count = sum([v[0] - cells_in_image[id_] for id_, v in clones.items() if v[0] - cells_in_image[id_] > 0])

        # Add a couple of conditions to prevent getting stuck
        max_loops = size ** 2 * 10  # Limit total loops
        max_loops_without_change = size  # Limit loops without improvement
        while i < max_loops and j < max_loops_without_change:
            cells_to_change = sum(
                [v[0] - cells_in_image[id_] for id_, v in clones.items() if v[0] - cells_in_image[id_] > 0])
            if cells_to_change == last_count:
                j += 1
            else:
                j = 0
            i += 1
            finished = True
            reduce_a_clone = False
            for id_, (num_cells, parent) in clones.items():
                if num_cells > cells_in_image[id_]:  # Need to add more cells for this clone
                    finished = False
                    clone_cells = np.argwhere(grid == id_)  # Cells of current clone
                    clones_to_reduce = []
                    for id2_, (num_cells2, parent2) in clones.items():
                        if num_cells2 < cells_in_image[id2_]:
                            # Need to reduce the number of cells for this clone
                            clones_to_reduce.append(id2_)

                    # If the first cell of a clone, first try to replace a parent cell:
                    if len(clone_cells) == 0:
                        parent_cells = np.argwhere(grid == parent)
                        if len(parent_cells) > 1 or (len(parent_cells) > 0 and parent in clones_to_reduce):
                            # Replace parent cell directly
                            adj_point = parent_cells[np.random.randint(len(parent_cells))]
                            grid[adj_point[0], adj_point[1]] = id_
                            cells_in_image[id_] += 1
                            cells_in_image[parent] -= 1
                            continue
                            # Otherwise we continue and the cell will be placed  adjacent to the parent

                    selected_cell = None  # Cell adjacent to a clone to reduce
                    if len(clone_cells) > 0:
                        selected_cell = self._randomly_select_cell(grid, clone_cells, clones_to_reduce, id_, size)

                    if selected_cell is not None:
                        # We can replace a cell which needs to be removed
                        random_direction_order = random.sample(DIRECTIONS, len(
                            DIRECTIONS))  # Randomise the order we try the directions
                        for d in random_direction_order:
                            new_point = self._add_direction(selected_cell, d)
                            if 0 <= new_point[0] < size and 0 <= new_point[1] < size:
                                clone_id_to_reduce = grid[new_point[0], new_point[1]]
                                if clone_id_to_reduce in clones_to_reduce:
                                    if cells_in_image[clone_id_to_reduce] > clones[clone_id_to_reduce][
                                        0]:  # Need to remove cells from this clone
                                        grid[new_point[0], new_point[1]] = id_
                                        cells_in_image[id_] += 1
                                        cells_in_image[clone_id_to_reduce] -= 1
                                        break
                    else:
                        clone_ids_to_reduce, cells_to_choose_from = self._get_clones_to_reduce(cells_in_image, clones)
                        if len(clone_ids_to_reduce) == 1 or cells_to_choose_from < size ** 2 / 10:  # Arbitrary cut-off
                            # Not many cells left to find and replace. Start from these rather than the expanding clones
                            reduce_a_clone = True
                            continue

                        # Allow any cell to be replaced if it is a different colour and not the last of a surviving clone
                        selected_cell = None  # Cells adjacent to a clone to reduce
                        if len(clone_cells) > 0:
                            selected_cell = self._randomly_select_cell(grid, clone_cells, [], id_, size)
                            if selected_cell is None:
                                raise Exception('Did not find any cells in clone adjacent to another clone')
                        else:
                            # Rare case where parent cell -> mutation + normal daughter
                            # Or where the parent does not have a cell in the previous generation
                            # Weren't able to replace a parent cell directly.
                            # Must place the new cell next to the parent.
                            # Or replace the grandparent/greatgrandparent
                            parent_cells = np.argwhere(grid == parent)
                            if len(parent_cells) == 0:
                                parent_row_num = np.argwhere(self.clones_array[:,
                                                             self.sim.id_idx] == parent)[0][0]
                                parent_row = self.clones_array[parent_row_num]
                                grandparent = parent_row[self.sim.parent_idx]
                                grandparent_cells = np.argwhere(grid == grandparent)
                                if len(grandparent_cells) > 0:
                                    parent_cells = grandparent_cells
                                else:
                                    grandparent_row_num = np.argwhere(self.clones_array[:,
                                                                      self.sim.id_idx] == grandparent)[0][0]
                                    grandparent_row = self.clones_array[grandparent_row_num]
                                    greatgrandparent = grandparent_row[self.sim.parent_idx]
                                    greatgrandparent_cells = np.argwhere(grid == greatgrandparent)
                                    if len(greatgrandparent_cells) > 0:
                                        parent_cells = greatgrandparent_cells
                                    else:
                                        print('no great grandparents')
                                        raise ValueError
                            selected_cell = parent_cells[np.random.randint(len(parent_cells))]

                        random_direction_order = random.sample(DIRECTIONS, len(
                            DIRECTIONS))  # Randomise the order we try the directions
                        for d in random_direction_order:
                            new_point = self._add_direction(selected_cell, d)
                            if 0 <= new_point[0] < size and 0 <= new_point[1] < size:
                                clone_id_to_reduce = grid[new_point[0], new_point[1]]
                                if clone_id_to_reduce != id_:
                                    if cells_in_image[clone_id_to_reduce] > 1:  # Not the last cell of this clone
                                        grid[new_point[0], new_point[1]] = id_
                                        cells_in_image[id_] += 1
                                        cells_in_image[clone_id_to_reduce] -= 1
                                        break
            if reduce_a_clone:
                for id_, (num_cells, parent) in clones.items():
                    if num_cells < cells_in_image[id_]:  # Need to remove cells from this clone
                        finished = False
                        clone_cells = np.argwhere(grid == id_)  # Cells of current clone
                        clones_to_increase = []
                        for id2_, (num_cells2, parent2) in clones.items():
                            if num_cells2 > cells_in_image[id2_]:
                                # Need to increase the number of cells for this clone
                                clones_to_increase.append(id2_)

                        selected_cell = None  # Cells adjacent to a clone to increase
                        if len(clone_cells) > 0:
                            selected_cell = self._randomly_select_cell(grid, clone_cells, clones_to_increase,
                                                                       id_, size)

                        if selected_cell is not None:
                            # We can replace a cell which needs to be removed
                            random_direction_order = random.sample(DIRECTIONS, len(
                                DIRECTIONS))  # Randomise the order we try the directions
                            for d in random_direction_order:
                                new_point = self._add_direction(selected_cell, d)
                                if 0 <= new_point[0] < size and 0 <= new_point[1] < size:
                                    clone_id_to_increase = grid[new_point[0], new_point[1]]
                                    if clone_id_to_increase in clones_to_increase:
                                        if cells_in_image[clone_id_to_increase] < clones[clone_id_to_increase][
                                            0]:  # Need to add cells to this clone
                                            grid[selected_cell[0], selected_cell[1]] = clone_id_to_increase
                                            cells_in_image[id_] -= 1
                                            cells_in_image[clone_id_to_increase] += 1
                                            break
                        else:
                            # Allow any cell to be replaced if it is a different colour and not the last of a surviving clone
                            selected_cell = None  # Cells adjacent to a different clone
                            if len(clone_cells) > 0:
                                selected_cell = self._randomly_select_cell(grid, clone_cells, [], id_, size)
                                if selected_cell is None:
                                    raise Exception('Did not find any cells in clone adjacent to another clone')

                            random_direction_order = random.sample(DIRECTIONS, len(
                                DIRECTIONS))  # Randomise the order we try the directions
                            for d in random_direction_order:
                                new_point = self._add_direction(selected_cell, d)
                                if 0 <= new_point[0] < size and 0 <= new_point[1] < size:
                                    clone_id_to_increase = grid[new_point[0], new_point[1]]
                                    if clone_id_to_increase != id_:
                                        grid[selected_cell[0], selected_cell[1]] = clone_id_to_increase
                                        cells_in_image[id_] -= 1
                                        cells_in_image[clone_id_to_increase] += 1
                                        break
            if finished:
                break
        if i == max_loops or j == max_loops_without_change:
            print('hit limit')
            print('Remaining changes to make:')
            if i == max_loops:
                print('Hit max limit', max_loops)
            if j == max_loops_without_change:
                print('Hit loop without improvement limit', max_loops_without_change)
            for id_, (num_cells, parent) in clones.items():
                if num_cells != cells_in_image[id_]:
                    print(id_, num_cells - cells_in_image[id_])
            print(">>Try running animation with min_prop > 0. Increasing min_prop makes the animation easier.<<")

            raise Exception("Cannot make image! Failed to place all cells.")

        return grid


def absorb_small_clones_and_replace_parents(comp, min_prop=0.01):
    """For easier animation. Remove small clones, and updates the parent array so that remaining clones appear from
    surviving clones. Slightly different from GeneralSimClass.absorb_small_clones which retains any clones which have large
    enough descendants

    Only used with the NonSpatialToGridAnimator.
    """

    clones_to_remove = set()
    new_pop_array = comp.population_array.copy()
    new_clones_array = comp.clones_array.copy()
    for i in range(len(comp.clones_array) - 1, -1, -1):  # Start from the youngest clones
        if new_pop_array[i].max() / comp.total_pop < min_prop:  # If smaller than minimum proportion
            parent = int(comp.clones_array[i, comp.parent_idx])  # Find the parent of this small clone
            new_pop_array[parent] += new_pop_array[i]  # Add the population of the small clone to the parent
            new_pop_array[i] = 0  # Remove the population of the small clone
            clones_to_remove.add(i)
            new_clones_array[new_clones_array[:, comp.parent_idx] == i, comp.parent_idx] = parent
    clones_to_keep = sorted(set(range(len(comp.clones_array))).difference(clones_to_remove))
    return new_clones_array[clones_to_keep], new_pop_array[clones_to_keep] / comp.total_pop
