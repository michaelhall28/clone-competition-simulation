"""
A class to set up the hexagonal grids and general functions that apply to both the Moran2D and WF2D simulations.
"""
from dataclasses import dataclass
from typing import Self
import numpy as np
import matplotlib.cm as cm
from scipy.sparse import lil_matrix
from ..plotting.animator import HexAnimator, HexFitnessAnimator
from .base_sim_class import CurrentData, BaseSimClass


@dataclass
class SpatialCurrentData(CurrentData):
    grid_array: np.ndarray[tuple[int], np.dtype[np.int_]]  # The clone id of each cell in the grid

    @classmethod
    def from_sim(cls, sim: BaseSimClass) -> Self:
        grid_array = np.ravel(sim.parameters.population.initial_grid)
        return cls(
            grid_array=grid_array, 
        )

    def update(self, grid_array: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        """Update the current data

        Using a function to ensure all attributes are updated

        Args:
            current_population (np.ndarray[tuple[int], np.dtype[np.int_]]): New cell count for each clone
            grid_array (np.ndarray[tuple[int], np.dtype[np.int_]]): New layout of clones in the grid
        """
        self.grid_array = grid_array

    def update_population_array(self, population_array: lil_matrix, 
                                plot_idx: int) -> None:
        """Insert the current clone cell counts into the right rows and columns of the population array

        Args:
            population_array (lil_matrix): The array recording the clone sizes over time
            plot_idx (int): The index of the current sample point
        """
        current_population = np.bincount(self.grid_array)
        non_zero = np.where(current_population > 0)[0]
        population_array[non_zero, plot_idx] = current_population[non_zero]


class BaseHexagonalGridSim:
    """
    Contains functions that can be used with any hexagonal grid.
    Should be inherited along with the GeneralSimClass (or a subclass of it) to create a 2D simulation class.
    """
    current_data_cls = SpatialCurrentData

    def _add_label(self, current_data: CurrentData, label_frequency, label, label_fitness, label_gene) -> CurrentData:
        """
        Add some labelling at the current label frequency.
        The labelling is not exact, so each cell has same chance.
        Apply the mutants to the grid.
        """
        grid_array = current_data.grid_array

        num_labels = np.random.binomial(self.total_pop, label_frequency)

        # Extend the arrays
        self._extend_arrays_fixed_amount(num_labels)

        mutant_locs = np.random.choice(self.total_pop, num_labels, replace=False)

        for m in mutant_locs:
            # Convert the random draws into array indices
            parent = grid_array[m]
            grid_array[m] = self.next_mutation_index
            self._add_labelled_clone(parent, label, label_fitness, label_gene)

        self.label_count += 1
        if len(self.label_times) > self.label_count:
            self.next_label_time = self.label_times[self.label_count]
        else:
            self.next_label_time = np.inf

        current_data.update(
            grid_array=grid_array
        )
        return current_data

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

    def animate(self, animation_file: str, figsize=None, figxsize=5, bitrate=500, min_prop=0, external_call=False, dpi=100, fps=5,
                fitness=False, fitness_cmap=cm.Reds, min_fitness=0, fixed_label_text=None, fixed_label_loc=(0, 0),
                fixed_label_kwargs=None, show_time_label=False, time_label_units=None,
                time_label_decimal_places=0, time_label_loc=(0, 0), time_label_kwargs=None, equal_aspect=False):
        """
        Output an animation of the simulation on a 2D grid.

        :param animation_file: Output file. Needs the file type included, e.g. 'out.mp4'
        :param figsize: Figure size.
        :param figxsize: If not using figsize, this gives the x-dimension of the video. The
          y-dimension will be calculated based on the grid dimensions.
        :param bitrate: Bitrate of the video.
        :param external_call: Will run a version which is cruder and may run faster
        :param dpi: DPI of the video.
        :param fps: Frames per second.
        :param fitness: Boolean. Colour cells by their fitness instead of their clone_id.
        :param fitness_cmap: Colourmap for the fitness.
        :param min_fitness: The lower limit for the colourbar in the fitness animation.
        :param fixed_label_text: Text to add as a label over the video.
        :param fixed_label_loc: Tuple. The location for the fixed_label_text.
        :param fixed_label_kwargs: Dictionary. Any kwargs to pass to ax.text for the fixed_label_text.
        :param show_time_label: If True, will show the time of each frame overlaid on the video.
        The time will be based on the times from the simulation (which may not be the frame number).
        :param time_label_units: String, the units for the time label. 
         Will not adjust the values, is just a string to follow the number. E.g. 'days', 'weeks', 'years'.
        :param time_label_decimal_places: Number of decimal places to show for the time label.
        :param time_label_loc: Tuple. Location of the time label.
        :param time_label_kwargs: Dictionary. Any kwargs to pass to ax.text for the time label.
        :param equal_aspect: If True, will force the aspect ratio of the x and y axes to have the same scale.
         However, this will not look equal aspect in terms of the number of cells per unit due to the tesselation of the hexagons.
        :return:
        """
        if self.is_lil:
            self.change_sparse_to_csr()

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


        animator.animate(animation_file)


def get_neighbour_map(grid_shape: tuple[int, int], cell_in_own_neighbourhood: bool) -> \
        np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    """
    This creates the map of neighbouring cells for the simulation.
    Prevents the recalculation of neighbouring cells at every simulation step.
    Because of the mapping of a hexagonal grid to a 1D-array, the even and odd columns need a different mapping.
    The periodic boundary conditions are also accounted for.

    Args:
        grid_shape:
        cell_in_own_neighbourhood:

    Returns:
        Array with a map of neighbours for every position in the grid.

    """
    depth, width = grid_shape
    in_own_neighbourhood = [0] * cell_in_own_neighbourhood

    # Define the positions of the neighbours relative to each cell. Will depend on the row and column of the cell.
    even_col_base = [-width - 1, -width, -width + 1, -1, 1, width] + in_own_neighbourhood
    odd_col_base = [-width, -1, +1, width - 1, width, width + 1] + in_own_neighbourhood
    start_row_base = [+width - 1, -width, -width + 1, -1, 1, width] + in_own_neighbourhood
    end_row_base = [-width, -1, +1, width - 1, width, -width + 1] + in_own_neighbourhood

    # Combine into the positions for an entire row.
    full_row = [start_row_base] + [odd_col_base, even_col_base] * int((width - 2) / 2) + [end_row_base]

    # Multiply by the number of rows
    base_array = full_row * int(depth)

    # Add the index of the grid position to the base array of neighbours.
    # This converts the relative positions to the absolute positions of the neighbours.
    res = np.array(base_array) + np.arange(width * depth).reshape((width * depth, 1))

    # Run mod to create periodic boundary conditions.
    res = np.mod(res, (width * depth))
    return res


def get_1D_coord(row: int, col: int, grid_shape: tuple[int, int]) -> int:
    """
    Convert a 2D grid coordinate into the index for the same cell in the 1D array
    return int
    """
    return row * grid_shape[0] + col


def get_2D_coord(idx: int, grid_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Convert an index from the 1D array to a location on the 2D grid.
    return tuple: (row, column)
    """
    return idx // grid_shape[0], idx % grid_shape[0]


def get_neighbour_coords_2D(sim: BaseHexagonalGridSim, idx: int, col: int | None = None) -> \
        np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    """
    Get the neighbouring coordinates in the 2D grid from either the index in the 1D array or the coordinates in the
    2D grid.
    return array of ints
    """
    if col is not None:
        # Given the 2D coordinates
        idx = get_1D_coord(idx, col, grid_shape=sim.grid_shape)

    neighbours = sim.neighbour_map[idx]
    return np.array([get_2D_coord(n, sim.grid_shape) for n in neighbours])
