from clone_competition_simulation.fitness_classes import MutationGenerator, NormalDist, UnboundedFitness, Gene
from clone_competition_simulation.colourscales import get_random_colourscale
from clone_competition_simulation.wf import WrightFisherSim
from clone_competition_simulation.moran import MoranSim
from clone_competition_simulation.moran2D import Moran2D
from clone_competition_simulation.branching_process import SimpleBranchingProcess
from clone_competition_simulation.wf2D import WrightFisher2D
from clone_competition_simulation.general_differentiated_cell_class import Moran2DWithDiffcells, MoranWithDiffCells, BranchingWithDiffCells
from clone_competition_simulation.stop_conditions import WFStop, WF2DStop, MoranStop, Moran2DStop
import sys
import numpy as np
from scipy.stats import expon


class ParameterException(Exception):
    pass


class Parameters(object):
    """
    Defines the parameters for a simulation.
    Performs some checks on the parameters.

    get_simulator()
        Sets up and returns a simulator with the requested parameters.
    """
    algorithm_options = ('WF', 'Moran', 'Moran2D', 'Branching', 'WF2D')
    spatial_algorithms = ('Moran2D', 'WF2D')
    moran_algorithms = ('Moran', 'Moran2D')
    wf_algorithms = ('WF', 'WF2D')

    def __init__(self,
                 algorithm=None,
                 initial_cells=None,
                 grid_shape=None,  # 2D simulations only
                 initial_size_array=None,
                 initial_grid=None,  # 2D simulations only

                 max_time=None,
                 times=None,
                 division_rate=None,
                 simulation_steps=None,  # Alternative to supplying max_time and division_rate
                 samples=None,

                 fitness_array=None,
                 label_array=None,  # Like a YFP label
                 initial_mutant_gene_array=None,  # If multiple clones at the start, this labels each with a mutant gene

                 mutation_generator=None,
                 mutation_rates=None,

                 label_times=None,  # For labeling at times after the start. Array (e.g. np.array([3, 7]))
                 label_frequencies=None,  # The proportion of cells to be labelled. List or array (e.g. [0.01, 0.02])
                 label_values=None,  # The values for the new labels. List or array (e.g. [1, 2])
                 label_fitness=None,  # Fitness of the labelled cell, for if accompanied by a mutation.
                 label_genes=None,  # Gene to be mutated with the label applied

                 treatment_timings=None,
                 treatment_effects=None,
                 treatment_replace_fitness=False,

                 figsize=None,
                 colourscales=None,

                 cell_in_own_neighbourhood=False,  # 2D only. A cell can divide to replace itself. 7 cell neighbourhood.

                 r=None,
                 gamma=None,
                 stratification_sim_percentile=None,

                 population_limit=None,  # For the branching model, will stop if exceeding this limit
                 end_condition_function=None,

                 tmp_store=None,  # File to store the partial results of simulation.

                 progress=False,
                 print_warnings=True,  # Will warn when using a default parameter or calculating one

                 ):
        """

        :param algorithm: string. Name of algorithm to run. 'Moran', 'Moran2D', 'WF', 'WF2D' or 'Branching'.

            Define the cells/clones to simulate. Only supply one of the following arguments.
        :param initial_cells: Number of cells to start with. Will all be assumed to be wildtype.
        :param grid_shape: (x, y). For 2D simulations. Defines shape of the grid. Must be even numbers because of how
        the hexagonal grid is defined.
        :param initial_size_array: List/tuple/np.array of integers. For starting a (non-spatial) simulation with
        multiple clones. Defines the number of cells in each clone. The attributes of each clone can be defined using
        fitness_array, label_array, initial_mutant_gene_array
        :param initial_grid: 2D np.array of integers. For starting a 2D simulation with
        multiple clones. Defines the starting location of the cells from each clone using a grid of the clone labels.
        The labels are the index of each clone in the arrays fitness_array, label_array and initial_mutant_gene_array.

            Define length of time to run simulation, and which time to recored results:
        :param max_time: Time at which the simulation stops. Used along with the division rate to calculate the number
        of simulation steps to run. Default=10.
        :param times: Ordered list/tuple/np.array of float/int. Alternative to max_time. Will run until the last time
        and will record the results at every time in the list.
        :param division_rate: Float/int. Default=1.
        :param simulation_steps: Alternative to using max_time/times and division_rate to define the length of
        simulations, can directly specify how many simulation steps to run.
        :param samples: Int. Number of times to record the results of the simulation. Will divide the time evenly.
        May record fewer samples if there are fewer simulation steps than samples requested. Default=100.

            Define the attributes of the initial clones. For non-spatial simulations, these arrays should be the same
            length as initial_size_array and define the clones in that order. For spatial simulations, the clones in
            initial_grid are defined by the index of these arrays, e.g. the first value in these arrays defines the
            attributes of cells marked with a 0 in initial_grid, the second value defines 1 in the initial grid etc.
        :param fitness_array: List/np.array of floats. The fitness of the initial cells.
        If multi_gene_array is used for the mutation_generator, then this should be a 2D array of fitness with one row
        per clone, and one column for WT then one column for each gene defined in the mutation_generator (in order).
        Non-mutated genes should be np.nan.
        :param label_array: List of int. Inheritable label for the initial clones. For plotting/counting label
        populations to compare to lineage tracing experiments.
        :param initial_mutant_gene_array: List of int. For associating the initial clones with a particular mutation
        defined in the mutation_generator. Index of the gene in the mutation_generator genes list. Will add this to
        the clones_array for these clones.

            Define the mutations to appear during the simulation.
        :param mutation_generator: fitness_classes.MutationGenerator. Defines the effects of mutations and how they
        combine within a clone.
        :param mutation_rates: Float or 2D np.array. Can give a single float for the mutation rate per division.
        Or can provide a 2D array of times (first column) and mutation rates (second column) to define a varying
        mutation rate. The times define the start of the accompanying mutation rate, so the last rate will continue
        until the end of the simulation.

            Define labels that appear at some point during the simulations. The labels are inherited and are like
            fluorescent labels in lineage tracing experiments.
        :param label_times: Float or list of floats. Time(s) to apply the labels.
        :param label_frequencies: Float or list of floats. Proportion(s) of cells that randomly acquire the label.
        :param label_values: Int or list of ints. Integer label for the clone.
        :param label_fitness: Float/None or list of floats/None. Fitness of the labelled cell, for if label is
        accompanied by a mutation. If mutation_generator.multi_gene_array = False, then the fitness will replace the
        previous fitness (will not combine with any mutations previously in the cell).
        If mutation_generator.multi_gene_array = True, will replace the label_gene fitness if defined, otherwise will
        replace the WT fitness, and will be combined with the effect of any mutations in other genes.
        :param label_genes: Int or list of ints. Index of the gene (as defined in the mutation_generator) mutated
        with the label.

            Define any treatment (changes to clone fitness during the simulations.) Either defines fitness changes for
            any initial clones (with no further mutations), or defines fitness changes per gene (can have mutations if
            using multi_gene_array=True in the mutation_generator.
        :param treatment_timings: List of floats. Time to start each treatment.
        :param treatment_effects: List of lists or 2D np.array of floats . For each treatment, list/array of values to
        multiply or replace the fitness of that clone/gene. Must in same order as initial_size_array or
        mutation_generator.genes. If treatment_replace_fitness=False, use value of 1 or np.nan to not change fitness.
        If treatment_replace_fitness=True, use np.nan to not change fitness.
        :param treatment_replace_fitness: Boolean. If True, the new effects of the treatment will replace the previous
        fitness of the clones. If False, will multiply the previous fitness by the new values.

            Parameters for plotting output.
        :param figsize: (float, float). Size of the output figures. Can be overwritten for most figures.
        :param colourscales: ColourScale class for defining the colours in the figures and animation. The colour of each
        clone can be defined using various attributes of the clone.

            2D neighbour definitions. Currently, only hexagonal grids with periodic boundary conditions are available.
        :param cell_in_own_neighbourhood: Boolean. Include the cell location as one of its own neighbours. Makes more
        sense for the WF2D simulations than the Moran2D. Default=False.

            Parameters for simulating differentiated basal cell numbers. Only for Moran, Moran2D and Branching algorithms.
        :param r: Float. Symmetric division/differentiation rate.
        :param gamma: Float. Stratification_rate.
        :param stratification_sim_percentile: Float in (0, 1]. If the gap between sample times is much larger than the
        mean stratification time of the differentiated cells, then many differentiated cells will be born and stratify
        before being sampled. Setting this value to less than 1 will only simulate differentiated cells for times where
        the chance of them surviving to the next sample point is greater than the stratification_sim_percentile.

            Conditions to end simulations early. Useful for ABC or to find first occurence of mutant combinations.
        :param population_limit: Int. For Branching process only. Stops simulating clones if the population limit is reached.
        :param end_condition_function: Function that takes a simulation object as the only argument and
        raises EndConditionError if the simulation should stop.


        :param tmp_store: String - filepath. File to store the partial results of simulation.
        Useful for very long simulations so they can be restarted from a sample point.

            Logging/print outputs.
        :param progress: Int. Prints after this number of simulation steps (not for the Branching algorithm).
        :param print_warnings: Boolean. Prints a warning message about parameters that were defined in this class using
        default values

        """

        self.algorithm = algorithm
        self.max_time = max_time
        self.division_rate = division_rate
        self.mutation_generator = mutation_generator
        self.initial_cells = initial_cells
        self.initial_size_array = initial_size_array
        self.fitness_array = fitness_array
        self.label_array = label_array
        self.gene_label_array = initial_mutant_gene_array
        self.mutation_rates = mutation_rates
        self.samples = samples
        self.figsize = figsize
        self.colourscales = colourscales
        self.progress = progress
        self.grid_shape = grid_shape # 2D simulations only
        self.initial_grid = initial_grid  # 2D simulations only
        self.simulation_steps = simulation_steps  # Alternative to supplying max_time and division_rate (for some algorithms)
        self.tmp_store = tmp_store
        # self.direction_bias = direction_bias
        self.label_times = label_times
        self.label_frequencies = label_frequencies
        self.label_values = label_values
        self.label_fitness = label_fitness
        self.label_genes = label_genes
        self.treatment_timings = treatment_timings
        self.treatment_effects = treatment_effects
        self.treatment_replace_fitness = treatment_replace_fitness
        self.r = r
        self.gamma = gamma
        self.stratification_sim_percentile = stratification_sim_percentile
        self.times = times
        self.population_limit = population_limit
        self.end_condition_function = end_condition_function
        self.cell_in_own_neighbourhood = cell_in_own_neighbourhood

        self.non_zero_calc = True  # Needs to be true for some algorithms, false for others. Will be set automatically
        # Essentially, if the algorithm uses the entire current population to calculate the next step,
        # this should speed it up.

        # Values to calculate from those provided
        self.sample_points = None  # Array of simulation_steps at which the samples are taken
        self.diff_sim_starts = None  # Times to simulated differentiated cells (if at all)
        self.diff_sim_ends = None

        # DEFAULT VALUES. If any of the required parameters are not provided, the defaults will be used
        self.default_division_rate = 1
        self.default_max_time = 10
        self.default_mutation_generator = MutationGenerator(combine_mutations='multiply', multi_gene_array=False,
                                                            genes=(Gene('all', NormalDist(0.1),
                                                                        synonymous_proportion=0.5, weight=1),),
                                                            mutation_combination_class=UnboundedFitness())
        self.default_mutation_rate = 0
        self.default_samples = 100
        self.default_figsize = (10, 10)
        self.default_colourscales = get_random_colourscale(None)
        self.default_label = 0
        self.default_fitness = 1
        self.default_mutation_type = -1  # Not associated with any of the genes
        self.default_large_grid = True
        self.default_wraparound_grid = False
        self.default_stratification_sim_percentile = 1

        # Other. Attributes used internally to help printing etc.
        self.initial_grid_provided = False
        self.print_warnings = print_warnings
        self.warnings = []

        if not self._check_parameters():
            sys.exit(1)
        self.sim_class = None
        self._select_simulator_class()

    def __str__(self):
        s = "Parameters:"
        s += "\n\tAlgorithm: {0}".format(self.algorithm)
        s += "\n\tMax time: {0}".format(self.max_time)
        s += "\n\tDivision rate: {0}".format(self.division_rate)
        s += "\n\tMutation generator: {0}".format(self.mutation_generator)
        s += "\n\tInitial cells: {0}".format(self.initial_cells)
        s += "\n\tInitial size array: {0}".format(self.initial_size_array)
        s += "\n\tInitial fitness array: {0}".format(self.fitness_array)
        s += "\n\tInitial label array: {0}".format(self.label_array)
        s += "\n\tInitial mutation types array: {0}".format(self.gene_label_array)
        s += "\n\tMutation rate: {0}".format(self.mutation_rates)
        s += "\n\tNumber of samples: {0}".format(self.samples)
        s += "\n\tFigsize: {0}".format(self.figsize)
        s += "\n\tColourscales: {0}".format(self.colourscales)
        s += "\n\tProgress reporting: {0}".format(self.progress)
        s += "\n\tSimulation steps: {0}".format(self.simulation_steps)
        if self.algorithm in self.spatial_algorithms:
            s += "\n\tGrid shape: {0}".format(self.grid_shape)
            s += "\n\tInitial grid provided: {0}".format(self.initial_grid_provided)

        return s

    def _check_parameters(self):
        """Check parameters for consistency. Defines any missing parameters which can be calculated from others."""
        try:
            if self.mutation_generator is None:
                if self.mutation_rates is not None and self.mutation_rates != 0:
                    self.warnings.append('Using the default mutation generator: {0}'.format(
                        self.default_mutation_generator.__str__()))
                self.mutation_generator = self.default_mutation_generator
            self._check_algorithm()
            self._check_populations()
            self._check_timing()
            self._check_samples()
            self._get_sample_times()  # Must be called after the timing and samples have been checked

            if self.mutation_rates is None:
                self.warnings.append('Using the default mutation rate: {0}'.format(self.default_mutation_rate))
                self.mutation_rates = self.default_mutation_rate
            self._check_mutation_rates()

            self._check_labels()

            self._check_treatment()

            if self.figsize is None:
                self.figsize = self.default_figsize

            if self.progress is None:
                self.progress = False

            if self.colourscales is None:
                self.colourscales = self.default_colourscales

            self._check_diff_cell_parameters()

        except ParameterException as e:
            print('Error with parameters, unable to run simulation.\n')
            print(e)
            return False

        if self.print_warnings:
            print('============== Setting up ==============')
            for w in self.warnings:
                print(w)
            print('========================================')
        return True

    def _check_algorithm(self):
        """Checks that the algorithm asked for is one of the options"""
        if self.algorithm not in self.algorithm_options:
            raise ParameterException('Algorithm {0} is not valid. Pick from {1}'.format(self.algorithm, self.algorithm_options))
        if self.algorithm in ['Branching', 'Moran', 'WF']:
            self.non_zero_calc = True
        elif self.algorithm in ['Moran2D', 'WF2D']:
            self.non_zero_calc = False

    def _check_populations(self):
        """Checks that only one population parameter has been given"""
        num_defined = sum([self.initial_cells is not None, self.initial_size_array is not None,
                           self.grid_shape is not None, self.initial_grid is not None])
        if num_defined != 1:
            raise ParameterException('Must provide exactly one of:\n\tinitial_cells\n\tinitial_size_array\n\t'
                                     'grid_shape (Moran2D/WF2D only)\n\tinitial_grid (Moran2D/WF2D only)')
        if self.algorithm not in ['Moran2D', 'WF2D']:
            if self.initial_cells is None and self.initial_size_array is None:
                raise ParameterException('Must provide initial_cells or initial_size_array')
            self._setup_initial_population_non_spatial()
        else:
            self._setup_2D_initial_population()

        self._define_remaining_initial_arrays()
        self._convert_lists_to_arrays()

    def _convert_lists_to_arrays(self):
        # Some cases need to be numpy arrays not lists
        self.initial_size_array = np.array(self.initial_size_array)

    def _setup_initial_population_non_spatial(self):
        if self.initial_cells is not None:
            self.initial_size_array = [self.initial_cells]
        elif self.initial_size_array is not None:
            self.initial_cells = sum(self.initial_size_array)

    def _setup_2D_initial_population(self):
        if self.initial_cells is not None:
            self._try_making_square_grid()
            self.initial_size_array = [self.initial_cells]
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_size_array is not None:
            if len(self.initial_size_array) == 1:
                self.initial_cells = sum(self.initial_size_array)
                self._try_making_square_grid()
                self.initial_grid = np.zeros(self.grid_shape, dtype=int)
            else:
                raise ParameterException('Cannot use initial_size_array with 2D simulation. Provide initial_grid instead.')
        elif self.grid_shape is not None:
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.initial_size_array = [self.initial_cells]
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_grid is not None:
            self.grid_shape = self.initial_grid.shape
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.initial_grid = self.initial_grid.astype(int)
            self._create_initial_size_array_from_grid()
            self.initial_grid_provided = True
        else:
            raise ParameterException('Please provide one of the population size inputs')

        self._check_other_2D_parameters()

    def _try_making_square_grid(self):
        poss_grid_size = int(np.sqrt(self.initial_cells))
        if poss_grid_size ** 2 == self.initial_cells:
            self.grid_shape = (poss_grid_size, poss_grid_size)
            self.warnings.append('Using a grid of {0}x{0}'.format(self.grid_shape[0], self.grid_shape[1]))
        else:
            raise ParameterException('Square grid not compatible with {0} cells. To run a rectangular grid provide a grid shape'.format(
                self.initial_cells))

    def _define_remaining_initial_arrays(self):
        # Define the initial arrays if they have not been defined yet.
        if self.label_array is None:
            self.label_array = [self.default_label for i in range(len(self.initial_size_array))]

        elif len(self.label_array) != len(self.initial_size_array):
            raise ParameterException('Inconsistent initial_size_array and label_array. Ensure same length.')

        if self.gene_label_array is None:
            self.gene_label_array = np.array([self.default_mutation_type for i in range(len(self.initial_size_array))])
        elif len(self.gene_label_array) != len(self.initial_size_array):
            raise ParameterException('Inconsistent initial_size_array and additional_label_array. Ensure same length.')
        else:
            self.gene_label_array = np.array(self.gene_label_array)

        self._check_fitness_array()

    def _check_fitness_array(self):

        if self.mutation_generator.multi_gene_array:
            # Make sure the fitness array has the appropriate dimensions.
            # All non-mutated genes have np.nan
            # First column is the wild type/non-gene associated fitness. Can still vary if specified.
            if self.fitness_array is None:  # Assume all genes have default fitness.
                self.fitness_array = np.full((len(self.initial_size_array), len(self.mutation_generator.genes)+1),
                                             np.nan, dtype=float)
                self.fitness_array[:, 0] = self.default_fitness
            else:
                self.fitness_array = np.array(self.fitness_array)
                if self.fitness_array.shape == (len(self.initial_size_array),):
                    # One dimensional array.
                    # One gene mutated for each initial clone (could be the wild type fitness that is given)

                    # Make blank array
                    blank_fitness_array = np.full((len(self.initial_size_array), len(self.mutation_generator.genes)+1),
                                                  np.nan, dtype=float)
                    blank_fitness_array[:, 0] = self.default_fitness
                    blank_fitness_array[
                        np.arange(len(self.initial_size_array)), self.gene_label_array + 1] = self.fitness_array

                    self.fitness_array = blank_fitness_array
                elif self.fitness_array.shape == (len(self.initial_size_array), len(self.mutation_generator.genes)+1):
                    pass  # Fully specified fitness array for each initial clone.
                else:
                    raise ParameterException("Incorrect shape of fitness_array. \
                    Ensure either 1D or has a wild type column plus column for each gene and \
                    that the length/num rows equals the number of initial clones.")
        else:  # Just want 1D fitness array
            if self.fitness_array is None:
                self.fitness_array = np.full(len(self.initial_size_array), self.default_fitness, dtype=float)
            elif len(self.fitness_array) != len(self.initial_size_array):
                raise ParameterException('Inconsistent initial_size_array and fitness_array. Ensure same length.' +
                                         '\nlen(initial_size_array): {}, len(fitness_array): {}'.format(
                                             len(self.initial_size_array), len(self.fitness_array)))

    def _check_other_2D_parameters(self):
        # Check that the hexagonal grid has only even dimensions.
        if self.grid_shape[0] % 2 != 0 or self.grid_shape[1] % 2 != 0:
            raise ParameterException('Must have even number of rows/columns in the hexagonal grid.')

        # if self.direction_bias is not None:
        #     raise ParameterException('Division direction bias not currently implemented')
        #     if len(self.direction_bias) != 6:
        #         raise ParameterException('position_weighting must be an array of length 6')
        #     if isinstance(self.direction_bias, list):
        #         raise ParameterException('position_weighting must be a numpy array, not a list')

    def _check_timing(self):
        """Checks that times, max_time, division_rate and simulation_steps are consistent and defines any missing values"""
        if self.times is not None:
            if not isinstance(self.times, (list, np.ndarray)):
                raise ParameterException("Times must be list or numpy array")
            if isinstance(self.times, list):
                self.times = np.array(self.times)
            diff = np.diff(self.times)
            if np.any(diff <= 0):
                raise ParameterException("Times must be in increasing order")
            if self.max_time is not None:
                if self.max_time != self.times[-1]:
                    raise ParameterException(
                        "Max time does not match times given. Do not need to provide max_time if times provided.")
            else:
                self.max_time = self.times[-1]

        if self.simulation_steps is not None:
            if self.algorithm in ['Branching']:
                raise ParameterException(
                    'Cannot specify number of simulations steps for the branching process algorithm.\n'
                    'Please provide a max_time and division_rate instead')
            if self.max_time is not None:
                if self.division_rate is not None:
                    # All defined, check they are consistent
                    sim_steps = self._get_simulation_steps()
                    if sim_steps != self.simulation_steps:  # Raise error if not consistent
                        st = 'Simulation_steps does not match max_time and division_rate.\n' \
                             'Provide only 2 of the three or ensure all are consistent.\n' \
                             'simulation_steps={0}, steps calculated from time and division rate={1}'.format(
                            self.simulation_steps, sim_steps
                        )
                        raise ParameterException(st)

                else:
                    # simulation_steps and max_time given. Calculate division rate
                    self.division_rate = self._get_division_rate()
                    self.warnings.append('Division rate for the simulation is {0}'.format(self.division_rate))
            else:
                if self.division_rate is None:
                    # Simulation steps defined but not max_time and division_rate
                    # Use the default division rate
                    self._use_default_division_rate()
                # simulation_steps and division_rate given, calculate max_time
                self.max_time = self._get_max_time()
                self.warnings.append('Max time for the simulation is {0}'.format(self.max_time))
        else:  # No simulation steps or time points defined. Calculate from max_time and division rate if given
            if self.division_rate is None:
                self._use_default_division_rate()
            if self.max_time is None:
                self._use_default_max_time()
            self.simulation_steps = self._get_simulation_steps()
            self.warnings.append('{0} simulation_steps'.format(self.simulation_steps))

    def _check_samples(self):
        if self.times is None:
            if self.samples is None:
                self.samples = self.default_samples
        else:
            self.samples = len(self.times)

        if self.simulation_steps is not None:
            if self.samples > self.simulation_steps:
                self.samples = self.simulation_steps
                if self.times is not None:
                    self.warnings.append('Fewer simulation steps than number of time points requested!')

    def _get_sample_times(self):
        if self.times is None:
            # The time points at each sample.
            self.times = np.linspace(0, self.max_time, self.samples + 1)

        if self.algorithm not in ['Branching']:
            steps_per_unit_time = self.simulation_steps / self.max_time
            # Which points to take a sample
            sample_points_float = self.times * steps_per_unit_time
            self.sample_points = np.around(sample_points_float).astype(int)
            # Can sometimes have duplicates for close time points and slow division rate
            self.sample_points = np.unique(self.sample_points)
            if self.algorithm in self.moran_algorithms:
                rounded_times = self.sample_points / self.division_rate / self.initial_cells
            else:
                rounded_times = self.sample_points / self.division_rate

            times_changed = False
            if len(self.times) != len(rounded_times):
                times_changed = True
            elif not np.allclose(self.times, rounded_times) > 0:
                times_changed = True
            if times_changed:
                self.warnings.append('Times rounded to match simulation steps:')
                self.warnings.append('\tTimes used: {}'.format(rounded_times))
            self.times = rounded_times
        else:
            self.sample_points = None

    def _check_mutation_rates(self):
        """If only a single mutation rate is give, convert to the numpy array to match the required format"""
        if isinstance(self.mutation_rates, (int, float)):
            self.mutation_rates = np.array([[0, self.mutation_rates]])
        self.mutation_rates = self.mutation_rates.copy()  # In case same mutation rates used in multiple simulations.
        if self.mutation_rates[0][0] != 0:
            # Adding a section of zero mutation rate at the start
            self.mutation_rates = np.concatenate([np.array([[0, 0]]), self.mutation_rates])

        if np.any(self.mutation_rates[:, 0] > self.max_time):
            raise ParameterException('Mutation rates change at point beyond end of simulation.')

    def _create_initial_size_array_from_grid(self):
        """For the 2D simulations, if an initial grid of clone positions is provided, fill in the initial_size_array"""

        # If the initial size array is not given, define it here.
        idx_counts = {k:v for k,v in zip(*np.unique(self.initial_grid, return_counts=True))}
        self.initial_size_array = []
        for i in range(int(max(idx_counts))+1):
            if i in idx_counts:
                self.initial_size_array.append(idx_counts[i])
            else:
                self.initial_size_array.append(0)

    def _check_labels(self):
        if self.label_times is not None:
            if self.label_frequencies is None or self.label_values is None:
                raise ParameterException('Label frequencies and label values must also be defined to apply labels.')
            if isinstance(self.label_times, (int, float)):
                self.label_times = np.array([self.label_times])
            elif isinstance(self.label_times, list):
                self.label_times = np.array(self.label_times)
            elif not isinstance(self.label_times, np.ndarray):
                raise ParameterException('Do not recognise the type of the label times.')

            if isinstance(self.label_frequencies, (int, float)):
                self.label_frequencies = [self.label_frequencies]
            if isinstance(self.label_values, (int, float)):
                self.label_values = [self.label_values]
            if self.label_fitness is None:
                self.label_fitness = [None]*len(self.label_times)
            elif isinstance(self.label_fitness, (int, float)):
                self.label_fitness = [self.label_fitness]

            if self.label_genes is None:
                self.label_genes = [None]*len(self.label_times)
            else:
                if isinstance(self.label_genes, int):
                    self.label_genes = [self.label_genes]
                if any(g > -1 for g in self.label_genes):  # Means applying a mutant to a particular gene
                    # Requires multi-gene setup
                    if not self.mutation_generator.multi_gene_array:
                        raise ParameterException('Applying labels with mutations to particular genes requires a '
                                                 'mutation generator with multi_gene_array=True')

            if len(self.label_times) != len(self.label_frequencies) or len(self.label_times) != len(self.label_values) \
                    or len(self.label_times) != len(self.label_fitness):
                raise ParameterException('Length of label times, frequencies and values is not consistent.')

    def _check_treatment(self):
        if self.treatment_timings is None and self.treatment_effects is None:
            self.treatment_effects = 1
            self.treatment_timings = None
            self.treatment_replace_fitness = False
        else:
            if not isinstance(self.mutation_generator.mutation_combination_class, UnboundedFitness):
                self.warnings.append('Treatment effects not tested with any diminishing returns')
                if self.mutation_generator.multi_gene_array:
                    self.warnings.append('Treatment effects will be applied prior to the diminishing returns')
                else:
                    self.warnings.append('Treatment effects will be applied after the diminishing returns')
            if not isinstance(self.treatment_timings, (list, np.ndarray)) or not isinstance(self.treatment_effects,
                                                                                            (list, np.ndarray)):
                raise ParameterException('treatment_timings and treatment_effects must be lists or arrays')
            elif len(self.treatment_timings) != len(self.treatment_effects):  # One time for each treatment
                raise ParameterException('treatment_timings and treatment_effects must be same length')
            elif self.mutation_generator.multi_gene_array:
                # Fitness changes apply to each gene
                self.warnings.append('Treatment effects are per gene')
                # Always needs to be a treatment so it can be applied to new mutations.
                # Insert a neutral treatment at start if initial treatment is not defined.
                if self.treatment_timings[0] != 0:
                    if self.treatment_replace_fitness:
                        self.treatment_effects = [np.full(len(self.mutation_generator.genes) + 1, np.nan)] + list(
                            self.treatment_effects)
                    else:
                        self.treatment_effects = [np.ones(len(self.mutation_generator.genes)+1)] + list(self.treatment_effects)
                    self.treatment_timings = [0] + self.treatment_timings

                if any([len(t) != len(self.mutation_generator.genes)+1 for t in self.treatment_effects]):
                    raise ParameterException(
                        'Each treatment effect must have the same length as the number of genes plus 1.')
            else:
                # Fitness changes apply to each clone
                if not np.array_equal(self.mutation_rates, np.array([[0, 0]])):
                    raise ParameterException(
                        'Cannot have treatment changes and introduce new mutations unless using a multi-gene array')

                self.warnings.append('Treatment effects are per clone.')
                # Always needs to be a treatment so it can be applied to new mutations.
                # Insert a neutral treatment at start if initial treatment is not defined.
                if self.treatment_timings[0] != 0:
                    if self.treatment_replace_fitness:
                        self.treatment_effects = [np.full(len(self.initial_size_array), np.nan)] + list(
                            self.treatment_effects)
                    else:
                        self.treatment_effects = [np.ones(len(self.initial_size_array))] + list(
                            self.treatment_effects)
                    self.treatment_timings = [0] + self.treatment_timings

                if any([len(t) != len(self.initial_size_array) for t in self.treatment_effects]):
                    raise ParameterException(
                        'Each treatment effect must have the same length as the number of initial mutations.')

            # Convert the treatment effects to np.array
            self.treatment_effects = np.array(self.treatment_effects)

    def _get_simulation_steps(self):
        if self.algorithm in ['Moran', 'Moran2D']:
            sim_steps = round(self.max_time * self.division_rate * self.initial_cells)
        elif self.algorithm in ['WF', 'WFBal', 'WF2D']:
            sim_steps = round(self.max_time * self.division_rate)
        elif self.algorithm in ['Branching']:
            sim_steps = None
        else:
            raise ParameterException('Calculation of sim_steps for {0} not implemented.'.format(self.algorithm))
        if sim_steps == 0:
            raise ParameterException('Zero simulation steps for the given algorithm, max_time and division rate')
        return sim_steps

    def _get_division_rate(self):
        if self.algorithm in self.moran_algorithms:
            div_rate = self.simulation_steps/self.max_time/self.initial_cells
        elif self.algorithm in self.wf_algorithms:
            div_rate = self.simulation_steps / self.max_time
        else:
            raise ParameterException('Calculation of division rate for {0} algorithm not implemented. \
            Please provide division_rate as an argument.'.format(self.algorithm))
        return div_rate

    def _get_max_time(self):
        if self.algorithm in self.moran_algorithms:
            max_time = self.simulation_steps/self.division_rate/self.initial_cells
        elif self.algorithm in self.wf_algorithms:
            max_time = self.simulation_steps / self.division_rate
        else:
            raise ParameterException('Calculation of max_time for {0} algorithm not implemented. \
                        Please provide division_rate as an argument.'.format(self.algorithm))
        return max_time

    def _use_default_division_rate(self):
        self.warnings.append('Using the default division rate: {0}'.format(self.default_division_rate))
        self.division_rate = self.default_division_rate

    def _use_default_max_time(self):
        self.warnings.append('Using the default max time: {0}'.format(self.default_max_time))
        self.max_time = self.default_max_time

    def _check_diff_cell_parameters(self):
        if self.r is not None or self.gamma is not None:
            if self.algorithm not in {'Moran', 'Moran2D', 'Branching'}:
                raise ParameterException(
                    'Cannot run {} algorithm with B cells. Change algorithm or do not supply r or gamma arguments'.format(self.algorithm))
            if self.r is None:
                raise ParameterException(
                    'Must provide both r and gamma to run with B cells. Please provide r')
            if self.gamma is None:
                raise ParameterException(
                    'Must provide both r and gamma to run with B cells. Please provide gamma')

            if self.r > 0.5 or self.r < 0:
                raise ParameterException('Must have 0<=r<=0.5')
            if self.gamma <= 0:
                raise ParameterException('gamma must be > 0')

            if self.stratification_sim_percentile is None:
                self.stratification_sim_percentile = self.default_stratification_sim_percentile
                self.warnings.append('Using the default stratification_sim_percentile: {0}'.format(self.stratification_sim_percentile))

            self._get_diff_cell_simulation_times()

    def _get_diff_cell_simulation_times(self):
        """
        Need to simulate for a period prior to the sample time.
        Use a specified percentile of the exponential distribution of stratification times
        to find the minimum time to simulate.
        For the Moran models, find the last simulation step prior to this minimum time before the sample point.
        For the Branching process, the times can be compared to the time of the next progenitor division
        """
        if self.stratification_sim_percentile < 1:
            min_diff_sim_time = expon.ppf(self.stratification_sim_percentile, scale=1 / self.gamma)
            if self.algorithm == 'Branching':
                diff_sim_starts = self.times - min_diff_sim_time
                diff_sim_starts[diff_sim_starts < 0] = 0
                self.diff_cell_sim_switches = self._merge_time_intervals(diff_sim_starts, self.times) + [np.inf]
            else:
                steps_per_unit_time = self.simulation_steps / self.max_time
                sim_steps_for_diff_sims = min_diff_sim_time * steps_per_unit_time
                sim_steps_to_start_diff = (self.sample_points - sim_steps_for_diff_sims).astype(int)
                sim_steps_to_start_diff[sim_steps_to_start_diff < 0] = 0
                self.diff_cell_sim_switches = self._merge_time_intervals(sim_steps_to_start_diff, self.sample_points)
        else:
            self.diff_cell_sim_switches = [0, np.inf]

    def _merge_time_intervals(self, starts, ends):
        merged_starts = [starts[0]]
        merged_ends = []
        gaps = starts[1:] - ends[:-1]
        for i, g in enumerate(gaps):
            if g <= 0:
                # No gap. Merged with next section
                pass
            else:
                merged_ends.append(ends[i])
                merged_starts.append(starts[i + 1])
        merged_ends.append(ends[-1])
        all_events = sorted(merged_starts + merged_ends)
        return all_events

    def _select_simulator_class(self):
        if self.r is None:
            if self.algorithm == 'WF':
                if self.end_condition_function is not None:
                    self.sim_class = WFStop
                else:
                    self.sim_class = WrightFisherSim
            elif self.algorithm == 'Moran':
                if self.end_condition_function is not None:
                    self.sim_class = MoranStop
                else:
                    self.sim_class = MoranSim
            elif self.algorithm == 'Moran2D':
                if self.end_condition_function is not None:
                    self.sim_class = Moran2DStop
                else:
                    self.sim_class = Moran2D
            elif self.algorithm == 'Branching':
                if self.end_condition_function is not None:
                    raise ParameterException('Cannot use an end_condition_function for the branching algorithm')
                self.sim_class = SimpleBranchingProcess
            elif self.algorithm == 'WF2D':
                if self.end_condition_function is not None:
                    self.sim_class = WF2DStop
                else:
                    self.sim_class = WrightFisher2D
        else:  # Simulations including B cells.
            if self.end_condition_function is not None:
                raise ParameterException(
                    'Cannot use an end_condition_function for the simulations including differentiated cells')
            if self.algorithm == 'Moran':
                self.sim_class = MoranWithDiffCells
            elif self.algorithm == 'Moran2D':
                self.sim_class = Moran2DWithDiffcells
            elif self.algorithm == 'Branching':
                self.sim_class = BranchingWithDiffCells

    def get_simulator(self):
        if self.end_condition_function is not None:
            return self.sim_class(self, self.end_condition_function)
        else:
            return self.sim_class(self)

