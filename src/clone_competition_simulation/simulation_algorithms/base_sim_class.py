"""
This is a super class for all of the simulations.
It contains the common function to setup, run and plot results from simulations.

The subclasses have to define the sim_step and any other functions required for the specific algorithm
"""
import sys
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

import gzip
import dill as pickle
import numpy as np
from loguru import logger
from scipy.sparse import SparseEfficiencyWarning

from .current_data import CurrentData, NonSpatialCurrentData
from .exceptions import EndConditionError
from .mutation_manager import MutationManagerMixin
from .simulation_analysis import SimulationAnalysisMixin
from .simulation_loop import SimulationLoopMixin
from .simulation_setup import SimulationSetupMixin
from .simulation_plotting import SimulationPlottingMixin

if TYPE_CHECKING:
    from ..parameters.parameter_validation import Parameters

warnings.simplefilter('ignore', SparseEfficiencyWarning)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", enqueue=True)


class BaseSimClass(ABC, SimulationSetupMixin, SimulationLoopMixin,
                   MutationManagerMixin, SimulationAnalysisMixin, SimulationPlottingMixin):
    """
    Base class for all simulations. Contains the common attributes and functions for setting up, running and plotting simulations.
    The various mixins contain the different functions for these steps, which are shared across different algorithms.

    Subclasses for each algorithm overwrite the sim_step function and any other functions required for the specific algorithm. 
    """
    current_data_cls: ClassVar[type[CurrentData]] = CurrentData

    # Indices of the columns in the clones array
    id_idx = 0  # Unique integer id for each clone. Int.
    label_idx = 1  # The type of the clone. Inherited label does not change. Int. Represents GFP or similar.
    fitness_idx = 2  # The fitness of the clone. Float.
    generation_born_idx = 3  # The sample the clone first appeared in.  Int.
    parent_idx = 4  # The id of the clone that this clone emerged from.  Int.
    gene_mutated_idx = 5  # The gene (index) the mutation appears in (or similar info). Int.
    # Could encode gene and/or nonsense/missense/... depending on how genes are defined

    def __init__(self, parameters: "Parameters"):
        # Get attributes from the parameters
        self.total_pop = parameters.population.initial_cells
        self.initial_size_array = parameters.population.initial_size_array
        self.mutation_rates = parameters.fitness.mutation_rates
        self.fitness_calculator = parameters.fitness.fitness_calculator
        self.division_rate = parameters.times.division_rate
        self.max_time = parameters.times.max_time
        self.times = parameters.times.times
        self.sample_points = parameters.times.sample_points
        self.non_zero_calc = parameters.non_zero_calc
        self.label_times = parameters.labels.label_times
        self.label_frequencies = parameters.labels.label_frequencies
        self.label_values = parameters.labels.label_values
        self.label_fitness = parameters.labels.label_fitness
        self.label_genes = parameters.labels.label_genes
        self.parameters = parameters

        # Define new attributes required for the simulation
        self.initial_clones = len(self.initial_size_array)
        self.current_fitness_multiplier = None  # The effect of the current treatment
        self.sim_length = len(self.times)
        self._search_times = None
        self.s_muts = set()  # Synonymous mutations. Indices of the first clone they appear in
        self.ns_muts = set()  # Non-synonymous mutations. Indices of the first clone they appear in
        self.label_muts = set()  # Labelled clones. Indices of the clones that get a label (after init)
        self.label_count = 0
        self.next_label_time = None
        self.treatment_count = -1
        self.next_treatment_time = 0
        self.treatment_replace_fitness = None
        self.treatment_timings = None
        self.treatment_effects = None
        self.plot_idx = 0  # Keeping track of x-coordinate of the plot
        self.tree = None
        self.new_mutation_count = None
        self.mutations_to_add = None

        # Details for plotting
        self.figsize = parameters.plotting.figsize
        self.descendant_counts = {}
        self.plot_colour_maps = parameters.plotting.plot_colour_maps
        self.i = 0
        self.colours = None

        # Stores the sizes of clones containing particular mutants.
        self.mutant_clone_array = None
        self.trimmed_tree = None  # Used for mutant clone arrays. Clone tree with only observed clones and their ancestors. 
        self.sampled_clones = None  # The clones which are observed at sample points.  

        # A few attributes to help with the simulation running and storage
        self.tmp_store = parameters.tmp_store
        self.store_rotation = 0  # Alternates between two tmp stores (0, 1) in case error occurs during pickle dump.
        self.is_lil = True  # Is the population array stored in scipy.sparse.lil_matrix (True) or numpy array (False)
        self.finished = False
        self.random_state = None  # For storing the state of the random sequence for continuing

        # Setup the various initial arrays
        self._calculate_search_times()
        self._setup_label_times()
        self._setup_treatment(parameters)
        self._setup_clone_tree()
        self.new_mutation_count, self.mutations_to_add = self._precalculate_mutations()  # Faster to pre-calculate mutations for long, mutation heavy simulations
        self.total_clone_count = self.initial_clones + self.new_mutation_count
        self.next_mutation_index = self.initial_clones  # Keeping track of how many mutations added

        self._init_arrays(parameters.labels.initial_label_array, parameters.fitness.initial_mutant_gene_array, parameters.fitness.initial_fitness_array)
        
        # Attributes for early stopping
        self.stop_time: float | None = None
        self.stop_condition_result: Any | None = None   # Spare attribute to place any relevant result from the stopping
        self.stop_function = parameters.end_condition_function


def pickle_load(filename: str, change_sparse_to_csr: bool=True) \
        -> BaseSimClass:
    """Load a simulation from a gzipped pickle

    Parameters
    ----------
    filename : str
        Name of the pickle file containing the simulation
    change_sparse_to_csr : bool, optional
        Convert population array to CSR format after loading. 
        By default True

    Returns
    -------
    BaseSimClass
        A simulation object
    """
    with gzip.open(filename, 'rb') as f:
        sim = pickle.load(f)

    if change_sparse_to_csr:
        sim.change_sparse_to_csr()

    return sim
