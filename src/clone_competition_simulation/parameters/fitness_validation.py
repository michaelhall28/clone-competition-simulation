from mimetypes import init
from typing import Annotated, Literal
import numpy as np
from pydantic import (
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import (
    assign_config_settings,
    ValidationBase,
    ParameterBase,
    FloatOrArrayParameter,
    AlwaysValidateNoneField
)
from .times_validation import TimeValidator
from .population_validation import PopulationValidator
from ..fitness import FitnessCalculator


class FitnessParameters(ParameterBase):
    """Parameters that control fitness and mutation in the simulation.

    Fields:
        fitness_calculator:
            A FitnessCalculator object that defines how mutations affect fitness.
            This is required if mutations are to occur during the simulation or if 
            multiple fitness values need to be combined (e.g. if two labels with 
            fitness effects are applied). The generator specifies fitness distributions
            for different genes and defines how fitness values are combined.

            Example:
                fitness_calculator = FitnessCalculator(...)  # Complex object, see docs

        mutation_rates:
            The rate at which mutations occur, specified as a float or array.
            If a single float is provided, mutations occur at that constant rate
            throughout the simulation. For time-dependent rates, provide a 2D array
            where each row is [time, rate], indicating the mutation rate starting
            at that time.

            Example:
                mutation_rates = 0.01  # Constant mutation rate of 0.01
                mutation_rates = [[0, 0.0], [5, 0.01], [10, 0.02]]  # Rates change over time

        initial_fitness_array:
            Fitness values for the initial clones in the population. Fitness is
            relative, with 1.0 typically representing neutral fitness. If a single
            float is provided, all initial clones have that fitness. If an array
            is provided, it must match the length of the initial population. 
            If not provided, all clones are assumed to have neutral fitness (1.0).

            Example:
                initial_fitness_array = 1.0  # All clones neutral fitness
                initial_fitness_array = [1.0, 0.8, 1.2]  # Different fitness for each clone

        initial_mutant_gene_array:
            Specifies which genes are mutated in the initial clones. Use ``None`` for
            wild-type (no mutation). If an str is provided, all clones have
            mutations in the gene with that name. If a list is provided, it must match the
            length of the initial population, with each element indicating the
            mutated gene name.

            Example:
                initial_mutant_gene_array = None  # All clones wild-type
                initial_mutant_gene_array = "gene1"  # All clones mutated in gene "gene1"
                initial_mutant_gene_array = [None, "gene1", "gene2"]  # Mixed mutations

    Examples:
        No mutations, neutral fitness (same as not specifying fitness parameters):
            fitness_calculator = None
            mutation_rates = 0
            initial_fitness_array = 1.0
            initial_mutant_gene_array = None

        Constant mutation rate with initial mutants:
            fitness_calculator = some_fitness_calculator
            mutation_rates = 0.005
            initial_fitness_array = [0.9, 1.0, 1.1]
            initial_mutant_gene_array = [None, "gene1", "gene2"]

        Time-varying mutation rates:
            fitness_calculator = some_fitness_calculator
            mutation_rates = [[0, 0.0], [20, 0.01]]
            initial_fitness_array = 1.0
            initial_mutant_gene_array = None
    """
    _field_name = "fitness"
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fitness_calculator: FitnessCalculator | None = AlwaysValidateNoneField
    mutation_rates: FloatOrArrayParameter = AlwaysValidateNoneField
    initial_fitness_array: FloatOrArrayParameter = AlwaysValidateNoneField
    initial_mutant_gene_array: str | list | None | np.ndarray[tuple[int], np.int_] = AlwaysValidateNoneField


class FitnessValidator(FitnessParameters, ValidationBase):
    _wt_fitness = 1.0
    _default_mutation_type = np.nan
    tag: Literal['Full']
    times: TimeValidator
    population: PopulationValidator
    config_file_settings: FitnessParameters | None = None

    def _validate_model(self):
        initial_size_array = self.population.initial_size_array
        self.initial_fitness_array = self.get_value_from_config("initial_fitness_array")
        if self.initial_fitness_array is None:
            self.initial_fitness_array = self._wt_fitness  # Assume all with neutral fitness if not otherwise specified

        if self.fitness_calculator is None:
            self.fitness_calculator = self.config_file_settings.fitness_calculator

        self._check_initial_mutant_gene_array()

        if self.fitness_calculator is not None and self.fitness_calculator.multi_gene_array:
            # Make sure the initial fitness array has the appropriate dimensions.
            # There will be one row per initial clone
            # and one column for the wild type fitness plus one column for each gene in the fitness calculator.
            # All non-mutated genes in a clone have np.nan in that row
            # First column is the wild type/non-gene associated fitness. Can still vary if specified.
            if isinstance(self.initial_fitness_array, float):
                # Assume all genes have the same fitness
                fitness_array = np.full((len(initial_size_array), len(self.fitness_calculator.genes)+1),
                                             np.nan, dtype=float)
                fitness_array[:, 0] = self.initial_fitness_array
                self.initial_fitness_array = fitness_array
            else:
                if self.initial_fitness_array.shape == (len(initial_size_array),):
                    # One dimensional array.
                    # One gene mutated for each initial clone (could be the wild type fitness that is given)

                    # Make blank array
                    blank_fitness_array = np.full((len(initial_size_array), len(self.fitness_calculator.genes)+1),
                                                  np.nan, dtype=float)
                    if self.initial_mutant_gene_array is None:
                        # No mutant genes specified, so all fitness values in the first column (wild type fitness)
                        blank_fitness_array[:, 0] = self.initial_fitness_array
                    else:
                        # The mutant genes are specified by the initial_mutant_gene_array. 
                        # The columns in the fitness array are shifted by one compared to the gene numbers because
                        # of the wild type column at the start, so add 1 to the mutant gene numbers to get the right 
                        # column in the fitness array.
                        
                        # Add the wt value to the first column
                        blank_fitness_array[:, 0] = self._wt_fitness
                        # Add the initial fitness values to the appropriate columns based on the mutant gene array
                        blank_fitness_array[
                            np.arange(len(initial_size_array)), self.initial_mutant_gene_array + 1] = self.initial_fitness_array

                    self.initial_fitness_array = blank_fitness_array
                elif self.initial_fitness_array.shape == (len(initial_size_array), len(self.fitness_calculator.genes)+1):
                    pass  # Fully specified fitness array for each initial clone.
                else:
                    raise ValueError("Incorrect shape of fitness_array. \
                    Ensure either 1D or has a wild type column plus column for each gene and \
                    that the length/num rows equals the number of initial clones.")
        else:  # Just want 1D fitness array
            if isinstance(self.initial_fitness_array, float):
                self.initial_fitness_array = np.full(len(initial_size_array), self.initial_fitness_array, dtype=float)
            elif len(self.initial_fitness_array) != len(initial_size_array):
                raise ValueError('Inconsistent initial_size_array and fitness_array. Ensure same length.' +
                                      '\nlen(initial_size_array): {}, len(fitness_array): {}'.format(
                                          len(initial_size_array), len(self.initial_fitness_array)))

        self._check_mutation_rates()


    def _check_initial_mutant_gene_array(self):
        """Check that the initial mutant gene array is compatible with the fitness calculator and the initial size array. 
        Convert the gene names to indices. 
        """
        if self.initial_mutant_gene_array is None:
            return  # No initial mutant genes, so nothing to check or convert
        
        if self.fitness_calculator is None:
            raise ValueError('Cannot specify initial mutant gene array without a fitness calculator.')
        
        if isinstance(self.initial_mutant_gene_array, str):
            # Convert the gene name to the gene index, then make an array with one value per initial clone. 
            gene_number = self.fitness_calculator.get_gene_number(self.initial_mutant_gene_array)
            self.initial_mutant_gene_array = np.full_like(self.population.initial_size_array, gene_number)
        elif len(self.initial_mutant_gene_array) != len(self.population.initial_size_array): 
            raise ValueError('Inconsistent initial_size_array and initial_mutant_gene_array. Ensure same length.')
        else:
            # Convert gene names to indices
            self.initial_mutant_gene_array = np.array([
                self.fitness_calculator.get_gene_number(gene) if gene is not None else self._default_mutation_type
                for gene in self.initial_mutant_gene_array
            ])

    def _check_mutation_rates(self):
        """If only a single mutation rate is give, convert to the numpy array to match the required format"""
        if self.mutation_rates is None:
            self.mutation_rates = self.get_value_from_config("mutation_rates")
            if self.mutation_rates is None:
                self.mutation_rates = 0  # If not defined, set the mutation rate to zero
        if isinstance(self.mutation_rates, (int, float)):
            self.mutation_rates = np.array([[0, self.mutation_rates]])
        self.mutation_rates = self.mutation_rates.copy()  # In case same mutation rates used in multiple simulations.
        if self.mutation_rates[0][0] != 0:
            # Mutation rates do not start from time zero.
            # Adding a section of zero mutation rate at the start
            self.mutation_rates = np.concatenate([np.array([[0, 0]]), self.mutation_rates])

        if np.any(self.mutation_rates[:, 0] > self.times.max_time):
            raise ValueError('Mutation rates change at point beyond end of simulation.')

        if np.any(self.mutation_rates[:, 1] > 0) and self.fitness_calculator is None:
            raise ValueError('Cannot generate mutations without a mutation generator set.')


fitness_validation_type = Annotated[
    (Annotated[FitnessParameters, Tag("Base")] | Annotated[FitnessValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
