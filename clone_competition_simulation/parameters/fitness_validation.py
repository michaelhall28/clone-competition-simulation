from typing import Annotated, Literal
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import (
    assign_config_settings,
    ValidationBase,
    ParameterBase,
    FloatOrArrayParameter,
    IntOrArrayParameter,
    AlwaysValidateNoneField
)
from .times_validation import TimeValidator
from .population_validation import PopulationValidator
from clone_competition_simulation.fitness import MutationGenerator


class FitnessParameters(ParameterBase):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mutation_generator: MutationGenerator | None = AlwaysValidateNoneField
    mutation_rates: FloatOrArrayParameter = AlwaysValidateNoneField
    fitness_array: FloatOrArrayParameter = AlwaysValidateNoneField
    initial_mutant_gene_array: IntOrArrayParameter = AlwaysValidateNoneField


class FitnessValidator(FitnessParameters, ValidationBase):
    _wt_fitness = 1.0
    _default_mutation_type = -1
    tag: Literal['Full']
    times: TimeValidator
    population: PopulationValidator
    config_file_settings: FitnessParameters | None = None

    def _validate_model(self):
        initial_size_array = self.population.initial_size_array
        if self.fitness_array is None:
            self.fitness_array = self.config_file_settings.fitness_array
            if self.fitness_array is None:
                self.fitness_array = self._wt_fitness  # Assume all with neutral fitness if not otherwise specified

        if self.mutation_generator is None:
            self.mutation_generator = self.config_file_settings.mutation_generator

        self.initial_mutant_gene_array = self.get_value_from_config("initial_mutant_gene_array")
        if self.initial_mutant_gene_array is None:
            self.initial_mutant_gene_array = np.full_like(initial_size_array, self._default_mutation_type)
        elif isinstance(self.initial_mutant_gene_array, int):
            self.initial_mutant_gene_array = np.full_like(initial_size_array, self.initial_mutant_gene_array)
        elif len(self.initial_mutant_gene_array) != len(initial_size_array):
            raise ValueError('Inconsistent initial_size_array and initial_mutant_gene_array. Ensure same length.')
        elif not isinstance(initial_size_array, np.ndarray):
            raise TypeError("Unexpected type for initial_mutant_gene_array. Should be integer or array like")

        if self.mutation_generator is not None and self.mutation_generator.multi_gene_array:
            # Make sure the fitness array has the appropriate dimensions.
            # All non-mutated genes have np.nan
            # First column is the wild type/non-gene associated fitness. Can still vary if specified.
            if isinstance(self.fitness_array, float):
                # Assume all genes have the same fitness
                fitness_array = np.full((len(initial_size_array), len(self.mutation_generator.genes)+1),
                                             np.nan, dtype=float)
                fitness_array[:, 0] = self.fitness_array
                self.fitness_array = fitness_array
            else:
                if self.fitness_array.shape == (len(initial_size_array),):
                    # One dimensional array.
                    # One gene mutated for each initial clone (could be the wild type fitness that is given)

                    # Make blank array
                    blank_fitness_array = np.full((len(initial_size_array), len(self.mutation_generator.genes)+1),
                                                  np.nan, dtype=float)
                    blank_fitness_array[:, 0] = self._wt_fitness
                    blank_fitness_array[
                        np.arange(len(initial_size_array)), self.initial_mutant_gene_array + 1] = self.fitness_array

                    self.fitness_array = blank_fitness_array
                elif self.fitness_array.shape == (len(initial_size_array), len(self.mutation_generator.genes)+1):
                    pass  # Fully specified fitness array for each initial clone.
                else:
                    raise ValueError("Incorrect shape of fitness_array. \
                    Ensure either 1D or has a wild type column plus column for each gene and \
                    that the length/num rows equals the number of initial clones.")
        else:  # Just want 1D fitness array
            if isinstance(self.fitness_array, float):
                self.fitness_array = np.full(len(initial_size_array), self.fitness_array, dtype=float)
            elif len(self.fitness_array) != len(initial_size_array):
                raise ValueError('Inconsistent initial_size_array and fitness_array. Ensure same length.' +
                                      '\nlen(initial_size_array): {}, len(fitness_array): {}'.format(
                                          len(initial_size_array), len(self.fitness_array)))

        self._check_mutation_rates()

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

        if np.any(self.mutation_rates[:, 1] > 0) and self.mutation_generator is None:
            raise ValueError('Cannot generate mutations without a mutation generator set.')


fitness_validation_type = Annotated[
    (Annotated[FitnessParameters, Tag("Base")] | Annotated[FitnessValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
