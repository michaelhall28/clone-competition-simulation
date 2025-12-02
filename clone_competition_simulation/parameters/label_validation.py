from typing import Annotated, Literal
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import assign_config_settings, ValidationBase
from .population_validation import PopulationValidator
from .fitness_validation import FitnessValidator


class LabelParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label_array: np.ndarray | int | None = None
    label_times: np.ndarray | None = None
    label_frequencies: np.ndarray | None = None
    label_values: np.ndarray | None = None
    label_fitness: np.ndarray | None = None
    label_genes: np.ndarray | None = None


class LabelValidator(LabelParameters, ValidationBase):
    _default_label = 0
    tag: Literal['Full']
    population: PopulationValidator
    fitness: FitnessValidator
    config_file_settings: LabelParameters | None = None

    def _validate_model(self):
        self.label_array = self.get_value_from_config("label_array")
        initial_size_array = self.population.initial_size_array
        if self.label_array is None:
            # Let's not be strict here. Just use the default label
            self.label_array = self._default_label

        if isinstance(self.label_array, int):
            self.label_array = np.full_like(initial_size_array, self._default_label)
        elif len(self.label_array) != len(initial_size_array):
            raise ValueError('Inconsistent initial_size_array and label_array. Ensure same length.')

        if self.label_times is not None:
            if self.label_frequencies is None or self.label_values is None:
                raise ValueError('Label frequencies and label values must also be defined to apply labels.')
            if isinstance(self.label_times, (int, float)):
                self.label_times = np.array([self.label_times])
            elif isinstance(self.label_times, list):
                self.label_times = np.array(self.label_times)
            elif not isinstance(self.label_times, np.ndarray):
                raise ValueError('Do not recognise the type of the label times.')

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
                    if self.fitness.mutation_generator and self.fitness.mutation_generator.multi_gene_array:
                        raise ValueError('Applying labels with mutations to particular genes requires a '
                                                 'mutation generator with multi_gene_array=True')

            if len(self.label_times) != len(self.label_frequencies) or len(self.label_times) != len(self.label_values) \
                    or len(self.label_times) != len(self.label_fitness):
                raise ValueError('Length of label times, frequencies and values is not consistent.')


label_validation_type = Annotated[
    (Annotated[LabelParameters, Tag("Base")] | Annotated[LabelValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
