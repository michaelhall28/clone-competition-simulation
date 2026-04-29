from typing import Annotated, Literal

import numpy as np
from pydantic import BeforeValidator, ConfigDict, Tag

from .fitness_validation import FitnessValidator
from .population_validation import PopulationValidator
from .validation_utils import (FloatOrArrayParameter, IntOrArrayParameter,
                               ParameterBase, ValidationBase,
                               assign_config_settings)


class LabelParameters(ParameterBase):
    """Parameters that control how labels are assigned to clones.

    Fields:
        initial_label_array:
            An integer or array specifying the initial label for each clone in the
            starting population. If an integer is provided, that label is applied to
            every clone. If an array is provided, its length must match the number of
            initial clones. If not provided, all clones are assigned a default label of 0.

            Example:
                initial_label_array = 1
                initial_label_array = [0, 1, 0]

        label_times:
            A scalar or array of times at which labels should be applied. If this is
            defined, then both ``label_frequencies`` and ``label_values`` must also be
            defined. The times may be a single float/int or a sequence of floats.

            Example:
                label_times = 10
                label_times = [5, 15, 30]

        label_frequencies:
            A scalar or array giving the fraction of the population to label at each
            corresponding time. When a single value is provided it is applied to every
            time point. When an array is provided, it must have the same length as
            ``label_times``.

            Example:
                label_frequencies = 0.1
                label_frequencies = [0.05, 0.1, 0.2]

        label_values:
            A scalar or array of label identifiers to assign at each corresponding time.
            If a scalar is supplied it is reused for all times. If an array is supplied,
            it must have the same length as ``label_times``.

            Example:
                label_values = 1
                label_values = [1, 2, 3]

        label_fitness:
            Optional fitness effect associated with each applied label. This can be a
            scalar or array. If an array is provided, it must align with the number of
            label times. If provided as a scalar, it is reused for every label event.
            If multiple labels are applied and the fitness effect depends on mutations,
            a multi-gene mutation generator is required.

            Example:
                label_fitness = 1.02
                label_fitness = [1.0, 1.05, 0.99]

        label_genes:
            Optional gene index or array of gene indices targeted by each label event.
            If an integer is supplied, the same gene index is used for all label times.
            If a list/array is supplied, it must have the same length as
            ``label_times``. The value ``-1`` indicates no specific gene targeting.

            Example:
                label_genes = -1
                label_genes = [0, 1, -1]

    Examples:
        Apply a single initial label to all clones:
            initial_label_array = 1

        Apply different starting labels to each clone:
            initial_label_array = [0, 1, 0, 2]

        Label 10% of the population at time 5 with label 1:
            label_times = 5
            label_frequencies = 0.1
            label_values = 1

        Apply two label events at times 5 and 20:
            label_times = [5, 20]
            label_frequencies = [0.05, 0.1]
            label_values = [1, 2]
            label_fitness = [0.0, 0.02]
            label_genes = [-1, 3]
    """
    _field_name = "labels"
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    initial_label_array: IntOrArrayParameter = None
    label_times: FloatOrArrayParameter = None
    label_frequencies: FloatOrArrayParameter= None
    label_values: FloatOrArrayParameter = None
    label_fitness: FloatOrArrayParameter = None
    label_genes: IntOrArrayParameter = None


class LabelValidator(LabelParameters, ValidationBase):
    _default_label = 0
    tag: Literal['Full']
    population: PopulationValidator
    fitness: FitnessValidator
    config_file_settings: LabelParameters | None = None

    def _validate_model(self):
        self.initial_label_array = self.get_value_from_config("initial_label_array")
        initial_size_array = self.population.initial_size_array
        if self.initial_label_array is None:
            # Let's not be strict here. Just use the default label
            self.initial_label_array = self._default_label

        if isinstance(self.initial_label_array, int):
            self.initial_label_array = np.full_like(initial_size_array, self.initial_label_array)
        elif len(self.initial_label_array) != len(initial_size_array):
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

            if self.label_fitness is not None and len(self.label_frequencies) > 1 and self.fitness.mutation_generator is None:
                raise ValueError(
                    'Applying multiple labels with fitness effects requires a mutation generator to define ' \
                    'how fitness combines across labels and mutations.')

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
                    if self.fitness.mutation_generator and not self.fitness.mutation_generator.multi_gene_array:
                        raise ValueError('Applying labels with mutations to particular genes requires a '
                                                 'mutation generator with multi_gene_array=True')

            if len(self.label_times) != len(self.label_frequencies) or len(self.label_times) != len(self.label_values) \
                    or len(self.label_times) != len(self.label_fitness):
                raise ValueError('Length of label times, frequencies and values is not consistent.')


label_validation_type = Annotated[
    (Annotated[LabelParameters, Tag("Base")] | Annotated[LabelValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
