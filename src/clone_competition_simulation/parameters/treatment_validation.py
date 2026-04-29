from typing import Annotated, Literal

import numpy as np
from loguru import logger
from pydantic import BeforeValidator, ConfigDict, Tag

from ..fitness.fitness_classes import UnboundedFitness
from .fitness_validation import FitnessValidator
from .population_validation import PopulationValidator
from .validation_utils import (FloatArrayParameter, ParameterBase,
                               ValidationBase, assign_config_settings)


class TreatmentParameters(ParameterBase):
    """Parameters that control when and how treatment is applied during the simulation.
    Treatments change the fitness of clones/gene mutations at specified times, which can be used to 
    model drug treatments or other interventions.

    Fields:
        treatment_timings:
            A list or array of times at which treatment changes occur. Each entry
            corresponds to a change in treatment effect. For a neutral treatment or
            no treatment changes, this may be left unset.

            Example:
                treatment_timings = [0, 10, 20]

        treatment_effects:
            A list or array of treatment effect values to apply at the corresponding
            ``treatment_timings``. If the simulation uses a multi-gene mutation
            generator, each treatment effect entry must have one value per gene plus
            one wild-type term. For non-multi-gene simulations, each effect entry
            should have one value per initial clone.

            Example:
                treatment_effects = [0.5, 1.0, 0.8]

        treatment_replace_fitness:
            A boolean that controls how the treatment effect is applied to fitness.
            If ``True``, the treatment effect replaces the existing fitness values.
            If ``False``, the treatment is applied multiplicatively to the current
            fitness values. When unset, the default is ``False`` (i.e. multiplicative 
            application).

            Example:
                treatment_replace_fitness = False

    Examples:
        Single treatment change at time 10:
            treatment_timings = [10]
            treatment_effects = [0.75]
            treatment_replace_fitness = False

        Two treatment changes with multi-gene effects:
            treatment_timings = [0, 15]
            treatment_effects = [[1.0, 0.8, 0.8], [1.0, 0.7, 0.6]]
            treatment_replace_fitness = True
    """
    _field_name = "treatment"
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    treatment_timings: FloatArrayParameter = None
    treatment_effects: FloatArrayParameter = None
    treatment_replace_fitness: bool | None = None


class TreatmentValidator(TreatmentParameters, ValidationBase):
    tag: Literal['Full']
    population: PopulationValidator
    fitness: FitnessValidator
    config_file_settings: TreatmentParameters | None = None

    def _validate_model(self):
        self.treatment_timings = self.get_value_from_config("treatment_timings")
        self.treatment_effects = self.get_value_from_config("treatment_effects")
        self.treatment_replace_fitness = self.get_value_from_config("treatment_replace_fitness")
        if self.treatment_timings is None and self.treatment_effects is None:
            self.treatment_effects = 1
            self.treatment_timings = None
            self.treatment_replace_fitness = False
        else:
            fitness_calculator = self.fitness.fitness_calculator
            if fitness_calculator and not isinstance(fitness_calculator.mutation_combination_class, UnboundedFitness):
                if self.fitness.fitness_calculator.multi_gene_array:
                    logger.debug('Treatment effects will be applied prior to the diminishing returns')
                else:
                    logger.debug('Treatment effects will be applied after the diminishing returns')
            if not isinstance(self.treatment_timings, (list, np.ndarray)) or not isinstance(self.treatment_effects,
                                                                                            (list, np.ndarray)):
                raise ValueError('treatment_timings and treatment_effects must be lists or arrays')
            elif len(self.treatment_timings) != len(self.treatment_effects):  # One time for each treatment
                raise ValueError('treatment_timings and treatment_effects must be same length')
            elif fitness_calculator and fitness_calculator.multi_gene_array:
                # Fitness changes apply to each gene
                logger.debug('Treatment effects are per gene')
                # Always needs to be a treatment so it can be applied to new mutations.
                # Insert a neutral treatment at start if initial treatment is not defined.
                if self.treatment_timings[0] != 0:
                    if self.treatment_replace_fitness:
                        self.treatment_effects = [np.full(len(fitness_calculator.genes) + 1, np.nan)] + list(
                            self.treatment_effects)
                    else:
                        self.treatment_effects = [np.ones(len(fitness_calculator.genes)+1)] + list(self.treatment_effects)
                    self.treatment_timings = np.concatenate(([0], self.treatment_timings))

                if any([len(t) != len(fitness_calculator.genes)+1 for t in self.treatment_effects]):
                    raise ValueError(
                        'Each treatment effect must have the same length as the number of genes plus 1.')
            else:
                # Fitness changes apply to each clone
                if not np.array_equal(self.fitness.mutation_rates, np.array([[0, 0]])):
                    raise ValueError(
                        'Cannot have treatment changes and introduce new mutations unless using a multi-gene array')

                logger.debug('Treatment effects are per clone.')
                # Always needs to be a treatment so it can be applied to new mutations.
                # Insert a neutral treatment at start if initial treatment is not defined.
                if self.treatment_timings[0] != 0:
                    if self.treatment_replace_fitness:
                        self.treatment_effects = [np.full(len(self.population.initial_size_array), np.nan)] + list(
                            self.treatment_effects)
                    else:
                        self.treatment_effects = [np.ones(len(self.population.initial_size_array))] + list(
                            self.treatment_effects)
                    self.treatment_timings = np.concatenate(([0], self.treatment_timings))

                if any([len(t) != len(self.population.initial_size_array) for t in self.treatment_effects]):
                    raise ValueError(
                        'Each treatment effect must have the same length as the number of initial mutations.')

            # Convert the treatment effects to np.array
            self.treatment_effects = np.array(self.treatment_effects)


treatment_validation_type = Annotated[
    (Annotated[TreatmentParameters, Tag("Base")] | Annotated[TreatmentValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
