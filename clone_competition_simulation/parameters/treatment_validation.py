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
from clone_competition_simulation.fitness.fitness_classes import UnboundedFitness
from loguru import logger


class TreatmentParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    treatment_timings: np.ndarray | None = None
    treatment_effects: np.ndarray | None = None
    treatment_replace_fitness: np.ndarray | None = None


class TreatmentValidator(TreatmentParameters, ValidationBase):
    tag: Literal['Full']
    population: PopulationValidator
    fitness: FitnessValidator
    config_file_settings: TreatmentParameters | None = None

    def _validate_model(self):
        self.treatment_timings = self.get_value_from_config("treatment_timings")
        self.treatment_effects = self.get_value_from_config("treatment_effects")
        if self.treatment_timings is None and self.treatment_effects is None:
            self.treatment_effects = 1
            self.treatment_timings = None
            self.treatment_replace_fitness = False
        else:
            mutation_generator = self.fitness.mutation_generator
            if mutation_generator and not isinstance(mutation_generator.mutation_combination_class, UnboundedFitness):
                if self.fitness.mutation_generator.multi_gene_array:
                    logger.debug('Treatment effects will be applied prior to the diminishing returns')
                else:
                    logger.debug('Treatment effects will be applied after the diminishing returns')
            if not isinstance(self.treatment_timings, (list, np.ndarray)) or not isinstance(self.treatment_effects,
                                                                                            (list, np.ndarray)):
                raise ValueError('treatment_timings and treatment_effects must be lists or arrays')
            elif len(self.treatment_timings) != len(self.treatment_effects):  # One time for each treatment
                raise ValueError('treatment_timings and treatment_effects must be same length')
            elif mutation_generator and mutation_generator.multi_gene_array:
                # Fitness changes apply to each gene
                logger.debug('Treatment effects are per gene')
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
                    self.treatment_timings = [0] + self.treatment_timings

                if any([len(t) != len(self.population.initial_size_array) for t in self.treatment_effects]):
                    raise ValueError(
                        'Each treatment effect must have the same length as the number of initial mutations.')

            # Convert the treatment effects to np.array
            self.treatment_effects = np.array(self.treatment_effects)


treatment_validation_type = Annotated[
    (Annotated[TreatmentParameters, Tag("Base")] | Annotated[TreatmentValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
