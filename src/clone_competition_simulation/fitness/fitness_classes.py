"""
Fitness sampling and fitness-transformation classes.

This module provides utilities for sampling mutation fitness effects, combining
those effects across genes, and transforming raw clone fitness values.
"""
from typing import Self, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict, Field, model_validator

from .fitness_distributions import DistributionProtocol
from .fitness_transformations import FitnessTransform, UnboundedFitness
from .fitness_combination import (
    FitnessCombinationType, 
    GeneCombinationType, 
    FITNESS_COMBINATION_FUNCTIONS, 
    GENE_COMBINATION_FUNCTIONS
)


################
# Class for storing information about a gene
class Gene(BaseModel):
    """Gene-level mutation metadata.

    Attributes
    ----------
    name : str
        Name of the gene.
    mutation_distribution : DistributionProtocol
        Distribution used to sample the fitness effect of each non-synonymous mutation.
    synonymous_proportion : float
        Fraction of mutations in this gene that are synonymous, i.e. have no effect on fitness.
    weight : float
        Relative mutation weight (how many of the simulation mutations will appear in this gene) 
        for this gene compared to other genes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    mutation_distribution: DistributionProtocol
    synonymous_proportion: float
    weight: float = 1

    @field_validator("synonymous_proportion", mode="after")
    @classmethod
    def validate_proportion(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError('synonymous_proportion must be between 0 and 1')
        return v

    @field_validator("weight", mode="after")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError('weight must be greater than 0')
        return v

    def __str__(self) -> str:
        return "Gene. Name: {0}, MutDist: {1}, SynProp: {2}, Weight: {3}".format(self.name,
                                                                                 self.mutation_distribution.__str__(),
                                                                                 self.synonymous_proportion,
                                                                                 self.weight)


class EpistaticEffect(BaseModel):
    """Defines an epistatic fitness interaction among multiple genes.

    Attributes
    ----------
    name : str
        Name of the epistatic interaction.
    gene_names : set[str]
        Genes participating in the interaction.
    fitness_distribution : DistributionProtocol
        Distribution for the epistatic fitness effect.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    gene_names: set[str]
    fitness_distribution: DistributionProtocol

    @field_validator('gene_names')
    @classmethod
    def validate_gene_names(cls, v):
        if len(v) <= 1:
            raise ValueError(
                "Must have at least two genes in an epistatic effect"
            )
        return v


class EpistaticEffectInfo(EpistaticEffect):
    """Extending the EpistaticEffect with information about related 
    genes and EpistaticEffects. 

    Parameters
    ----------
    index: int
        The index of this effect in the epistatics list
    full_col_idx: int
        The index of this effect in the full fitness array including WT
        and genes
    gene_columns: tuple[int]
        Columns in the fitness array for the genes
    subsets: list[int]
        Indices of genes and epistatic effects which have their fitness
        values superseded by this effect. Index including genes, i.e 
        first gene = 1
    subset_epistatics: list[int]
        Indices of any epistatic effects superseded by this one. Index 
        of just epistatic effects. First epistatic effect = 0.
    """
    index: int
    full_col_idx: int
    gene_columns: tuple[int, ...]
    subset_cols: list[int]
    superset_epistatics: list[int] = Field(default_factory=list)

    @classmethod
    def from_effect(cls, index: int, 
                    epistatic_effect: EpistaticEffect, 
                    fitness_calculator: "FitnessCalculator") -> Self:
        
        # Add one to the gene number to get the column index
        gene_columns = tuple(
            fitness_calculator.get_gene_number(e) + 1
            for e in epistatic_effect.gene_names
        )
        subset_cols = list(gene_columns)
        return cls(
            full_col_idx=fitness_calculator._num_genes + index + 1,
            index=index,
            gene_columns=gene_columns,
            subset_cols=subset_cols,
            **epistatic_effect.model_dump()
        )
    
    def register_if_superset(self, ee: "EpistaticEffectInfo"):
        """Add the index of an epistatic effect if it is a superset of this one

        Also add the column of this effect to the subsets of the input ee

        Parameters
        ----------
        ee : EpistaticEffectInfo
            Other epistatic effect to compare to. 
        """
        if (self.index != ee.index) and ee.gene_names.issuperset(self.gene_names):
            self.superset_epistatics.append(ee.index)
            ee.subset_cols.append(self.full_col_idx)


##################
# Class to put it all together

class FitnessCalculator(BaseModel):
    """Calculate clone fitness from gene-level mutation effects.

    The calculator samples new mutations from a set of genes and combines gene-level
    fitness effects according to mutation combination rules and 
    across-gene aggregation rules.

    Parameters
    ----------
    genes : list[Gene]
        List of gene definitions used for mutation sampling.
    combine_mutations : Callable:FitnessCombinationType, default multiply_fitness
        How fitness effects are combined within a gene.
    multi_gene_array : bool, default False
        Whether to keep per-gene fitness effects separate before combining them. 
        Required for epistatic interactions and treatments with per-gene fitness effects.
    combine_array : Callable:GeneCombinationType, default multiply_array_fitness
        How fitness effects are combined across genes, used when ``multi_gene_array``
        is ``True``.
    mutation_combination_class : FitnessTransform, default UnboundedFitness()
        Transformation applied to the combined raw fitness values.
    epistatics : list[EpistaticEffect] | None, default None
        Optional epistatic interactions that override per-gene fitness effects.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    genes: list[Gene]
    combine_mutations: FitnessCombinationType = FITNESS_COMBINATION_FUNCTIONS["multiply"]
    multi_gene_array: bool = False
    combine_array: GeneCombinationType = GENE_COMBINATION_FUNCTIONS["multiply"]
    mutation_combination_class: FitnessTransform = Field(default_factory=UnboundedFitness)
    epistatics: list[EpistaticEffect] | None = None

    # attributes used internally
    _num_genes: int | None = None
    _gene_indices: dict[str, int] | None = None
    _mutation_distributions: list[DistributionProtocol] | None = None
    _synonymous_proportions: NDArray[np.float64] | None = None
    _overall_synonymous_proportion: float | None = None
    _relative_weights_cumsum: NDArray[np.float64] | None = None
    _epistatics_info: list[EpistaticEffectInfo] | None = None
    _epistatic_cols: NDArray[np.int64] | None = None

    @field_validator("combine_mutations", mode="before")
    @classmethod
    def validate_combine_mutations(cls, func: Callable | str):
        """Validate that the combine_mutations function is a callable"""
        if isinstance(func, str):
            if func not in FITNESS_COMBINATION_FUNCTIONS:
                raise ValueError(f"combine_mutations must be one of {list(FITNESS_COMBINATION_FUNCTIONS.keys())}")
            func = FITNESS_COMBINATION_FUNCTIONS[func]
        if not callable(func):
            raise ValueError("combine_mutations must be callable")

        return func

    @field_validator("combine_array", mode="before")
    @classmethod
    def validate_combine_array(cls, func: Callable | str):
        """Validate that the combine_array function is a callable."""
        if isinstance(func, str):
            if func not in GENE_COMBINATION_FUNCTIONS:
                raise ValueError(f"combine_array must be one of {list(GENE_COMBINATION_FUNCTIONS.keys())}")
            func = GENE_COMBINATION_FUNCTIONS[func]
        if not callable(func):
            raise ValueError("combine_array must be callable")

        return func

    @model_validator(mode="after")
    def validate_fitness_calculator(self) -> Self:
        """Validate and finalize internal calculator state after model creation.

        This method computes derived fields such as gene indices, distribution lists,
        synonymous proportions, and epistatic index lookups.
        """
        self._num_genes = len(self.genes)
        self._gene_indices = {g.name: i for i, g in enumerate(self.genes)}
        self._mutation_distributions = [g.mutation_distribution for g in self.genes]
        gene_weights = np.array([g.weight for g in self.genes])
        self._synonymous_proportions = np.array([g.synonymous_proportion for g in self.genes])
        self._overall_synonymous_proportion = np.array(
            [g.synonymous_proportion * g.weight for g in self.genes]).sum() / gene_weights.sum()

        relative_weights = gene_weights / gene_weights.sum()
        self._relative_weights_cumsum = relative_weights.cumsum()
        if self.epistatics is not None:
            if not self.multi_gene_array:
                logger.debug('Using multi_gene_array because there are epistatic relationships')
                self.multi_gene_array = True
            # Add the names to the gene list
            self._gene_indices.update({e.name: j+self._num_genes for j, e in enumerate(self.epistatics)})

            # Process the epistatic effects
            self._process_epistatic_effects()
            self._epistatic_cols = np.arange(len(self.genes) + 1, len(self.genes) + len(self.epistatics) + 1)

        return self
    
    def _process_epistatic_effects(self) ->  dict[tuple[int, ...], EpistaticEffect]:
        """Find included genes and epistatic effects for each epistatic

        Epistatic effects for larger gene sets will replace the effect
        for any subsets. 
        E.g. effect for Gene1 + Gene2 will be replaced by the effect
        for Gene1 + Gene2 + Gene3. 
        """
        if not self.multi_gene_array:
            logger.debug('Using multi_gene_array because there are epistatic relationships')
            self.multi_gene_array = True
        
        # Add the names to the gene list
        self._gene_indices.update({e.name: j+self._num_genes for j, e in enumerate(self.epistatics)})
        self._epistatic_cols = np.arange(len(self.genes) + 1, len(self.genes) + len(self.epistatics) + 1)
        
        self._order_epistatic_effects()

    def _order_epistatic_effects(self) -> list[EpistaticEffectInfo]:
        # Create the information about each epistatic specific to this 
        # FitnessCalculator
        epistatic_info = [
            EpistaticEffectInfo.from_effect(
                index=i, 
                epistatic_effect=e, 
                fitness_calculator=self)
            for i, e in enumerate(self.epistatics)
        ]
        # Sort by number of genes involved.
        # This will make sure epistatics with larger sets of genes 
        # are applied before/instead of subsets of those genes
        epistatic_info.sort(key=lambda x: -len(x.gene_names))

        # Find any subsets
        for e in epistatic_info:
            for ee in epistatic_info:
                e.register_if_superset(ee)

        self._epistatics_info = epistatic_info

    def __str__(self) -> str:
        s = f"<MutGen: comb_muts={self.combine_mutations}, genes={self.genes}, fitness_class={self.mutation_combination_class}>"
        return s
    
    @property
    def n_cols(self) -> int:
        """The number of columns in the fitness array

        Returns
        -------
        int
            Number of columns in the array
        """
        if self.multi_gene_array:
            cols = 1 + len(self.genes)
            if self.epistatics is not None:
                cols += len(self.epistatics)
            return cols
        return 1

    def get_new_fitnesses(self, old_fitnesses: NDArray[np.float64], old_mutation_arrays: NDArray[np.float64]) \
        -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """Sample new mutation fitness and update clone fitness arrays.

        This method processes multiple mutated clones at once, sampling a new mutation
        for each clone and combining it with the existing clone fitness profile.

        Parameters
        ----------
        old_fitnesses : NDArray[np.float64]
            1D array of current clone fitness values used for clonal dynamics.
        old_mutation_arrays : NDArray[np.float64]
            2D array of per-gene fitness effects, including wild-type and any epistatic columns.

        Returns
        -------
        tuple
            ``(new_fitnesses, new_mutation_arrays, syns, genes_mutated)``.
        """
        num = len(old_fitnesses)
        genes_mutated = self._get_genes(num)
        syns = self._are_synonymous(genes_mutated)

        new_fitnesses, new_mutation_arrays = self._update_fitness_arrays(old_mutation_arrays, genes_mutated, syns)

        return new_fitnesses, new_mutation_arrays, syns, genes_mutated

    def _are_synonymous(self, mut_types: NDArray[np.int64]) -> NDArray:
        """Determine whether mutations are synonymous.

        Parameters
        ----------
        mut_types : NDArray[np.int64]
            Indices of genes selected for mutation.

        Returns
        -------
        NDArray
            Binary array where 1 indicates a synonymous mutation.
        """
        return np.random.binomial(1, self._synonymous_proportions[mut_types])

    def _get_genes(self, num: int) -> NDArray[np.int64]:
        """Select gene indices for new mutations.

        Parameters
        ----------
        num : int
            Number of mutated clones to sample.

        Returns
        -------
        NDArray[np.int64]
            Indices of selected genes for each clone.
        """
        r = np.random.rand(num, 1)
        k = (self._relative_weights_cumsum < r).sum(axis=1)
        return k

    def _update_fitness_arrays(self, old_mutation_arrays: NDArray[np.float64],
                               genes_mutated:  NDArray[np.int64],
                               syns: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update per-gene mutation arrays with newly sampled fitness effects.

        Parameters
        ----------
        old_mutation_arrays : NDArray[np.float64]
            Existing per-gene fitness effects for each clone.
        genes_mutated : NDArray[np.int64]
            Selected gene indices for the current mutations.
        syns : NDArray[np.int64]
            Binary array indicating whether each mutation is synonymous.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            ``(new_fitnesses, new_mutation_arrays)`` where ``new_fitnesses`` is the
            combined clone fitness and ``new_mutation_arrays`` is the updated gene-level array.
        """
        # Only have to update the cells in which non-synonymous mutations occur
        non_syns = np.where(1 - syns)
        new_mutation_fitnesses_non_syn = [self._mutation_distributions[g]() for g in
                                          genes_mutated[non_syns]]  # The fitness of the new mutation alone
        if self.multi_gene_array:
            array_idx = genes_mutated[non_syns] + 1  # +1 for the wild type column
        else:
            array_idx = np.zeros(len(non_syns), dtype=int)

        new_mutation_arrays = old_mutation_arrays.copy()

        # Get the effects of any existing mutations in the newly mutated genes
        old_fitnesses = old_mutation_arrays[non_syns, array_idx]
        # Combine the old effects with the new ones and assign to the new mutation array.
        new_fitnesses = self._combine_fitnesses(old_fitnesses, new_mutation_fitnesses_non_syn)
        new_mutation_arrays[(non_syns, array_idx)] = new_fitnesses

        # Combine the effects of mutations in different genes into a single 1D array of fitness per cell
        # Also applies any diminishing returns to the fitness
        new_fitnesses, new_mutation_arrays = self.combine_vectors(new_mutation_arrays)

        return new_fitnesses, new_mutation_arrays

    def _combine_fitnesses(self, old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) \
            -> NDArray[np.float64]:
        """Combine old and new mutation fitness values for a gene.

        Parameters
        ----------
        old_fitnesses : NDArray[np.float64]
            Existing fitness values for the gene, with ``np.nan`` for absent mutations.
        new_mutation_fitnesses : NDArray[np.float64]
            Newly sampled fitness values for the new mutations.

        Returns
        -------
        NDArray[np.float64]
            Combined fitness values after applying the configured mutation-combination rule.
        """

        old_fitnesses[np.where(np.isnan(old_fitnesses))] = 1
        return self.combine_mutations(old_fitnesses, new_mutation_fitnesses)

    def _epistatic_combinations(self, fitness_arrays: NDArray[np.float64]) \
            -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply epistatic interactions to fitness arrays.

        When a clone has mutated all genes in an epistatic set, the raw gene-level
        fitness effects for that set are replaced with the epistatic fitness
        distribution.

        Parameters
        ----------
        fitness_arrays : NDArray[np.float64]
            Per-gene fitness arrays, including wild-type and epistatic columns.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            ``(fitness_array, epistatic_fitness_array)`` where ``fitness_array`` is the
            raw fitness array with epistatic effects added and ``epistatic_fitness_array``
            has individual gene contributions blanked where the epistatic effect applies.
        """

        raw_gene_arr = fitness_arrays[:, :self._epistatic_cols[0]]
        non_nan = ~np.isnan(raw_gene_arr)

        epi_rows = fitness_arrays[:, self._epistatic_cols]
        not_already_epi_rows = np.isnan(epi_rows)
        row_positions_to_blank = []
        col_positions_to_blank = []
        for epistatic_effect in self._epistatics_info:
            # Find all rows where this epistatic effect applies
            matching_rows = np.all(
                non_nan[:, epistatic_effect.gene_columns], axis=1) 
            # Exclude rows it has already been applied to
            new_matching_rows = matching_rows * not_already_epi_rows[:, epistatic_effect.index]

            # Exclude any rows that already have a superseding effect applied
            for i in epistatic_effect.superset_epistatics:
                new_matching_rows *= not_already_epi_rows[:, i]
            
            # Draw new values for the epistatic effect and add to the epistatic fitness array
            new_draws = [epistatic_effect.fitness_distribution() for j in new_matching_rows if j]
            epi_rows[new_matching_rows, epistatic_effect.index] = new_draws

            for col in epistatic_effect.subset_cols:
                row_positions_to_blank.extend(np.where(matching_rows)[0])
                col_positions_to_blank.extend([col] * matching_rows.sum())

            # Update the positions that don't have epistatic effects applied
            not_already_epi_rows = np.isnan(epi_rows)


        fitness_array = np.concatenate([raw_gene_arr, epi_rows], axis=1)
        epistatic_fitness_array = fitness_array.copy()
        epistatic_fitness_array[row_positions_to_blank, col_positions_to_blank] = np.nan
        return fitness_array, epistatic_fitness_array

    def combine_vectors(self, fitness_arrays: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine per-gene fitness arrays into clone fitness values.

        Parameters
        ----------
        fitness_arrays : NDArray[np.float64]
            Per-gene fitness array for each clone.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            ``(combined_fitness, full_fitness_arrays)`` where ``combined_fitness`` is
            the transformed clone fitness vector and ``full_fitness_arrays`` is the
            expanded gene-level array including epistatic effects.
        """
        # Combines the raw fitness values from each gene. Can apply any diminishing returns etc here.
        if self.epistatics is not None:
            # Replace the raw fitness array with one including the epistatic effects
            full_fitness_arrays, fitness_arrays = self._epistatic_combinations(fitness_arrays)
            # fitness_arrays now updated for calculation of epistatic fitness
            # full_fitness_arrays also contains the raw fitness of the genes
        else:
            full_fitness_arrays = fitness_arrays

        if self.multi_gene_array:
            combined_fitness = self.combine_array(fitness_arrays)
        else: # Don't have to combine genes, just reduce to 1D array
            combined_fitness = fitness_arrays[:, 0]

        return self.mutation_combination_class.fitness(combined_fitness), full_fitness_arrays

    def get_gene_number(self, gene_name: str) -> int:
        """Convert a gene name to its numeric index.

        Parameters
        ----------
        gene_name : str
            Name of the gene or epistatic effect.

        Returns
        -------
        int
            Index of the gene in the internal gene list.

        Raises
        ------
        ValueError
            If the gene name is not found.
        """
        if gene_name not in self._gene_indices:
            raise ValueError(f"Gene name {gene_name} not found")
        return self._gene_indices[gene_name]

    def get_gene_name(self, gene_number: int | float | None) -> str | None:
        """Convert a gene index to a gene name.

        Parameters
        ----------
        gene_number : int | float | None
            Index of a gene in the gene list, or ``np.nan`` for no mutation.

        Returns
        -------
        str | None
            Name of the gene or epistatic effect, or ``None`` if input is ``None`` or ``np.nan``.

        Raises
        ------
        ValueError
            If ``gene_number`` is negative.
        """
        if gene_number is None or np.isnan(gene_number):
            return None
        if gene_number < 0:
            raise ValueError(f"Gene number {gene_number} is invalid. Must be non-negative or None.")
        return self.genes[int(gene_number)].name

    def get_synonymous_proportion(self, gene_num: int | None) -> float:
        """Return the synonymous mutation proportion for a gene.

        Parameters
        ----------
        gene_num : int | None
            Gene index, or ``None`` to return the overall weighted average.

        Returns
        -------
        float
            Synonymous mutation proportion for the requested gene or overall average.
        """
        if gene_num is None:
            return self._overall_synonymous_proportion
        else:
            return self._synonymous_proportions[gene_num]

    def plot_fitness_combinations(self) -> pd.Series:
        """Plot fitness outcomes for all combinations of gene mutations.

        This function visualizes the average clone fitness for every possible
        combination of gene mutations defined in ``self.genes``. It is primarily
        intended as a sanity check for configured mutation and epistatic rules.

        Returns
        -------
        NDArray[np.float64]
            Computed fitness values for all mutation combinations.
        """
        if not self.multi_gene_array and self.combine_mutations == GENE_COMBINATION_FUNCTIONS['replace']:
            # No combinations here. Just need to plot individual genes
            logger.info('No combinations defined. Only most recent non-silent mutation defines fitness')
            xticklabels = ['Background']
            fitness_values = [1.]
            for i, gene in enumerate(self.genes):
                xticklabels.append(gene.name)
                fitness_values.append(gene.mutation_distribution.get_mean())
            plt.bar(range(len(fitness_values)), fitness_values)
            plt.ylabel('Fitness')
            plt.xticks(range(len(fitness_values)), xticklabels, rotation=90)
            fitness_values = pd.Series(
                fitness_values, index=xticklabels
            )
            return fitness_values
        else:
            # Make a fitness array with all possible combinations of genes
            num_genes = len(self.genes)

            if self.epistatics is None:
                num_epi = 0
            else:
                num_epi = len(self.epistatics)
            fitness_array = np.full((2 ** num_genes, num_genes + num_epi + 1), np.nan)

            fitness_array[:, 0] = 1  # Assume background fitness is 1
            xticklabels = ['Background']
            for i in range(fitness_array.shape[0]):
                binary_string = format(i, '#0{}b'.format(num_genes + 2))[2:][::-1]
                tick_label = []
                for j, b in enumerate(binary_string):
                    if b == '1':
                        # Mutate the gene
                        gene_fitness = self._mutation_distributions[j].get_mean()
                        fitness_array[i, j + 1] = gene_fitness
                        tick_label.append(self.genes[j].name)
                if i > 0:
                    xticklabels.append(" + ".join(tick_label))

            if self.multi_gene_array:
                new_fitnesses, new_mutation_arrays = self.combine_vectors(fitness_array)
            else:
                raise NotImplementedError("Plotting of non-multi-gene-array fitness not implemented")

            plt.bar(range(fitness_array.shape[0]), new_fitnesses)
            plt.ylabel('Fitness')
            plt.xticks(range(fitness_array.shape[0]), xticklabels, rotation=90)

            new_fitnesses = pd.Series(
                new_fitnesses, index=xticklabels
            )
            return new_fitnesses