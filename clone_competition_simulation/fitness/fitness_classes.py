"""
Functions/classes to randomly draw and/or calculate the fitness of clones
"""
import math
from enum import Enum
from typing import Protocol, TypeVar, runtime_checkable, Callable, Self

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict, Field, model_validator


# Probability distributions for drawing the fitness of new mutations.
# Set up so can be called like functions without argument, but can print the attributes

@runtime_checkable  # So it can be used in Pydantic fields
class DistributionProtocol(Protocol):

    def __call__(self) -> float:
        ...

    def get_mean(self) -> float:
        ...


class NormalDist:
    """
    A wrapper for numpy.random.normal
    Will draw again if the value is below zero.
    """
    def __init__(self, var: float, mean: float=1.):
        self.var = var
        self.mean = mean

    def __str__(self):
        return 'Normal distribution(mean {0}, variance {1})'.format(self.mean, self.var)

    def __call__(self) -> float:
        g = np.random.normal(self.mean, self.var)
        if g < 0:
            return self()
        return g

    def get_mean(self) -> float:
        return self.mean


class FixedValue:
    """
    Just returns the fixed value given.
    This is a wrapper for that number so that it functions like the random distributions.
    """
    def __init__(self, value: float):
        self.mean = value

    def __str__(self):
        return 'Fixed value {0}'.format(self.mean)

    def __call__(self) -> float:
        return self.mean

    def get_mean(self) -> float:
        return self.mean


class ExponentialDist:
    """
    An exponential distribution.
    A wrapper for numpy.random.exponential

    The parameters are mean and offset.
    The mean defines the mean of the distribution after the offset has been applied,
    i.e. the random number is
    offset +  np.random.exponential(mean - offset)

    mean must be greater than the offset.
    """
    def __init__(self, mean: float, offset: float=1):
        if mean <= offset:
            raise ValueError('mean must be greater than offset')
        self.mean = mean
        self.offset = offset  # Offset of 1 means the mutations will start from neutral.

    def __str__(self):
        return 'Exponential distribution(mean {0}, offset {1})'.format(self.mean, self.offset)

    def __call__(self) -> float:
        g = self.offset + np.random.exponential(self.mean - self.offset)
        return g

    def get_mean(self) -> float:
        return self.mean


class UniformDist:
    """
    A uniform distribution between low and high
    A wrapper for numpy.random.uniform
    """
    def __init__(self, low: float, high: float):
        if low >= high:
            raise ValueError('high bound must be higher than the low bound')
        self.low = low
        self.high = high

    def __str__(self):
        return 'Uniform distribution(low {0}, high {1})'.format(self.low, self.high)

    def __call__(self) -> float:
        g = np.random.uniform(self.low, self.high)
        return g

    def get_mean(self) -> float:
        return (self.high + self.low)/2


##################
# Classes for diminishing returns or other transformations of the raw fitness
T = TypeVar('T', bound=float | NDArray[float])

@runtime_checkable  # So it can be used in Pydantic fields
class FitnessTransform(Protocol):
    def fitness(self, x: T) -> T:
        ...

    def inverse(self, y: T) -> T:
        ...


class UnboundedFitness:
    """
    No bound or transformation on the fitness.
    """

    def __str__(self):
        return 'UnboundedFitness'

    def fitness(self, x: T) -> T:
        return x

    def inverse(self, y: T) -> T:
        return y


class BoundedLogisticFitness:
    """
    The effect of new (beneficial) mutations tails off as the clone gets stronger.
    There is a maximum fitness.
    """
    def __init__(self, a: float, b: float=math.exp(1)):
        """
        fitness = a/(1+c*b**(-x)) where x is the product of all mutation effects
        c is picked so that fitness(1) = 1
        :param a: The maximum output fitness of a clone
        :param b: Controls the slope of the transformation
        """
        assert (a > 1)
        assert (b > 1)
        self.a = a
        self.b = b
        self.c = (a - 1) * self.b

    def __str__(self):
        return 'Bounded Logistic: a {0}, b {1}, c {2}'.format(self.a, self.b, self.c)

    def fitness(self, x: T) -> T:
        return self.a / (1 + self.c * (self.b ** (-x)))

    def inverse(self, y: T) -> T:
        return np.emath.logn(self.b, (self.c / (self.a / y - 1)))


################
# Class for storing information about a gene
class Gene(BaseModel):
    """
    Determines the mutations which are created for a gene

        :param name: The name for the gene.
        :param mutation_distribution: A class from clone_competition_simulation.fitness_classes,
        e.g. NormalDist, UniformDist, ExponentialDist or FixedValue. This defines the distribution from which the
        fitness of a non-synonymous mutation in this gene is drawn.
        :param synonymous_proportion: The proportion of mutations in this gene which are synonymous. These will have no
        impact on cell fitness, but are used in dN/dS calculations.
        :param weight: Along with the mutations_rates, this determines the rate of mutation in this gene. The
        total mutation rate of all genes passed to the MutationGenerator will equal the mutation_rates defined in
        Parameters. The relative weights of the genes are used. E.g. if Gene1 has a weight of 3 and Gene2 has a
        weight of 7, then 30% of the mutations will be drawn from Gene1 and 70% from Gene2.

    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    mutation_distribution: DistributionProtocol
    synonymous_proportion: float
    weight: float = 1

    @field_validator("synonymous_proportion", mode="after")
    @classmethod
    def validate_proportion(cls, v):
        if v < 0 or v > 1:
            raise ValueError('synonymous_proportion must be between 0 and 1')
        return v

    @field_validator("weight", mode="after")
    @classmethod
    def validate_weight(cls, v):
        if v < 0:
            raise ValueError('weight must be greater than 0')
        return v

    def __str__(self):
        return "Gene. Name: {0}, MutDist: {1}, SynProp: {2}, Weight: {3}".format(self.name,
                                                                                 self.mutation_distribution.__str__(),
                                                                                 self.synonymous_proportion,
                                                                                 self.weight)


##################
# Functions for defining how to combine mutations in the same or different genes

def add_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    combined_fitness = old_fitnesses + new_mutation_fitnesses - 1
    combined_fitness[combined_fitness < 0] = 0
    return combined_fitness


class MutationCombination(Enum):
    MULTIPLY = "multiply", lambda old, new: old * new
    ADD = "add", add_fitness
    REPLACE = "replace", lambda old, new: new
    MAX = "max", lambda old, new: np.maximum(old, new)
    MIN = "min", lambda old, new: np.minimum(old, new)

    def __new__(cls, value, func):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.function = func
        return obj


def add_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    combined_fitness = np.nansum(fitness_arrays, axis=1) - np.count_nonzero(~np.isnan(fitness_arrays),
                                                                            axis=1) + 1
    combined_fitness[combined_fitness < 0] = 0
    return combined_fitness


def priority_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    # Find the right-most non-nan value. Useful for epistatic interactions that are superseded by another
    # To find the last non-nan columns, reverse the column order and find the first non-zero entry.
    fitness_arrays = fitness_arrays[:, ::-1]
    c = np.isnan(fitness_arrays)
    d = np.argmin(c, axis=1)
    combined_fitness = fitness_arrays[range(len(fitness_arrays)), d]
    return combined_fitness


class ArrayCombination(Enum):
    MULTIPLY = "multiply", lambda fitness_arrays: np.nanprod(fitness_arrays, axis=1)
    ADD = "add", add_array_fitness
    MAX = "max", lambda fitness_arrays: np.nanmax(fitness_arrays, axis=1)
    MIN = "min", lambda fitness_arrays: np.nanmin(fitness_arrays, axis=1)
    PRIORITY = "priority", priority_array_fitness

    def __new__(cls, value, func):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.function = func
        return obj


class EpistaticEffect(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    gene_names: list[str]
    fitness_distribution: DistributionProtocol



##################
# Class to put it all together

class MutationGenerator(BaseModel):
    """
    This class determines the effects of mutations and how they are combined to define the fitness of a clone.

    New mutations are drawn at random from the given genes
    Each gene in the model can have a different mutation fitness distribution and synonymous proportion.
     - If multi_gene_array=False, the effects of mutations are simply combined according to the combine_mutations option
     - If multi_gene_array=True, the effects of mutations within each gene are calculated first according to the
       combine_mutations option, then the effects of each gene are combined using the combine_array option.
       This is useful for cases such as where a second mutation in a gene will not have a further fitness effect.

    combine_mutations options:
        multiply - multiplies the fitness effect of all mutations in a gene to get a new fitness
        add - multiplies the fitness effect of all mutations in a gene to get a new fitness
        replace - a new mutation will define the fitness of a gene and any previous effects are ignored
        max - the new gene fitness is the max of the old and the new
        min - the new gene fitness is the min of the old and the new

    combine_array options (only if multi_gene_array=True):
        multiply - multiplies the fitness effects on each gene to get a new fitness for the cell
        add - adds the fitness effects on each gene to get a new fitness for the cell
        max - the cell fitness is given by the gene with the highest fitness
        min - the cell fitness is given by the gene with the minimum fitness
        priority - the cell fitness is given by the last gene in the list given
                   (or by the last epistatic effect in any are used)

    epistatics can be used to define more complex relationships between genes.
        This is a list of epistatic relationships which define a set of genes and the distribution of fitness effects if
        all of those genes are mutated (which replaces the fitness effects of those individual genes).
        Those epistatic effects are then combined (along with any genes not in a triggered epistatic effect)
        according to the combine_array option.
        Each item in the list is (name for epistatic, gene_name1, gene_name2, ..., epistatic fitness distribution)
        Only intended for quite simple combinations of a few genes.

    Fitness changes imposed by labelling events will be applied elsewhere.
    Any effects due to treatment are applied elsewhere.
    There are many options here and when applying treatment and labels, so be careful that the clone fitnesses are
    combining as intended, especially if defining epistatic effects.
    Can use MutationGenerator.plot_fitness_combinations to check the fitness combinations are as intended.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    genes: list[Gene]
    combine_mutations: MutationCombination=MutationCombination.MULTIPLY
    multi_gene_array: bool = False
    combine_array: ArrayCombination = ArrayCombination.MULTIPLY
    mutation_combination_class: FitnessTransform = Field(default_factory=UnboundedFitness)
    epistatics: list[EpistaticEffect] | None = None

    # attributes used internally
    num_genes: int | None = None
    gene_indices: dict[str, int] | None = None
    mutation_distributions: list[DistributionProtocol] | None = None
    synonymous_proportion: NDArray[np.float64] | None = None
    overall_synonymous_proportion: float | None = None
    relative_weights_cumsum: NDArray[np.float64] | None = None
    epistatics_dict: dict[tuple[int, ...], EpistaticEffect] | None = None
    epistatic_cols: NDArray[np.int64] | None = None
    combine_fitness_function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] = Field(
        None, validate_default=True)
    combine_array_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = Field(
        None, validate_default=True)

    @field_validator("combine_fitness_function", mode="before")
    @classmethod
    def value_combine_fitness_function(cls, v, info):
        # TODO allow any functions with a valid form, not just from the enum
        return info.data['combine_mutations'].function

    @field_validator("combine_array_function", mode="before")
    @classmethod
    def value_combine_array_function(cls, v, info):
        # TODO allow any functions with a valid form, not just from the enum
        return info.data['combine_array'].function

    @model_validator(mode="after")
    def validate_mutation_generator(self) -> Self:
        self.num_genes = len(self.genes)
        self.gene_indices = {g.name: i for i, g in enumerate(self.genes)}
        self.mutation_distributions = [g.mutation_distribution for g in self.genes]
        gene_weights = np.array([g.weight for g in self.genes])
        self.synonymous_proportion = np.array([g.synonymous_proportion for g in self.genes])
        self.overall_synonymous_proportion = np.array(
            [g.synonymous_proportion * g.weight for g in self.genes]).sum() / gene_weights.sum()

        relative_weights = gene_weights / gene_weights.sum()
        self.relative_weights_cumsum = relative_weights.cumsum()
        if self.epistatics is not None:
            if not self.multi_gene_array:
                logger.debug('Using multi_gene_array because there are epistatic relationships')
                self.multi_gene_array = True
            # Add the names to the gene list
            self.gene_indices.update({e.name: j+self.num_genes for j, e in enumerate(self.epistatics)})
            # Convert to the gene indices
            self.epistatics_dict = {tuple([self.get_gene_number(ee) for ee in e.gene_names]): e for e in self.epistatics}
            self.epistatic_cols = np.arange(len(self.genes) + 1, len(self.genes) + len(self.epistatics) + 1)

        return self

    def __str__(self):
        s = f"<MutGen: comb_muts={self.combine_mutations}, genes={self.genes}, fitness_class={self.mutation_combination_class}>"
        return s

    def get_new_fitnesses(self, old_fitnesses, old_mutation_arrays):
        """
        Gets the effects of the new mutations and combines them with the old mutations in those cells.
        Multiple mutated cells can be processed at once.
        However, each cell can only get one new mutation. If multiple mutations occur in the same step in the same cell
        then this function is called multiple times.
        :param old_fitnesses: 1D array of fitnesses. These are the actual fitness of the clones used for calculating
        clonal dynamics.
        :param old_mutation_arrays: 2D array of fitnesses. Has an array of fitness effects for each gene (plus WT and
        any epistatics used). This is updated with the new mutations and used to calculate the new overall fitness of
        each mutated cell.
        :return:
        """
        num = len(old_fitnesses)
        genes_mutated = self._get_genes(num)
        syns = self._are_synonymous(genes_mutated)

        new_fitnesses, new_mutation_arrays = self._update_fitness_arrays(old_mutation_arrays, genes_mutated, syns)

        return new_fitnesses, new_mutation_arrays, syns, genes_mutated

    def _are_synonymous(self, mut_types: NDArray[np.int64]) -> NDArray:
        """
        Determines whether the new mutations are synonymous
        :param mut_types:
        :return:
        """
        return np.random.binomial(1, self.synonymous_proportion[mut_types])

    def _get_genes(self, num: int) -> NDArray[np.int64]:
        """
        Determines which genes are mutated
        :param num:
        :return:
        """
        r = np.random.rand(num, 1)
        k = (self.relative_weights_cumsum < r).sum(axis=1)
        return k

    def _update_fitness_arrays(self, old_mutation_arrays: NDArray[np.float64],
                               genes_mutated:  NDArray[np.int64],
                               syns: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Only have to update the cells in which non-synonymous mutations occur
        non_syns = np.where(1 - syns)
        new_mutation_fitnesses_non_syn = [self.mutation_distributions[g]() for g in
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
        """
        Applies the selected rules to combine the new mutations with those already in the cell.
        If using multi_gene_array=True, this will just combine the fitness of mutations within the same gene.

        The arrays have nans where the gene is not mutated.
        Need to turn these into ones for the calculations.
        :param old_fitnesses:
        :param new_mutation_fitnesses:
        :return:
        """

        old_fitnesses[np.where(np.isnan(old_fitnesses))] = 1
        return self.combine_fitness_function(old_fitnesses, new_mutation_fitnesses)

    def _epistatic_combinations(self, fitness_arrays: NDArray[np.float64]) \
            -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Take the mutated (non-nan) genes and check whether they complete an epistatic set.
        The effects of genes in an epistatic set are replaced by the epistatic effect.
        Epistatic effects are stored in extra columns in the fitness array.
        Then any multiple epistatic results can be combined as usual (along with any uninvolved genes).
        Assume no mutations back to wild type, so once an epistatic effect is in a clone, it is not lost.
        :param fitness_arrays:
        :return:
        """

        raw_gene_arr = fitness_arrays[:, :self.epistatic_cols[0]]
        non_nan = ~np.isnan(raw_gene_arr)

        epi_rows = fitness_arrays[:, self.epistatic_cols]
        not_already_epi_rows = np.isnan(epi_rows)
        row_positions_to_blank = []
        col_positions_to_blank = []
        for i, (epi_genes, epi) in enumerate(self.epistatics_dict.items()):
            matching_rows = np.all(non_nan[:, tuple([g+1 for g in epi_genes])], axis=1)  # +1 because of the WT column
            new_matching_rows = matching_rows * not_already_epi_rows[:, i]
            new_draws = [epi.fitness_distribution() for j in new_matching_rows if j]
            epi_rows[new_matching_rows, i] = new_draws
            for g in epi_genes:
                row_positions_to_blank.extend(np.where(matching_rows)[0])
                col_positions_to_blank.extend([g + 1] * matching_rows.sum())

        fitness_array = np.concatenate([raw_gene_arr, epi_rows], axis=1)
        epistatic_fitness_array = fitness_array.copy()
        epistatic_fitness_array[row_positions_to_blank, col_positions_to_blank] = np.nan
        return fitness_array, epistatic_fitness_array

    def combine_vectors(self, fitness_arrays: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """

        :param fitness_arrays:
        :return:
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
            combined_fitness = self.combine_array_function(fitness_arrays)
        else: # Don't have to combine genes, just reduce to 1D array
            combined_fitness = fitness_arrays[:, 0]

        return self.mutation_combination_class.fitness(combined_fitness), full_fitness_arrays

    def get_gene_number(self, gene_name: str) -> int:
        if gene_name is None:
            return None
        return self.gene_indices[gene_name]

    def get_gene_name(self, gene_number: int) -> str:
        if gene_number is None or gene_number == -1:
            return None
        return self.genes[gene_number].name

    def get_synonymous_proportion(self, gene_num: int) -> float:
        if gene_num is None:
            return self.overall_synonymous_proportion
        else:
            return self.synonymous_proportion[gene_num]

    def plot_fitness_combinations(self):
        """
        The combinations of multiple mutations can be complicated, especially if epistatic relationships are defined.
        This will plot the average fitness of all fitness combinations of all genes defined as a visual check that
        it is as intended.
        Assumes that the background fitness (first column of the fitness array) is 1.
        """
        if not self.multi_gene_array and self.combine_mutations == MutationCombination.REPLACE:
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
                        gene_fitness = self.mutation_distributions[j].get_mean()
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

            return new_fitnesses



