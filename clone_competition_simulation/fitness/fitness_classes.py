"""
Functions/classes to randomly draw and/or calculate the fitness of clones
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import csv


# Probability distributions for drawing the fitness of new mutations.
# Set up so can be called like functions without argument, but can print the attributes
class NormalDist(object):
    """
    A wrapper for numpy.random.normal
    Will draw again if the value is below zero.
    """
    def __init__(self, var, mean=1.):
        self.var = var
        self.mean = mean

    def __str__(self):
        return 'Normal distribution(mean {0}, variance {1})'.format(self.mean, self.var)

    def __call__(self):
        g = np.random.normal(self.mean, self.var)
        if g < 0:
            # print('growth rate below zero! Redrawing a new rate')
            return self()
        return g

    def get_mean(self):
        return self.mean


class FixedValue(object):
    """
    Just returns the fixed value given.
    This is a wrapper for that number so that it functions like the random distributions.
    """
    def __init__(self, value):
        self.mean = value

    def __str__(self):
        return 'Fixed value {0}'.format(self.mean)

    def __call__(self):
        return self.mean

    def get_mean(self):
        return self.mean


class ExponentialDist(object):
    """
    An exponential distribution.
    A wrapper for numpy.random.exponential

    The parameters are mean and offset.
    The mean defines the mean of the distribution after the offset has been applied,
    i.e. the random number is
    offset +  np.random.exponential(mean - offset)

    mean must be greater than the offset.
    """
    def __init__(self, mean, offset=1):
        if mean <= offset:
            raise ValueError('mean must be less than offset')
        self.mean = mean
        self.offset = offset  # Offset of 1 means the mutations will start from neutral.

    def __str__(self):
        return 'Exponential distribution(mean {0}, offset {1})'.format(self.mean, self.offset)

    def __call__(self):
        g = self.offset + np.random.exponential(self.mean - self.offset)
        return g

    def get_mean(self):
        return self.mean


class UniformDist(object):
    """
    A uniform distribution between low and high
    A wrapper for numpy.random.uniform
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __str__(self):
        return 'Uniform distribution(low {0}, high {1})'.format(self.low, self.high)

    def __call__(self):
        g = np.random.uniform(self.low, self.high)
        return g

    def get_mean(self):
        return (self.high + self.low)/2


##################
# Classes for diminishing returns or other transformations of the raw fitness
class UnboundedFitness:
    """
    No bound or transformation on the fitness.
    """

    def __str__(self):
        return 'UnboundedFitness'

    def fitness(self, x):
        return x

    def inverse(self, x):
        return x


class BoundedLogisticFitness:
    """
    The effect of new (beneficial) mutations tails off as the clone gets stronger.
    There is a maximum fitness.
    """
    def __init__(self, a, b=math.exp(1)):
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

    def fitness(self, x):
        return self.a / (1 + self.c * (self.b ** (-x)))

    def inverse(self, y):
        return math.log(self.c / (self.a / y - 1), self.b)


################
# Class for storing information about a gene
class Gene(object):
    """Determines the mutations which are created for a gene"""

    def __init__(self, name, mutation_distribution, synonymous_proportion, weight=1.):
        """

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
        if weight < 0:
            raise ValueError('weight cannot be below zero')
        self.name = name
        self.mutation_distribution = mutation_distribution
        self.synonymous_proportion = synonymous_proportion
        self.weight = weight

    def __str__(self):
        return "Gene. Name: {0}, MutDist: {1}, SynProp: {2}, Weight: {3}".format(self.name,
                                                                                 self.mutation_distribution.__str__(),
                                                                                 self.synonymous_proportion,
                                                                                 self.weight)


##################
# Class to put it all together

class MutationGenerator(object):
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
    # Options for combining mutations in the same gene or when multi_gene_array=False
    combine_options = ('multiply', 'add', 'replace', 'max', 'min')

    # Options for combining mutations in different genes or epistatic effects.
    combine_array_options = ('multiply', 'add', 'max', 'min', 'priority')

    def __init__(self, combine_mutations='multiply', multi_gene_array=False, combine_array='multiply',
                 genes=(Gene('all', NormalDist(1.1), synonymous_proportion=0.5, weight=1),),
                 mutation_combination_class=UnboundedFitness(), epistatics=None):
        if combine_mutations not in self.combine_options:
            raise ValueError(
                "'{0}' is not a valid option for 'combine_mutations'. Pick from {1}".format(combine_mutations,
                                                                                           self.combine_options))
        if combine_array not in self.combine_array_options:
            raise ValueError("'{0}' is not a valid option for 'combine_array'. Pick from {1}".format(combine_array,
                                                                                            self.combine_array_options))

        self.combine_mutations = combine_mutations
        self.multi_gene_array = multi_gene_array
        self.combine_array = combine_array
        self.genes = genes
        self.num_genes = len(self.genes)
        self.gene_indices = {g.name: i for i, g in enumerate(genes)}
        self.mutation_distributions = [g.mutation_distribution for g in genes]
        self.gene_weights = [g.weight for g in genes]
        self.synonymous_proportion = np.array([g.synonymous_proportion for g in genes])
        self.overall_synonymous_proportion = np.array(
            [g.synonymous_proportion * g.weight for g in genes]).sum() / np.array(self.gene_weights).sum()

        self.relative_weights = np.array(self.gene_weights) / sum(self.gene_weights)
        self.relative_weights_cumsum = self.relative_weights.cumsum()
        self.mutation_combination_class = mutation_combination_class  # E.g. BoundedLogisticFitness above
        if epistatics is not None:
            if not multi_gene_array:
                print('Using multi_gene_array because there are epistatic relationships')
                self.multi_gene_array = True
            # List of epistatic relationships
            # Each item in the list is (name, gene_name1, gene_name2, ..., epistatic fitness distribution)
            # Add the names to the gene list
            self.gene_indices.update({e[0]: j+self.num_genes for j, e in enumerate(epistatics)})
            # Convert to the gene indices
            self.epistatics = [tuple([self.get_gene_number(ee) for ee in e[1:-1]] + [e[-1]]) for e in epistatics]
            self.epistatic_cols = range(len(genes) + 1, len(genes) + len(self.epistatics) + 1)
        else:
            self.epistatics = None
        self.params = {
            'combine_mutations': combine_mutations,
            'genes': [g.__str__() for g in self.genes],
            'fitness_class': mutation_combination_class.__str__(),
        }

    def __str__(self):
        s = "<MutGen: comb_muts={0}, genes={1}, fitness_class={2}>".format(self.params['combine_mutations'],
                                                                           self.params['genes'],
                                                                           self.params['fitness_class'])
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

    def _are_synonymous(self, mut_types):
        """
        Determines whether the new mutations are synonymous
        :param mut_types:
        :return:
        """
        return np.random.binomial(1, self.synonymous_proportion[mut_types])

    def _get_genes(self, num):
        """
        Determines which genes are mutated
        :param num:
        :return:
        """
        r = np.random.rand(num, 1)
        k = (self.relative_weights_cumsum < r).sum(axis=1)
        return k

    def _update_fitness_arrays(self, old_mutation_arrays, genes_mutated, syns):
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

    def _combine_fitnesses(self, old_fitnesses, new_mutation_fitnesses):
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
        if self.combine_mutations == 'multiply':
            combined_fitness = old_fitnesses * new_mutation_fitnesses
        elif self.combine_mutations == 'add':
            combined_fitness = old_fitnesses + new_mutation_fitnesses - 1
            combined_fitness[combined_fitness < 0] = 0
        elif self.combine_mutations == 'replace':
            combined_fitness = new_mutation_fitnesses
        elif self.combine_mutations == 'max':
            combined_fitness = np.maximum(new_mutation_fitnesses, old_fitnesses)
        elif self.combine_mutations == 'min':
            combined_fitness = np.minimum(new_mutation_fitnesses, old_fitnesses)
        else:
            raise NotImplementedError("Have tried to use {}".format(self.combine_mutations))
        return combined_fitness

    def _epistatic_combinations(self, fitness_arrays):
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
        for i, epi in enumerate(self.epistatics):
            epi_genes, dfe = epi[:-1], epi[-1]
            matching_rows = np.all(non_nan[:, tuple([g+1 for g in epi_genes])], axis=1)  # +1 because of the WT column
            new_matching_rows = matching_rows * not_already_epi_rows[:, i]
            new_draws = [dfe() for j in new_matching_rows if j]
            epi_rows[new_matching_rows, i] = new_draws
            for g in epi_genes:
                row_positions_to_blank.extend(np.where(matching_rows)[0])
                col_positions_to_blank.extend([g + 1] * matching_rows.sum())

        fitness_array = np.concatenate([raw_gene_arr, epi_rows], axis=1)
        epistatic_fitness_array = fitness_array.copy()
        epistatic_fitness_array[row_positions_to_blank, col_positions_to_blank] = np.nan
        return fitness_array, epistatic_fitness_array

    def combine_vectors(self, fitness_arrays):
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

        if not self.multi_gene_array:  # Don't have to combine genes, just reduce to 1D array
            combined_fitness = fitness_arrays[:, 0]
        elif self.combine_array == 'multiply':
            combined_fitness = np.nanprod(fitness_arrays, axis=1)
        elif self.combine_array == 'add':
            combined_fitness = np.nansum(fitness_arrays, axis=1) - np.count_nonzero(~np.isnan(fitness_arrays),
                                                                                    axis=1) + 1
            combined_fitness[combined_fitness < 0] = 0
        elif self.combine_array == 'max':
            combined_fitness = np.nanmax(fitness_arrays, axis=1)
        elif self.combine_array == 'min':
            combined_fitness = np.nanmin(fitness_arrays, axis=1)
        elif self.combine_array == 'priority':
            # Find the right-most non-nan value. Useful for epistatic interactions that are superseded by another
            # To find the last non-nan columns, reverse the column order and find the first non-zero entry.
            fitness_arrays = fitness_arrays[:, ::-1]
            c = np.isnan(fitness_arrays)
            d = np.argmin(c, axis=1)
            combined_fitness = fitness_arrays[range(len(fitness_arrays)), d]
        else:
            raise NotImplementedError("Have tried to use {}".format(self.combine_array))
        return self.mutation_combination_class.fitness(combined_fitness), full_fitness_arrays

    def get_gene_number(self, gene_name):
        if gene_name is None:
            return None
        return self.gene_indices[gene_name]

    def get_gene_name(self, gene_number):
        if gene_number is None or gene_number == -1:
            return None
        return self.genes[gene_number].name

    def get_synonymous_proportion(self, gene_num):
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
        if not self.multi_gene_array and self.combine_mutations == 'replace':
            # No combinations here. Just need to plot individual genes
            print('No combinations defined. Only most recent non-silent mutation defines fitness')
            xticklabels = ['Background']
            fitness_values = [1]
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
                # Temporarily change the combine_mutations attribute so same combination functions can be used
                if self.combine_mutations in ('add', 'multiply'):
                    print('This allows multiple mutations per gene to have an effect. ' \
                          'Just showing combinations of up to one (mean fitness) mutation from each gene.')
                ca = self.combine_array
                self.combine_array = self.combine_mutations
                self.multi_gene_array = True
                new_fitnesses, new_mutation_arrays = self.combine_vectors(fitness_array)
                self.combine_array = ca
                self.multi_gene_array = False

            plt.bar(range(fitness_array.shape[0]), new_fitnesses)
            plt.ylabel('Fitness')
            plt.xticks(range(fitness_array.shape[0]), xticklabels, rotation=90)

            return new_fitnesses



