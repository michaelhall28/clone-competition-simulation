"""
Functions for defining how to combine mutations in the same or different genes
"""
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Type for the functions which combine old and new fitness values. 
FitnessCombinationType = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def add_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine fitness values additively.

    The total fitness is computed as ``old + new - 1`` and clipped at zero.

    Parameters
    ----------
    old_fitnesses : NDArray[np.float64]
        Previous fitness values.
    new_mutation_fitnesses : NDArray[np.float64]
        New mutation fitness effects.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values.
    """
    combined_fitness = old_fitnesses + new_mutation_fitnesses - 1
    combined_fitness[combined_fitness < 0] = 0
    return combined_fitness


def multiply_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine fitness values multiplicatively.

    The total fitness is computed as ``old * new``.

    Parameters
    ----------
    old_fitnesses : NDArray[np.float64]
        Previous fitness values.
    new_mutation_fitnesses : NDArray[np.float64]
        New mutation fitness effects.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values.
    """
    return old_fitnesses * new_mutation_fitnesses


def replace_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replace old fitness values with new mutation fitness effects.

    Parameters
    ----------
    old_fitnesses : NDArray[np.float64]
        Previous fitness values. Not used in this function, but included for consistency with other combination functions.
    new_mutation_fitnesses : NDArray[np.float64]
        New mutation fitness effects.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values.
    """
    return new_mutation_fitnesses


def max_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine fitness values by taking the maximum.

    The total fitness is computed as ``max(old, new)``.

    Parameters
    ----------
    old_fitnesses : NDArray[np.float64]
        Previous fitness values.
    new_mutation_fitnesses : NDArray[np.float64]
        New mutation fitness effects.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values.
    """
    return np.maximum(old_fitnesses, new_mutation_fitnesses)


def min_fitness(old_fitnesses: NDArray[np.float64], new_mutation_fitnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine fitness values by taking the minimum.

    The total fitness is computed as ``min(old, new)``.

    Parameters
    ----------
    old_fitnesses : NDArray[np.float64]
        Previous fitness values.
    new_mutation_fitnesses : NDArray[np.float64]
        New mutation fitness effects.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values.
    """
    return np.minimum(old_fitnesses, new_mutation_fitnesses)


# Type for the functions which combine fitness across genes. 
GeneCombinationType = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def add_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine gene fitness arrays by additive aggregation.

    The function sums each row of ``fitness_arrays``, subtracts the number of
    non-missing genes to account for the neutral fitness baseline, and clips the
    result at zero.

    Parameters
    ----------
    fitness_arrays : NDArray[np.float64]
        Array of gene-level fitness effects, where missing values are ``np.nan``.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values for each row.
    """
    combined_fitness = np.nansum(fitness_arrays, axis=1) - np.count_nonzero(~np.isnan(fitness_arrays),
                                                                            axis=1) + 1
    combined_fitness[combined_fitness < 0] = 0
    return combined_fitness


def priority_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine gene fitness arrays by priority.

    Selects the last non-missing value in each row, 
    which is useful for superseding epistatic relationships.

    Parameters
    ----------
    fitness_arrays : NDArray[np.float64]
        Array of gene-level fitness effects, where missing values are ``np.nan``.

    Returns
    -------
    NDArray[np.float64]
        Selected fitness value from each row.
    """
    # Find the right-most non-nan value. Useful for epistatic interactions that are superseded by another
    # To find the last non-nan columns, reverse the column order and find the first non-zero entry.
    fitness_arrays = fitness_arrays[:, ::-1]
    c = np.isnan(fitness_arrays)
    d = np.argmin(c, axis=1)
    combined_fitness = fitness_arrays[range(len(fitness_arrays)), d]
    return combined_fitness


def multiply_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine gene fitness arrays by multiplicative aggregation.

    The function computes the product of each row of ``fitness_arrays``, ignoring
    missing values.

    Parameters
    ----------
    fitness_arrays : NDArray[np.float64]
        Array of gene-level fitness effects, where missing values are ``np.nan``.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values for each row.
    """
    return np.nanprod(fitness_arrays, axis=1)


def max_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine gene fitness arrays by taking the maximum.

    The function computes the maximum of each row of ``fitness_arrays``, ignoring
    missing values.

    Parameters
    ----------
    fitness_arrays : NDArray[np.float64]
        Array of gene-level fitness effects, where missing values are ``np.nan``.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values for each row.
    """
    return np.nanmax(fitness_arrays, axis=1)


def min_array_fitness(fitness_arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine gene fitness arrays by taking the minimum.

    The function computes the minimum of each row of ``fitness_arrays``, ignoring
    missing values.

    Parameters
    ----------
    fitness_arrays : NDArray[np.float64]
        Array of gene-level fitness effects, where missing values are ``np.nan``.

    Returns
    -------
    NDArray[np.float64]
        Combined fitness values for each row.
    """
    return np.nanmin(fitness_arrays, axis=1)
