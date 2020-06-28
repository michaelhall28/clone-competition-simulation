# cython: language_level=3
cimport cython
from cython_gsl cimport *

import numpy as np
from numpy cimport *


cdef gsl_rng *RR = gsl_rng_alloc(gsl_rng_mt19937)  # This is a global random number generator


cpdef set_random_seed(int s):
    gsl_rng_set (RR, s)


cdef int single_gillespie_cy(int a, int b, double time_step,
                             double asym_div_rate, double gamma) except -1:
    """
    a or b must be > 0. Not checking here to make function faster for the Moran2D algorithm.
    Check is done in the functions that call this one.
    a = number of progenitor cells
    b = number of differentiated cells
    """
    cdef double t = 0
    cdef double b_born_rate = a * asym_div_rate
    cdef double b_dead_rate, total_rate, selector
    while t < time_step:
        # Term for dying differentiated cells
        b_dead_rate = b * gamma
        total_rate = b_dead_rate + b_born_rate
        t += gsl_ran_exponential(RR, 1/total_rate)
        if t < time_step:
            selector = gsl_rng_uniform(RR)
            if selector < b_dead_rate / total_rate:  # We kill a B cell
                b -= 1
                if b == 0 and a == 0:
                    break
            else:  # We add a new B cell
                b += 1

    return b


cpdef int single_gillespie_cy_with_check(int a, int b, double time_step,
                             double asym_div_rate, double gamma) except -1:
    if a == 0 and b == 0:
        return 0
    return single_gillespie_cy(a, b, time_step, asym_div_rate, gamma)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef ndarray bcell_cy(long[:] current_population,
                       long[:] current_diff_cell_population,
                       double time_step, double asym_div_rate, double gamma):
    """
    Add differentated cells born through asymmetric division and remove diff cells that have stratified
    Goes through a Gillespie algorithm for the time between the last progenitor cell division and the next.
    Symmetric divisions are dealt with in self.sim_step since they involve a change in progenitor cell numbers.

    For a small value of r, there are likely to be many events here before the next progenitor cell division.
    Therefore cannot assume that all differentiated cells born in this function will survive until the end.

    Despite the exponential stratification times, cannot generate all the differentiated events and assign at the end.
    The differentiated cells born before the start have more chances to die than those born at the end.

    Loop through each clone and run gillespie on it.
    """
    cdef int i = 0
    cdef int a, b
    cdef ndarray[long, ndim=1] new_b_cells = np.empty_like(current_diff_cell_population, dtype=int)

    for i in range(current_population.shape[0]):
        a, b = current_population[i], current_diff_cell_population[i]
        if a == 0 and b == 0:
            new_b_cells[i] = 0
        else:
            new_b_cells[i] = single_gillespie_cy(a, b, time_step, asym_div_rate, gamma)

    return new_b_cells