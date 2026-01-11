import numpy as np
from numpy.typing import NDArray
from scipy.stats import linregress
import matplotlib.pyplot as plt
from loguru import logger


def mean_clone_size_fit(times: NDArray, rlam: float) -> NDArray:
    """For the single progenitor models (for progenitor cells only)"""
    return 1+rlam*times


def surviving_clones_fit(times: NDArray, rlam: float, start_clones: int) -> NDArray:
    """For the single progenitor models (for progenitor cells only)"""
    return start_clones/mean_clone_size_fit(times, rlam)


def mean_clone_size(clone_size_dist: NDArray) -> float:
    # Mean of surviving clones from a clone size frequency array
    """Gets the mean of clones > 1 cell. For dists that start at 0 cell clones"""
    return (clone_size_dist[1:] * np.arange(1, len(clone_size_dist))).sum() / clone_size_dist[1:].sum()


# Incomplete moment functions
def incomplete_moment(clone_size_dist: NDArray) -> NDArray | None:
    # Assuming clone_size_dist starts from zero
    if clone_size_dist[1:].sum() == 0:
        return None
    mcs = mean_clone_size(clone_size_dist)
    total_living_clones = clone_size_dist[1:].sum()
    proportions = clone_size_dist / total_living_clones
    sum_terms = proportions * np.arange(len(proportions))
    moments = np.cumsum(sum_terms[::-1])[::-1]
    return moments / mcs


def incomplete_moment_sem(clone_size_dist: NDArray) -> NDArray:
    sems = []
    s1 = (np.arange(len(clone_size_dist))*clone_size_dist).sum()
    s2 = 0
    s3 = 0
    for i, v in zip(reversed(range(len(clone_size_dist))), reversed(clone_size_dist)):
        s2 += v*i
        s3 += v*i**2
        sem = ((s1-s2)/s1**2)*np.sqrt(s3)
        sems.append(sem)
    return sems[::-1]


def incomplete_moment_vaf_fixed_intervals(vafs: NDArray, interval: float) -> tuple[NDArray, NDArray]:
    vafs = np.flip(np.sort(vafs), axis=0)
    mean_clone_size = vafs.mean()

    x = np.arange(int(vafs.min() / interval) * interval, round(vafs.max() / interval) * interval,
                  interval)

    if len(x) == 0:  # No variation in clone sizes.
        return np.array([]), np.array([])

    x_idx = -1
    last_x = x[x_idx]
    y = []
    incom = 0
    for v in vafs:
        while v < last_x:
            y.append(incom)
            x_idx -= 1
            last_x = x[x_idx]
        incom += v
    y.append(incom)

    return x, np.flip(np.array(y), axis=0) / mean_clone_size / len(vafs)


def fit_straight_line_to_incomplete_moment(incom: NDArray, fix_intercept: bool=True) -> tuple[float, float, float]:
    """The intercept we refer to here is when x=min_clone_size since this is the point we want to fix
    We therefore will shift over the values by one to fit, then shift back to plot

    incom will already be from clone size min_clone_size as the first entry
    """
    log_incom = np.log(incom)
    if fix_intercept:  # Fix the intercept to (min_clone_size, 1)
        x = range(len(log_incom))
        B = np.vstack([x]).T
        slope, resids = np.linalg.lstsq(B, log_incom)[0:2]
        one_intercept = 0  # =log(1)
        r_squared_value = 1 - resids[0] / (len(log_incom) * log_incom.var())
    else:  # No fixed intercept at all
        x = range(len(log_incom))
        slope, one_intercept, r_value, p_value, std_err = linregress(x, log_incom)
        r_squared_value = r_value ** 2
    return slope, one_intercept, r_squared_value


def _get_fitting_section(csd: NDArray, fit_prop: float) -> int:
    """Find the clone size for which the cumulative total contains fit_prop of the total clones"""
    csd[0] = 0  # Ensure we don't count clones of size zero
    norm_csd = csd/csd.sum()
    cumprop = np.cumsum(norm_csd)
    ind = np.nonzero(cumprop > fit_prop)[0][0]+1  # The index after the last one we want
    logger.debug('fitting to first', ind, 'values out of', len(csd))
    return ind


def add_incom_to_plot(incom, clone_size_dist, sem=False, show_fit=False, fit_prop=1, min_size=1,
                      label='InMo', errorevery=1, fit_style='m--', ax=None):
    """Add an incomplete moment trace to a plot. Can also add the SEM or a fit to the trace."""
    if ax is None:
        fig, ax = plt.subplots()
    plot_incom = incom[min_size:]  # We don't plot for clone size 0
    plot_csd = clone_size_dist
    plot_csd[:min_size] = 0
    plot_x = np.arange(min_size, len(incom))
    if sem:
        yerr = incomplete_moment_sem(clone_size_dist)[min_size:]
        ax.errorbar(plot_x, plot_incom, yerr=yerr, label='{0}/SEM'.format(label), errorevery=errorevery)
    else:
        ax.plot(plot_x, plot_incom, label=label)

    if show_fit:
        if fit_prop < 1:
            fit_section = _get_fitting_section(plot_csd, fit_prop)
            fit_data = plot_incom[:fit_section]
        else:
            fit_data = plot_incom

        slope, intercept, r_squared_value = fit_straight_line_to_incomplete_moment(fit_data)
        y = np.exp(slope * (
        plot_x - min_size) + intercept)  # Take exp because was fit to the log. With intercept a clone size 1
        ax.plot(plot_x, y, fit_style, label='Straight line fit: r^2={0}'.format(r_squared_value))
