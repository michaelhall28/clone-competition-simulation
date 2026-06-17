from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import linregress


def mean_clone_size_fit(times: NDArray, rlam: float) -> NDArray:
    """Compute the mean clone size (progenitor cells only) over time for single progenitor models.

    Parameters
    ----------
    times : NDArray
        Array of time points
    rlam : float
        The slope of the line. r * lambda for single progenitor models.

    Returns
    -------
    NDArray
        Mean clone size at each time point.
    """
    return 1+rlam*times


def surviving_clones_fit(times: NDArray, rlam: float, start_clones: int) -> NDArray:
    """Compute the expected number of surviving clones over time for single progenitor models.

    Parameters
    ----------
    times : NDArray
        Array of time points for progenitor clone dynamics.
    rlam : float
        The slope of the line. r * lambda for single progenitor models.
    start_clones : int
        Number of clones present at the initial time point.

    Returns
    -------
    NDArray
        Expected number of surviving clones at each time point.
    """
    return start_clones / mean_clone_size_fit(times, rlam)


def mean_clone_size(clone_size_dist: NDArray) -> float:
    """Compute the mean size of surviving clones from a clone size distribution.

    Parameters
    ----------
    clone_size_dist : NDArray
        Frequency array of clone counts indexed by clone size, where index i corresponds
        to the number of clones of size i.

    Returns
    -------
    float
        Mean clone size among surviving clones (size >= 1).
    """
    return (clone_size_dist[1:] * np.arange(1, len(clone_size_dist))).sum() / clone_size_dist[1:].sum()


# Incomplete moment functions
def incomplete_moment(clone_size_dist: NDArray) -> NDArray:
    """Compute the incomplete moment values for a clone size distribution.

    The incomplete moment is computed from the clone size distribution starting at size zero.
    The returned values are normalized by the mean surviving clone size.

    Parameters
    ----------
    clone_size_dist : NDArray
        Frequency distribution of clones indexed by clone size.

    Returns
    -------
    NDArray
        Incomplete moment values for each clone size, or an empty array if there are no surviving clones.
    """
    if clone_size_dist[1:].sum() == 0:
        return np.array([])  # No surviving clones, so no incomplete moment can be computed
    mcs = mean_clone_size(clone_size_dist)
    total_living_clones = clone_size_dist[1:].sum()
    proportions = clone_size_dist / total_living_clones
    sum_terms = proportions * np.arange(len(proportions))
    moments = np.cumsum(sum_terms[::-1])[::-1]
    return moments / mcs


def incomplete_moment_sem(clone_size_dist: NDArray) -> NDArray:
    """Compute the standard error of the incomplete moment for each clone size.

    Parameters
    ----------
    clone_size_dist : NDArray
        Frequency distribution of clones indexed by clone size.

    Returns
    -------
    NDArray
        Standard error values corresponding to the incomplete moment at each clone size.
    """
    sems = []
    s1 = (np.arange(len(clone_size_dist)) * clone_size_dist).sum()
    s2 = 0
    s3 = 0
    for i, v in zip(reversed(range(len(clone_size_dist))), reversed(clone_size_dist)):
        s2 += v * i
        s3 += v * i**2
        sem = ((s1 - s2) / s1**2) * np.sqrt(s3)
        sems.append(sem)
    return np.array(sems[::-1])


def incomplete_moment_vaf_fixed_intervals(vafs: NDArray, interval: float) -> tuple[NDArray, NDArray]:
    """Compute incomplete moment values for VAFs using fixed intervals.

    Parameters
    ----------
    vafs : NDArray
        Array of VAFs for clones.
    interval : float
        Fixed interval size used to aggregate the VAF values.

    Returns
    -------
    tuple of NDArray
        x : Interval boundary values.
        y : Incomplete moment values for each interval.
    """
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
    """Fit a straight line to the incomplete moment in log space.

    Parameters
    ----------
    incom : NDArray
        Incomplete moment values starting at the minimum clone size.
    fix_intercept : bool, optional
        If True, fit with a fixed intercept of log(1) = 0. If False, perform an ordinary
        linear regression where the intercept is allowed to vary. Default is True.

    Returns
    -------
    tuple of float
        slope : Fitted slope in log space.
        intercept : Fitted intercept in log space.
        r_squared_value : Coefficient of determination for the fitted line.
    """
    log_incom = np.log(incom)
    if fix_intercept:  # Fix the intercept to (min_clone_size, 1)
        x = np.arange(len(log_incom), dtype=float)
        B = x[:, np.newaxis]
        slope = float(np.linalg.lstsq(B, log_incom, rcond=None)[0][0])
        one_intercept = 0.0  # =log(1)
        resids = np.linalg.lstsq(B, log_incom, rcond=None)[1]
        r_squared_value = 1 - resids[0] / (len(log_incom) * log_incom.var())
    else:  # No fixed intercept at all
        x = np.arange(len(log_incom), dtype=float)
        slope, one_intercept, r_value, p_value, std_err = linregress(x, log_incom)
        r_squared_value = float(r_value) ** 2
    return float(slope), float(one_intercept), float(r_squared_value)


def _get_fitting_section(csd: NDArray, fit_prop: float) -> int:
    """Determine the index range to use for fitting based on cumulative clone proportion.

    Parameters
    ----------
    csd : NDArray
        Clone size distribution as a frequency array, indexed by clone size.
    fit_prop : float
        Proportion of total clones to include in the fitting section.

    Returns
    -------
    int
        Index after the last clone size included in the fitting section.
    """
    csd = csd.copy()  # Avoid modifying the original array
    csd[0] = 0  # Ensure we don't count clones of size zero
    norm_csd = csd/csd.sum()
    cumprop = np.cumsum(norm_csd)
    ind = np.nonzero(cumprop > fit_prop)[0][0]+1  # The index after the last one we want
    logger.debug('fitting to first', ind, 'values out of', len(csd))
    return ind


def add_incom_to_plot(incom: NDArray, clone_size_dist: NDArray, sem=False, show_fit=False, 
                      fit_prop=1, min_size=1, label='InMo', errorevery=1, 
                      fit_style='m--', ax=None) -> Axes:
    """Add an incomplete moment trace to a Matplotlib axis.

    Parameters
    ----------
    incom : NDArray
        Incomplete moment values indexed by clone size.
    clone_size_dist : NDArray
        Clone size frequency distribution indexed by clone size.
    sem : bool, optional
        If True, draw standard error bars for the incomplete moment. Default is False.
    show_fit : bool, optional
        If True, add a straight line fit to the incomplete moment trace. Default is False.
    fit_prop : float, optional
        Proportion of clone sizes to use for fit calculation when ``show_fit`` is True.
        Default is 1.
    min_size : int, optional
        Minimum clone size to include in the plot. Default is 1.
    label : str, optional
        Label for the plot trace. Default is 'InMo'.
    errorevery : int, optional
        Frequency of error bars when ``sem`` is True. Default is 1.
    fit_style : str, optional
        Matplotlib line style used for the fitted trace. Default is 'm--'.
    ax : Axes or None, optional
        Axis to draw on. If None, a new figure and axis are created.

    Returns
    -------
    Axes
        Axis containing the incomplete moment plot.
    """
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

    return ax
