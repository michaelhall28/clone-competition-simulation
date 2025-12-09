import pytest
import numpy as np

from clone_competition_simulation.analysis.analysis import (
    mean_clone_size,
    mean_clone_size_fit,
    surviving_clones_fit,
    incomplete_moment,
    incomplete_moment_sem,
    incomplete_moment_vaf_fixed_intervals,
    _get_fitting_section,
    fit_straight_line_to_incomplete_moment,
)

@pytest.fixture
def clone_size_dist():
    return np.arange(10, 0, -1)


def test_mean_clone_size_fit():
    np.testing.assert_almost_equal(
        mean_clone_size_fit(np.arange(4), 0.2),
        np.array([1, 1.2, 1.4, 1.6])
    )


def test_surviving_clones_fit():
    np.testing.assert_almost_equal(
        surviving_clones_fit(np.arange(4), 0.2, 100),
        np.array([100, 100/1.2, 100/1.4, 100/1.6])
    )


def test_mean_clone_size(clone_size_dist):
    np.testing.assert_almost_equal(mean_clone_size(clone_size_dist), 3.6666666666666)


def test_incomplete_moment(clone_size_dist):
    np.testing.assert_almost_equal(
        incomplete_moment(clone_size_dist),
        np.array([
            1., 1.,0.94545455, 0.84848485, 0.72121212, 0.57575758,
            0.42424242, 0.27878788, 0.15151515, 0.05454545
        ])
    )


def test_incomplete_moment2():
    clone_size_dist = np.zeros(10)
    assert incomplete_moment(clone_size_dist) == None


def test_incomplete_moment_sem(clone_size_dist):
    np.testing.assert_almost_equal(
        incomplete_moment_sem(clone_size_dist),
        np.array([
            0., 0., 0.0094432, 0.0257117, 0.0453688,
            0.0642792,0.0780262, 0.0824715, 0.0743418, 0.0515702
        ])
    )


def test_incomplete_moment_vaf_fixed_intervals():
    vafs = np.array([0.2, 0.4, 0.5, 0.1])
    interval = 0.2

    x, im =incomplete_moment_vaf_fixed_intervals(vafs, interval)
    np.testing.assert_almost_equal(
        x, np.array([0., 0.2])
    )
    np.testing.assert_almost_equal(
        im, np.array([1., 0.91666667])
    )


def test_incomplete_moment_vaf_fixed_intervals2():
    vafs = np.array([0.2, 0.4, 0.5, 0.1])
    interval = 1

    x, im =incomplete_moment_vaf_fixed_intervals(vafs, interval)
    np.testing.assert_almost_equal(
        x, np.array([])
    )
    np.testing.assert_almost_equal(
        im, np.array([])
    )


def test_incomplete_moment_vaf_fixed_intervals3():
    vafs = np.array([0.2, 0.3, 0.2, 0.3])
    interval = 0.2

    x, im =incomplete_moment_vaf_fixed_intervals(vafs, interval)
    np.testing.assert_almost_equal(
        x, np.array([])
    )
    np.testing.assert_almost_equal(
        im, np.array([])
    )



def test_fit_straight_line_to_incomplete_moment():
    incom = np.array([
        1., 1.,0.94545455, 0.84848485, 0.72121212, 0.57575758,
        0.42424242, 0.27878788, 0.15151515, 0.05454545
    ])
    slope, one_intercept, r_squared_value = fit_straight_line_to_incomplete_moment(incom, fix_intercept=True)

    np.testing.assert_almost_equal(slope, -0.2106442)
    np.testing.assert_almost_equal(one_intercept, 0)
    np.testing.assert_almost_equal(r_squared_value, 0.7394017760709917)


def test_fit_straight_line_to_incomplete_moment2():
    incom = np.array([
        1., 1.,0.94545455, 0.84848485, 0.72121212, 0.57575758,
        0.42424242, 0.27878788, 0.15151515, 0.05454545
    ])
    slope, one_intercept, r_squared_value = fit_straight_line_to_incomplete_moment(incom, fix_intercept=False)

    np.testing.assert_almost_equal(slope, -0.2896893191388264)
    np.testing.assert_almost_equal(one_intercept, 0.5006191338651974)
    np.testing.assert_almost_equal(r_squared_value, 0.8259501470031084)


def test_get_fitting_section(clone_size_dist):
    assert _get_fitting_section(clone_size_dist, 0.5) == 4

