import os

import pytest
import numpy as np
from pydantic import ValidationError

from src.clone_competition_simulation.parameters import Parameters, Algorithm

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_validation_failure1():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters()

    assert "algorithm\n  Input should be 'WF', 'WF2D', 'Moran', 'Moran2D' or 'Branching'" in str(exc_info)
    assert 'tag' not in str(exc_info)


def test_validation_no_config():
    p = Parameters(algorithm=Algorithm.BRANCHING, population={'initial_cells': 100},
                   times={"division_rate": 1, "max_time": 10})
    assert p.algorithm == Algorithm.BRANCHING
    assert p.times.division_rate == 1
    assert p.times.max_time == 10


def test_validation_from_config1():
    p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config.yml"))
    assert p.algorithm == Algorithm.MORAN
    assert p.times.division_rate == 1
    assert p.times.max_time == 10
    assert p.times.samples == 100
    assert p.population.initial_cells == 100
    np.testing.assert_array_equal(p.fitness.fitness_array, [1])
    np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])


def test_validation_from_config2():
    p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config.yml"),
                   algorithm=Algorithm.BRANCHING, population={'initial_cells': 1000})
    assert p.algorithm == Algorithm.BRANCHING
    assert p.times.division_rate == 1
    assert p.times.max_time == 10
    assert p.times.samples == 100
    assert p.population.initial_cells == 1000
    np.testing.assert_array_equal(p.fitness.fitness_array, [1])
    np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])


def test_validation_from_config3():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config2.yml"))

    assert "config_file_settings.population.initial_cells\n  Input should be a valid integer" in str(exc_info)


def test_validation_from_config4():
    with pytest.raises(ValidationError) as exc_info:
        p = Parameters(run_config_file=os.path.join(CURRENT_DIR, "test_run_config3.yml"))

    assert "config_file_settings.algorithm\n  Input should be 'WF', 'WF2D', 'Moran', 'Moran2D'" in str(exc_info)


def test_validation_from_config5(monkeypatch):
    with monkeypatch.context() as m:
        monkeypatch.setenv('CCS_RUN_CONFIG', os.path.join(CURRENT_DIR, "test_run_config.yml"))

        p = Parameters()
        assert p.algorithm == Algorithm.MORAN
        assert p.times.division_rate == 1
        assert p.times.max_time == 10
        assert p.times.samples == 100
        assert p.population.initial_cells == 100
        np.testing.assert_array_equal(p.fitness.fitness_array, [1])
        np.testing.assert_array_equal(p.fitness.mutation_rates, [[0, 0]])
        del os.environ["CCS_RUN_CONFIG"]