import numpy as np

from src.clone_competition_simulation import (
    EndConditionError,
    Parameters,
    TimeParameters,
    PopulationParameters,
)
from src.clone_competition_simulation.simulation_algorithms.stop_conditions import MoranStop


def test_stop_condition(monkeypatch):

    def stop_when_10_clones_left(sim):
        current_clones = sim.population_array[:, sim.plot_idx - 1]

        # Count the number of clones with at least one cell alive
        num_surviving = (current_clones > 0).sum()

        if num_surviving <= 10:
            sim.stop_condition_result = {"My results": num_surviving}
            raise EndConditionError()

    with monkeypatch.context() as m:
        rng = np.random.RandomState()
        monkeypatch.setattr('numpy.random', rng)

        np.random.seed(1)
        p = Parameters(
            algorithm='Moran',
            times=TimeParameters(max_time=20, division_rate=1),
            population=PopulationParameters(initial_size_array=np.ones(100)),
            end_condition_function=stop_when_10_clones_left
        )
        s = p.get_simulator()
        s.run_sim()

        assert isinstance(s, MoranStop)
        assert s.stop_time == 12.8
        assert s.stop_condition_result == {"My results": 10}
