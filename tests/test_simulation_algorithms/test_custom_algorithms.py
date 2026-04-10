import numpy as np
from clone_competition_simulation import Parameters, Moran, TimeParameters, PopulationParameters
from clone_competition_simulation.simulation_algorithms.general_sim_class import CurrentData

def test_custom_moran():
    class MyCustomMoran(Moran):
        """Always take from the lowest id clone and add to the highest"""
        def get_dividing_clone(self, current_data: CurrentData) -> int:
            return len(current_data.current_population) - 1
        
        def get_differentiating_clone(self, current_data: CurrentData) -> int:
            return 0
        
    np.random.seed(0)
    params = Parameters(
        algorithm="Moran", 
        times=TimeParameters(max_time=1, division_rate=1, samples=6), 
        population=PopulationParameters(initial_size_array=np.ones(5, dtype=int))
    )
    sim = MyCustomMoran(parameters=params)
    sim.run_sim()

    # Assert that the highest clone grows and grows
    np.testing.assert_equal(sim.population_array.toarray()[-1], np.array([1, 2, 3, 4, 5, 5]))
        