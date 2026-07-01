"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
import pytest

from clone_competition_simulation import (Moran, Parameters,
                                          PopulationParameters, TimeParameters,
                                          pickle_load)


def test_storing():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(100))
    )
    s = p.get_simulator()
    s.run_sim()

    s.pickle_dump('sim.pickle.gz')

    s2 = pickle_load('sim.pickle.gz')
    s2.muller_plot(figsize=(4, 4))


def test_storing1():

    class CustomBreakingSim(Moran): 

        def __init__(self, parameters, last_step):
            super().__init__(parameters)
            self.last_step = last_step

        def get_dividing_cell(self, current_data):
            if self.i == self.last_step:
                raise TimeoutError("Stop simulation")
            return super().get_dividing_cell(current_data)
            
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(
            max_time=100,
            division_rate=1,
            samples=10,  
        ),
        population=PopulationParameters(initial_size_array=np.ones(5000)), 
        tmp_store='tmp_store.pickle.gz'
    )

    np.random.seed(0)
    s = CustomBreakingSim(p, 250000)
    with pytest.raises(TimeoutError) as exc_info:
        s.run_sim()

    s_continued = pickle_load('tmp_store.pickle.gz')
    s_continued1 = pickle_load('tmp_store.pickle.gz1')

    s_continued.i
    s_continued1.i
    
    s_continued1.last_step = -1
    s_continued1.continue_sim()

    np.random.seed(0)
    s2 = CustomBreakingSim(p, -1)
    s2.run_sim()

    np.testing.assert_array_equal(s2.population_array.toarray(), 
                                  s_continued1.population_array.toarray())
    
    np.testing.assert_array_equal(s2.clones_array, 
                                  s_continued1.clones_array)
