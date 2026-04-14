import numpy as np
from src.clone_competition_simulation import (
    Parameters, 
    TimeParameters, 
    PopulationParameters,
    Moran, 
    Moran2D
)
from src.clone_competition_simulation.simulation_algorithms.general_sim_class import NonSpatialCurrentData
from src.clone_competition_simulation.simulation_algorithms.general_2D_class import SpatialCurrentData


def test_custom_moran():
    class MyCustomMoran(Moran):
        """Always take from the lowest id clone and add to the highest"""
        def get_dividing_cell(self, current_data: NonSpatialCurrentData) -> int:
            return len(current_data.current_population) - 1
        
        def get_differentiating_cell(self, current_data: NonSpatialCurrentData) -> int:
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


def test_custom_moran2d():
    
    class MyCustomMoran2D(Moran2D):
        """Cells die in grid position order, and the highest number clone will divide"""

        def __init__(self, parameters: Parameters):
            super().__init__(parameters)

        def get_differentiating_cell(self, i: int, current_data: SpatialCurrentData) -> int:
            """
            If the dying cell depends on the current state of the simulation, this function needs to be overwritten
            Should return coordinate of the dying cell.
            """
            coord = i % self.total_pop  # The dying cell goes in order across the grid
        

            return coord

        def get_dividing_cell(self, coord: int, current_data: SpatialCurrentData) -> int:
            """
            This function determines which clone will divide.  

            For this custom class, it will return the highest clone id in the neighbourhood.

            This should return the clone_id of the clone that will divide.  
            """
            # get the clone ids in the neighbourhood using the grid array and the neighbour map
            neighbour_clones = current_data.grid_array[self.neighbour_map[coord]]

            # Select the highest clone in the neighbourhood
            return neighbour_clones.max()
        
    
    params = Parameters(
        algorithm="Moran2D", 
        times=TimeParameters(max_time=1, division_rate=1, samples=4), 
        population=PopulationParameters(initial_grid=np.arange(4).reshape(2, 2), cell_in_own_neighbourhood=False)
    )

    sim = MyCustomMoran2D(parameters=params)

    sim.run_sim()

    # Assert that the highest clone grows and grows
    np.testing.assert_equal(sim.population_array.toarray()[-1], np.array([1, 2, 3, 4, 4]))
        