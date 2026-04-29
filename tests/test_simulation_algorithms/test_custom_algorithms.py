import numpy as np
from src.clone_competition_simulation import (
    Parameters, 
    TimeParameters, 
    PopulationParameters,
    Moran, 
    Moran2D, 
    WF, 
    WF2D, 
    Branching
)
from clone_competition_simulation.simulation_algorithms.base_sim_class import NonSpatialCurrentData
from clone_competition_simulation.simulation_algorithms.base_2D_class import SpatialCurrentData


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


def test_custom_wf():

    class MyCustomWF(WF):
        """Always take from the lowest id clone and add to the highest"""
        def get_next_generation(self, current_data: NonSpatialCurrentData) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
            """This function should return the cell counts for all new clone

            Args:
                current_data (NonSpatialCurrentData): Current clone cell counts (
                 an array of clone sizes and an array of clone ids)

            Returns:
                np.ndarray[tuple[int], np.dtype[np.int_]]: The new clone cell counts (
                for the same clone ids as the current_data input)
            """

            # Kill off the first clone and add the cells to the last
            input_clone_counts = current_data.current_population

            output_clone_counts = input_clone_counts.copy()
            # Take the first surviving clone cells and add to the last clone
            output_clone_counts[-1] += output_clone_counts[0]
            output_clone_counts[0] = 0
            return output_clone_counts
        
    params = Parameters(
        algorithm="WF", 
        times=TimeParameters(max_time=4, division_rate=1, samples=6), 
        population=PopulationParameters(initial_size_array=np.array([1, 2, 3, 4, 1]))
    )
    sim = MyCustomWF(parameters=params)
    sim.run_sim()

    # Assert that the highest clone grows and grows
    expected_population_array = np.array([
        [1, 0, 0, 0, 0], 
        [2, 2, 0, 0, 0], 
        [3, 3, 3, 0, 0], 
        [4, 4, 4, 4, 0], 
        [1, 2, 4, 7, 11]
    ])
    np.testing.assert_equal(sim.population_array.toarray(), expected_population_array)



def test_custom_wf2d():
    
    class MyCustomWF2D(WF2D):
        """All cells just rotate along the grid by one position each step"""

        def get_next_generation(self, current_data: SpatialCurrentData) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
            """
            Move each cell along one position in the grid
            :return:
            """
            current_grid_array = current_data.grid_array

            new_grid_array = np.concat([current_grid_array[-1:], current_grid_array[:-1]])

            return new_grid_array
        
    
    params = Parameters(
        algorithm="WF2D", 
        times=TimeParameters(max_time=3, division_rate=1, samples=4), 
        population=PopulationParameters(initial_grid=np.arange(4).reshape(2, 2), cell_in_own_neighbourhood=False)
    )

    sim = MyCustomWF2D(parameters=params)

    sim.run_sim()

    
    np.testing.assert_equal(sim.grid_results[0], np.array([[0, 1], [2, 3]]))
    np.testing.assert_equal(sim.grid_results[1], np.array([[3, 0], [1, 2]]))
    np.testing.assert_equal(sim.grid_results[2], np.array([[2, 3], [0, 1]]))
    np.testing.assert_equal(sim.grid_results[3], np.array([[1, 2], [3, 0]]))
        

def test_custom_branching():
    
    class MyCustomBranching(Branching):
        """The cell division probability depends on the clone id"""

        def does_cell_divide(self, clone_id: int) -> bool:
            return clone_id > 3
        
    np.random.seed(0)
    params = Parameters(
        algorithm="Branching", 
        times=TimeParameters(max_time=3, division_rate=1, samples=4), 
        population=PopulationParameters(initial_size_array=np.array([5, 4, 3, 2, 1]))
    )

    sim = MyCustomBranching(parameters=params)

    sim.run_sim()

    expected_population_array = np.array([
        [5, 1, 0, 0, 0], 
        [4, 2, 1, 1, 0], 
        [3, 1, 0, 0, 0], 
        [2, 1, 1, 0, 0], 
        [1, 7, 26, 141, 629]
    ])
    np.testing.assert_equal(sim.population_array.toarray(), expected_population_array)


def test_complex_custom_example():

    class CompetitionRules:

        def __init__(self, interaction_effects: dict[tuple[int, int], float]):
            self.interaction_effects = interaction_effects


        @staticmethod
        def apply_interaction_effect(clone_a: int, clone_b: int, multiplier: float,
                                     neighbour_clones: np.ndarray[tuple[int], np.dtype[np.int_]], 
                                     weights: np.ndarray[tuple[int], np.dtype[np.float64]]) \
                                        -> None:
            """If two clones both exist in the same neighbourhood, alter the fitness of the cells in the second clone
            by updating the weights array.

            """
            if clone_a in neighbour_clones and clone_b in neighbour_clones:
                # Alter the fitness of the cells from clone_b
                np.multiply(weights, multiplier, where=(neighbour_clones == clone_b), out=weights)

        def select_dividing_cell(self, neighbour_clones: np.ndarray[tuple[int], np.dtype[np.int_]], 
                                 weights: np.ndarray[tuple[int], np.dtype[np.float64]]) -> int:
            """Apply any custom rules to the input weights

            Args:
                neighbour_clones (np.ndarray[tuple[int], np.dtype[np.int_]]): The clone ids in the cell neighbourhood.
                weights (np.ndarray[tuple[int], np.dtype[np.int_]]): The fitness of the cells (prior to these rules). 
                  This array will be overwritten, so make sure it is a copy, not a view of the original fitness values. 

            Returns:
                int: The clone id of the cell that will divide
            """
            if len(np.unique(neighbour_clones)) == 1:
                # Only one clone in the neighbourhood, so a cell from that clone must divide
                return neighbour_clones[0]
            
            # Update the cell fitness based on the clones in the neighbourhood
            for (clone_a, clone_b), multiplier in self.interaction_effects.items():
                self.apply_interaction_effect(clone_a, clone_b, multiplier, 
                                              neighbour_clones, weights)
                
            # Calculate the relative fitness of the cells
            weights_sum = weights.sum()
            rel_weights = weights / weights_sum

            # Randomly select a neighbour, with the probabilities weighted by the relative fitness
            selected_clone = neighbour_clones[np.random.multinomial(1, rel_weights).argmax()]
            return selected_clone
        
    
    # Create a custom class using those rules
    class MyCustomMoran2D(Moran2D):

        def __init__(self, parameters, custom_rules):
            super().__init__(parameters)
            self.custom_rules = custom_rules

        def get_dividing_cell(self, coord: int, current_data: SpatialCurrentData) -> int:
            # get the clone ids in the neighbourhood using the grid array and the neighbour map
            neighbour_clones = current_data.grid_array[self.neighbour_map[coord]]
            # Get the fitness of those cells (prior to the custom rules being applied)
            weights = self.clones_array[neighbour_clones, self.fitness_idx].copy()

            # Use the custom rules to select the dividing cell
            return self.custom_rules.select_dividing_cell(neighbour_clones=neighbour_clones, weights=weights)
        
    # Set up parameters as usual for a Moran2D simulation
    params = Parameters(
        algorithm="Moran2D", 
        times=TimeParameters(max_time=5, division_rate=1), 
        population=PopulationParameters(
            initial_grid=np.concatenate([np.zeros(shape=120, dtype=int), 
                                         np.ones(shape=140, dtype=int), 
                                         np.full(shape=140, fill_value=2)]).reshape(20, 20),
            cell_in_own_neighbourhood=False
        ), 
    )

     # Define the rules object
    custom_rules = CompetitionRules(
        interaction_effects={
            (0, 1): 0.25,  # The presence of clone 0 will half the fitness of clone 1
            (1, 2): 0.25,  # The presence of clone 1 will half the fitness of clone 2
            (2, 0): 0.25,   # The presence of clone 2 will more than half the fitness of clone 0
        }
    )
    np.random.seed(0)
    sim = MyCustomMoran2D(parameters=params, custom_rules=custom_rules)

    sim.run_sim()

    np.testing.assert_equal(sim.population_array.toarray()[:, -1], np.array([103, 193, 104]))

    
            
            
