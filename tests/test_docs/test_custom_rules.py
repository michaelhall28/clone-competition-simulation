"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np

from clone_competition_simulation import (WF, WF2D, Branching, Moran, Moran2D,
                                          NonSpatialCurrentData, Parameters,
                                          PopulationParameters,
                                          SpatialCurrentData, TimeParameters)


def test_custom_rules():
    class MyCustomMoran(Moran):  # Inherit from the Moran class
        """Custom rule: Always take from the lowest id clone and add to the highest"""

        def get_dividing_cell(self, current_data: NonSpatialCurrentData) -> int:
            """
            This function determines which clone will divide.  

            This should return the index of the clone that will divide.  
            NOTE: this is *not* the clone_id, this is the index of the 
            clone in the current_data.current_population array.
            """
            # Select the last clone in the current_population array
            return len(current_data.current_population) - 1
        
        def get_differentiating_cell(self, current_data: NonSpatialCurrentData) -> int:
            """
            This function determines which cell will die.  

            This should return the index of the clone that will lose a cell.  
            NOTE: this is *not* the clone_id, this is the index of the 
            clone in the current_data.current_population array.
            """
            # Select the first clone in the current_population array
            return 0
        
    # Set up the parameters as if you are running a normal Moran simulation
    params = Parameters(
        algorithm="Moran", 
        times=TimeParameters(max_time=1, division_rate=1, samples=6), 
        population=PopulationParameters(initial_size_array=np.ones(5, dtype=int))
    )
    # Instead of running params.get_simulator() to create the simulation object, pass the parameters to your custom class.
    sim = MyCustomMoran(parameters=params)

    # Then continue as normal. 
    sim.run_sim()

    sim.population_array.toarray()


def test_custom_rules1():
    class MyCustomMoran2D(Moran2D):
        """Cells die in grid position order, and the highest number clone will divide"""

        def __init__(self, parameters: Parameters):
            super().__init__(parameters)

            # We could replace the death coordinates here if they do not depend on 
            # the current state of the simulation (other than the simulation step)
            # self.death_coords = ...

        def get_differentiating_cell(self, i: int, current_data: SpatialCurrentData) -> int:
            """
            If the dying cell depends on the current state of the simulation, 
            this function needs to be overwritten
            Needs to return coordinate of the dying cell.
            """
            coord = i % self.total_pop  # The dying cell goes in order across the grid
            
            return  coord

        def get_dividing_cell(self, coord: int, current_data: SpatialCurrentData) -> int:
            """
            This function determines which clone will divide and should return the clone_id.

            For this custom class, it will return the highest clone id in the neighbourhood.
            """
            # get the clone ids in the neighbourhood using the grid array and the neighbour map
            neighbour_clones = current_data.grid_array[self.neighbour_map[coord]]

            # Select the highest clone in the neighbourhood
            return neighbour_clones.max()
    
    
    # Set up the parameters as if you are running a normal Moran2D simulation
    params = Parameters(
        algorithm="Moran2D", 
        times=TimeParameters(max_time=1, division_rate=1, samples=16), 
        population=PopulationParameters(initial_grid=np.arange(16).reshape(4, 4), cell_in_own_neighbourhood=False)
    )
    # Instead of running params.get_simulator() to create the simulation object, 
    # pass the parameters to your custom class.
    sim = MyCustomMoran2D(parameters=params)

    # Then continue as normal. 
    sim.run_sim()

    sim.grid_results[0]
    sim.grid_results[1]
    sim.grid_results[5]


def test_custom_rules2():
    class MyCustomWF(WF):
        """Always take from the lowest id clone and add to the highest"""

        def get_next_generation(self, current_data: NonSpatialCurrentData) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
            """This function should return the cell counts for the new generation of clones

            Args:
                current_data (NonSpatialCurrentData): Current clone cell counts (an array of clone sizes and an array of clone ids)

            Returns:
                np.ndarray[tuple[int], np.dtype[np.int_]]: The new clone cell counts (for the same clone ids as the current_data input)
            """

            # Kill off the first clone and add the cells to the last
            input_clone_counts = current_data.current_population

            output_clone_counts = input_clone_counts.copy()
            # Take the first surviving clone cells and add to the last clone
            output_clone_counts[-1] += output_clone_counts[0]
            output_clone_counts[0] = 0
            return output_clone_counts
        

    # Set up the parameters as if for a normal Wright-Fisher simulation
    params = Parameters(
        algorithm="WF", 
        times=TimeParameters(max_time=4, division_rate=1, samples=6), 
        population=PopulationParameters(initial_size_array=np.array([1, 2, 3, 4, 1]))
    )
    # Pass the parameters to the custom class
    sim = MyCustomWF(parameters=params)
    # Run the sim as normal
    sim.run_sim()

    # We can see the last clone takes the cells from the other clones, one clone at each step
    sim.population_array.toarray()


def test_custom_rules3():
    class MyCustomWF2D(WF2D):
        """All cells just rotate along the grid by one position each step"""
        
        def get_next_generation(self, current_data: SpatialCurrentData) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
            """This function needs to return the new grid (in a 1D array representing the 2D grid).  
            Each value in the array is the clone id of the cell in that location. 

            Here we move each cell along one position in the grid
            :return:
            """
            # Get the grid array from the last generation
            current_grid_array = current_data.grid_array

            # Create the new grid array and return it
            new_grid_array = np.concat([current_grid_array[-1:], current_grid_array[:-1]])

            return new_grid_array
            
    # Set up the parameters as if for a normal WF2D simulation.
    params = Parameters(
        algorithm="WF2D", 
        times=TimeParameters(max_time=3, division_rate=1, samples=4), 
        population=PopulationParameters(initial_grid=np.arange(4).reshape(2, 2), cell_in_own_neighbourhood=False)
    )

    # Pass the parameters to the custom class
    sim = MyCustomWF2D(parameters=params)

    # Run the simulation as normal
    sim.run_sim()

    sim.grid_results[0]
    sim.grid_results[1]
    sim.grid_results[2]
    sim.grid_results[3]


def test_custom_rules4():

    class MyCustomBranching(Branching):
        """The cell division probability depends on the clone id"""

        def does_cell_divide(self, clone_id: int) -> bool:
            # If the clone id is larger than 3 it divides, otherwise it dies
            return clone_id > 3
        
    # Set up the parameters for a normal Branching simulation
    params = Parameters(
        algorithm="Branching", 
        times=TimeParameters(max_time=3, division_rate=1, samples=4), 
        population=PopulationParameters(initial_size_array=np.array([5, 4, 3, 2, 1]))
    )

    # Pass the parameters to the custom class
    sim = MyCustomBranching(parameters=params)

    # And run the simulation as normal
    sim.run_sim()

    sim.population_array.toarray()


def test_custom_rules5():
    # Define the competition rules that we will apply at each step. 
    class CompetitionRules:
        """
        These are rules for how clones change the fitness of other clones in their vicinity. 
        The interaction_effects are a list of (clone A, clone B, fitness multiplier) tuples. 
        If cells from clone A and clone B are both present in a cell neighbourhood, then the 
        fitness of clone B will be multiplied by the fitness multiplier. 
        """

        def __init__(self, interaction_effects: list[tuple[int, int, float]]):
            self.interaction_effects = interaction_effects

        @staticmethod
        def apply_interaction_effect(clone_a: int, clone_b: int, multiplier: float,
                                        neighbour_clones: np.ndarray[tuple[int], np.dtype[np.int_]], 
                                        weights: np.ndarray[tuple[int], np.dtype[np.float64]]) \
                                        -> None:
            """If the two clones both exist in the same neighbourhood, alter the fitness of the cells in the second clone
            by updating the weights array. 

            If cells from clone a and clone b are both in the neighbourhood, multiply the fitness of the clone b cells by 
            the multiplier. 

            Returns:
                None
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
                # Only one clone in the neighbourhood, so a cell from that clone must divide. We can skip the calculations
                return neighbour_clones[0]
            
            # Update the cell fitness based on the clones in the neighbourhood
            for clone_a, clone_b, multiplier in self.interaction_effects:
                self.apply_interaction_effect(clone_a, clone_b, multiplier, 
                                                neighbour_clones, weights)
                
            # Calculate the relative fitness of the cells
            weights_sum = weights.sum()
            rel_weights = weights / weights_sum

            # Randomly select a neighbour, with the probabilities weighted by the relative fitness
            selected_clone = neighbour_clones[np.random.multinomial(1, rel_weights).argmax()]
            return selected_clone
        
    
    # Create a custom algorithm class using those rules
    class MyCustomMoran2D(Moran2D):

        # Overwrite the init function to include our new rules
        def __init__(self, parameters: Parameters, custom_rules: CompetitionRules):
            super().__init__(parameters)
            self.custom_rules = custom_rules 

        def get_dividing_cell(self, coord: int, current_data: SpatialCurrentData) -> int:
            """
            This function determines which clone will divide.  
            We will use our custom rules class. 

            """
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
            # We will have 3 different clones, with ids 0, 1 and 2
            initial_grid=np.concatenate([np.zeros(shape=120, dtype=int), 
                                            np.ones(shape=140, dtype=int), 
                                            np.full(shape=140, fill_value=2)]).reshape(20, 20),
            cell_in_own_neighbourhood=False
        ), 
    )

    # Define the rules object
    # 0 beats 1, 1 beats 2, 2 beats 0
    custom_rules = CompetitionRules(
        interaction_effects=[
            (0, 1, 0.25),  # The presence of clone 0 will quarter the fitness of clone 1
            (1, 2, 0.25),  # The presence of clone 1 will quarter the fitness of clone 2
            (2, 0, 0.25),  # The presence of clone 2 will quarter the fitness of clone 0
        ]
    )

    # Initialise the custom algorithm class
    sim = MyCustomMoran2D(parameters=params, custom_rules=custom_rules)

    # Run the simulation as usual
    sim.run_sim()
