# Custom Rules

There are 5 algorithms that can be run as a default: 

 - Moran
 - Moran 2D
 - Wright-Fisher
 - Wright-Fisher 2D
 - Branching

Each of these algorithms contain functions which determine how clone fitness controls cell fate, and therefore how clone sizes change in each step of the simulation.  

Variations of these algorithms can be made by creating subclasses and overwriting a few key functions (which vary by algorithm).  

In general, custom cell competition rules can be implemented as follows:

```python

# Define the custom class
class MyCustomAlgorithm([Inherit from an existing simulation class]): 

    # Customise the init function if any additional data or objects need to be accessed by the custom functions
    def __init__(self, parameters: Parameters, 
                # ...any additional arguments needed for the custom code
                ):
        super().__init__(parameters)
        # add any attributes here as needed


    # Customise some class methods. These vary depending on which type of algorithm is being customised. See below. 


# Run the custom simulation

# Define the simulation parameters, same as usual
params = Parameters(
    algorithm="Whichever algorithm type was inherited from", 
    ...
)
# Instead of running params.get_simulator() to create the simulation object, pass the parameters to your custom class.
sim = MyCustomAlgorithm(parameters=params)

# Then run the simulation as normal. 
sim.run_sim()
```

## Moran

Variants of this algorithm will:

- Have one dividing cell and one dying/differentiating cell per simulation step
- Not include any spatial considerations

-----

The default Moran algorithm will select the dividing cell based on cell fitness, and every cell has an equal chance of dying.  

To change the rules for selecting the dividing cell, overwrite the `get_dividing_cell` function.  
To change the rules for selecting the dying cell, overwrite the `get_differentiating_cell` function.  

Both functions have access to the `current_data` variable (see below) and any attributes of the simulation (see the section at the end of this guide). 

### CurrentData

This represents the current cell population in a simulation and is an object with two attributes:

`current_data.current_population` is a 1D array of integers. It is the cell count of all non-extinct clones.  
`current_data.non_zero_clones` is also a 1D array of integers. It is the clone ids of the non-extinct clones.  

For example, if clones 1, 3 and 5 have 10, 20 and 30 cells respectively, and all other clones have zero cells, then 

`current_data.current_population = [10, 20, 30]` and `current_data.non_zero_clones = [1, 3, 5]`.  

(Excluding extinct clones can massively reduce the length of the current population array, and therefore speeds up the simulations).


### Example

Here is an example where the dividing cell will always come from the non-extinct clone with the highest id, and the dying cell will always come from the non-extinct clone with the lowest id. 

```python

from clone_competition_simulation import Moran


class MyCustomMoran(Moran):
    """Always take from the lowest id clone and add to the highest"""

    def get_dividing_cell(self, current_data: CurrentData) -> int:
        """
        This function determines which clone will divide.  

        This should return the index of the clone that will divide.  
        NOTE: this is not the clone_id, this is the index of the clone in the current_data.current_population array.
        """
        # Select the last clone in the current_population array
        return len(current_data.current_population) - 1
    
    def get_differentiating_cell(self, current_data: CurrentData) -> int:
        """
        This function determines which cell will die.  

        This should return the index of the clone that will lose a cell.  
        NOTE: this is not the clone_id, this is the index of the clone in the current_data.current_population array.
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

# View the clone population over time
print(sim.population_array.toarray())
```
    array([[1., 0., 0., 0., 0., 0.],
          [1., 1., 0., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 1., 0., 0.],
          [1., 2., 3., 4., 5., 5.]])


Here, the last clone grows, while the other clones die off in order. 

## Moran2D

Variants of this algorithm will:

- Have one dividing cell and one dying/differentiating cell per simulation step
- The dividing cell will be selected from cells in the immediate neighbourhood of the differentiating cell

-----

The default Moran2D algorithm will select the dividing cell based on cell fitness, and every cell has an equal chance of dying.  

To change the rules for selecting the dividing cell, overwrite the `get_dividing_cell` function.  
To change the rules for selecting the dying cell, overwrite the `get_differentiating_cell` function.  


### Differentiating cells

By default, every cell has an equal chance of dying. To implement this efficiently, the coordinates of the dying cells are all random selected at the start of the simulation in the init line `self.death_coords = np.random.randint(0, self.total_pop, size=self.parameters.times.simulation_steps)`. 

The `get_differentiating_cell` function is passed the number of the current simulation step (`i`), and returns the coordinate of the dying cells (from `self.death_coords`) and the clone id of the cell currently in that coordinate.  

To change the rules for dying, you could either: 

- overwrite the calculation of `self.death_coords` in the init function of the custom class
- overwrite the `get_differentiating_cell`, probably ignoring `self.death_coords` and running new code to select the dying cell. 


### Dividing cells

By default, the dividing cell is selected from the cells in the neighbourhood of the dying cell, with the division chance in proportion to the cell fitness.  

`get_dividing_cell` gets given the coordinate of the dying cell, and returns the clone id of the cell that will divide. 


### Example

Here is an example where the dividing cell will always come from the non-extinct clone with the highest id in the neighbourhood, and the cells will die in order of grid position. 

```python

from clone_competition_simulation import Moran2D, Parameters


class MyCustomMoran2D(Moran2D):
    """Cells die in grid position order, and the highest number clone will divide"""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        # We could replace the death coordinates here if they do not depend on 
        # the current state of the simulation (other than the simulation step)
        # self.death_coords = ...

    def get_differentiating_cell(self, i: int) -> tuple[int, int]:
        """
        If the dying cell depends on the current state of the simulation, this function needs to be overwritten
        Needs to return the clone id and coordinate of the dying cell.
        """
        coord = i % self.total_pop  # The dying cell goes in order across the grid
        
        # The current cell in that coordinate can be access from self.grid_array
        clone_id = self.grid_array[coord]

        return clone_id, coord

    def get_dividing_cell(self, coord: int) -> int:
        """
        This function determines which clone will divide.  

        This needs to return the clone_id of the clone that will divide.  
        """
        # For this custom class, it will return the highest clone id in the neighbourhood.

        # get the clone ids in the neighbourhood using the grid array and the neighbour map
        neighbour_clones = self.grid_array[self.neighbour_map[coord]]

        # Select the highest clone in the neighbourhood
        return neighbour_clones.max()
    
    
        
# Set up the parameters as if you are running a normal Moran2D simulation
params = Parameters(
    algorithm="Moran2D", 
    times=TimeParameters(max_time=1, division_rate=1, samples=16), 
    population=PopulationParameters(initial_grid=np.arange(16).reshape(4, 4), cell_in_own_neighbourhood=False)
)
# Instead of running params.get_simulator() to create the simulation object, pass the parameters to your custom class.
sim = MyCustomMoran2D(parameters=params)

# Then continue as normal. 
sim.run_sim()

# View the grids over time
print(sim.grid_results[0])
print(sim.grid_results[1])
print(sim.grid_results[5])
```
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])


    array([[15,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    array([[15, 15, 15, 15],
           [15,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])


The cells are replaced from the top left onwards, with the largest clone id (15 for all of these neighbourhoods) dividing into the position of the dying cell. 



## Wright-Fisher


## Wright-Fisher 2D


## Branching


## Simulation inputs to functions

The custom functions all have access to the `current_data` input, as well as any other part of the simulation (accessed through `self`).  
These are a few objects from the simulation that could be useful input to the custom functions:

### Clones array


### Fitness array


### 



