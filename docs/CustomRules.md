# Custom Rules

There are 5 algorithms that can be run as a default: 

 - Moran
 - Moran 2D
 - Wright-Fisher
 - Wright-Fisher 2D
 - Branching

Each of these algorithms contain functions which determine how clone fitness controls cell fate, and therefore how clone sizes change in each step of the simulation.  

Variations of these algorithms can be made by created subclasses and overwriting a few key functions (which vary by algorithm). 

## Moran

Variants of this algorithm will:

- Have one dividing cell and one dying/differentiating cell per simulation step
- Not include any spatial considerations

-----

The default Moran algorithm will select the dividing cell based on cell fitness, and every cell has an equal chance of dying.  

To change the rules for selecting the dividing cell, overwrite the `get_dividing_clone` function.  
To change the rules for selecting the dying cell, overwrite the `get_differentiating_clone` function.  

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

    def get_dividing_clone(self, current_data: CurrentData) -> int:
        """
        This function determines which clone will divide.  

        This should return the index of the clone that will divide.  
        NOTE: this is not the clone_id, this is the index of the clone in the current_data.current_population array.
        """
        # Select the last clone in the current_population array
        return len(current_data.current_population) - 1
    
    def get_differentiating_clone(self, current_data: CurrentData) -> int:
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


## Wright-Fisher


## Wright-Fisher 2D


## Branching


## Simulation inputs to functions

The custom functions all have access to the `current_data` input, as well as any other part of the simulation (accessed through `self`).  
These are a few objects from the simulation that could be useful input to the custom functions:

### Clones array


### Fitness array


### 



