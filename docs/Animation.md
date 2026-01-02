# Animation

A guide to creating videos of the simulations.   
This is mostly for the 2D simulations (Moran2D and WF2D).   

There is also (an unreliable) method for visualising the non-spatial simulations on a 2D-plane 
(with the understanding that it is not particularly meaningful).  

To create the animations, you need ffmpeg installed.  

--------

First run a simulation:
```python
import numpy as np
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters

np.random.seed(2)
initial_grid = np.concatenate([np.zeros(5000), np.ones(5000)]).reshape(100, 100)
p = Parameters(
    algorithm='Moran2D', 
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=False)
)
s = p.get_simulator()
s.run_sim()
```


After running the simulation, a video can be made.

Various options can be set for the quality, size and speed of the video. 
By default, there will be one frame per simulation sample point

```python
s.animate('outfile1.mp4', figsize=(3, 3), dpi=300, bitrate=1000, fps=5)
```

-----

There are some options to overlay some basic text over the videos  

You can add a fixed label which is placed for the entire video. 

You can also add a time counter. 


```python
s.animate(
    'outfile2.mp4', figsize=(3, 3), dpi=300, bitrate=1000, fps=5, 
    fixed_label_text='My simulation', fixed_label_loc=(10, 40), 
    fixed_label_kwargs={'fontsize': 14, 'fontweight': 3}, 
    show_time_label=True, time_label_units='days', time_label_loc=(70, 5), 
    time_label_kwargs={'fontsize': 10, 'color': 'r'}, time_label_decimal_places=1
)
```


## Showing fitness values

This is an option to colour the videos by the fitness value of each cell instead of the clone id.  
Since this is a less used function, it does not currently have the same set of options (e.g. no text overlay)

```python
import matplotlib.cm as cm
from clone_competition_simulation import Gene, MutationGenerator, UniformDist, FitnessParameters

# Set up a simulation with some mutations that change the fitness. 
# Keeping the two initial neutral clones as before to demonstrate that they are not used to colour the cells here 
np.random.seed(2)
initial_grid = np.concatenate([np.zeros(5000), np.ones(5000)]).reshape(100, 100)
mut_gen = MutationGenerator(
    genes=[Gene(name='Gene1', mutation_distribution=UniformDist(1, 1.5), synonymous_proportion=0)], 
    combine_mutations='add'
)
p = Parameters(
    algorithm='Moran2D', 
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=False),
    fitness=FitnessParameters(mutation_generator=mut_gen, mutation_rates=0.1)
)
s = p.get_simulator()
s.run_sim()

s.animate('outfile3.mp4', fitness=True, fitness_cmap=cm.plasma, min_fitness=1,
          figsize=(4, 3), dpi=300, bitrate=1000, fps=5)
```



## Videos for non-spatial simulations

It is not recommended to use this. Describing here mostly just for completeness.   

This represents the relative sizes of clones on a 2D plane.   
It is not a true representation of the clonal competition, since the non-spatial simulations do not take account of the postions of cells.  

The method used to produce these videos attempts to place cells in an expanding clone on the border of that clone. 
It does not always work, and can fail to produce a video in some cases.   
Since it is very rarely used, it is not optimised and can be very slow (much slower than the simulations themselves).   


```python
# Run a non-spatial simulation
np.random.seed(0)
p = Parameters(
    algorithm='Moran',
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_size_array=np.full(10, 10))
)
s = p.get_simulator()
s.run_sim()

# The animation takes a long time...
s.animate('outfile4.mp4', grid_size=100,  #Will make a square grid
          figsize=(3, 3), dpi=300, bitrate=1000, fps=5)
```
