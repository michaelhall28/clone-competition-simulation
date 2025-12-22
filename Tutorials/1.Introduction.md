# Introduction

This notebook explains the basics of running simulations and plotting the results.  
See the subsequent notebooks for more in-depth guides to the various options available.  


# Setting up and running simulations

The Parameters class is used to define all aspects of the simulation to be run.  

This includes
- the algorithm used
- the timing of the simulation, including
  - how long it runs for
  - how fast the cells divide
  - how frequently to record the simulation state)
- the cell population, including 
  - total cell population size
  - initial clone populations
  - initial grid layout (if running a 2D simulation)
- the cell fitness, including
  - the fitness of the initial clones
  - the fitness (or fitness distribution) of any mutations that will appear during the simulation
  - how to combine the fitness of multiple mutations in the same cell
- and many more options


```python
# Import the Parameters class. This is used to set up all simulations
# Subsequent guides will go into more detail on the other parameter classes
from clone_competition_simulation import Parameters, PopulationParameters, TimeParameters, FitnessParameters

p = Parameters(
    algorithm='Moran',  # We will run a non-spatial Moran simulation. 
    times=TimeParameters(
        max_time=25,  # Run for 25 time units
        division_rate=1.4  # Set average division rate for all cells to 1.7 per time unit
    ), 
    population=PopulationParameters(  # Define the cell population
        initial_size_array=[100, 100, 100]  # There are three initial clones, with 100 cells in each
    ),
    fitness=FitnessParameters(  # Define the cell fitness
        fitness_array=[1, 1.02, 1.04]    # Each clone has a different fitness value
    )
)
```

Now that the parameters for the simulations have been defined, we can create and run the simulations themselves


```python
sim = p.get_simulator()
sim.run_sim()
```

If we want to run another simuation with the same parameters but a different random sequence, we can get another simulation from the parameters object


```python
sim2 = p.get_simulator()
sim2.run_sim()
```

The results are not identical to the first simulation because the random sequences are different.
E.g. the population arrays (which contain all of the clone sizes) are not the same
```python
import numpy as np
print(np.all(sim.population_array.toarray() == sim2.population_array.toarray()))
```
    False

For reproducibility, you can set the numpy random seed before running the simulation


```python
sim3 = p.get_simulator()
np.random.seed(0)
sim3.run_sim()

sim4 = p.get_simulator()
np.random.seed(0)
sim4.run_sim()
```

Here the population arrays are equal
```python
print(np.all(sim3.population_array.toarray() == sim4.population_array.toarray()))
```
    True

# Viewing simulation results

The simulation has now been run, and we can look at some of the simulation results. This is just a few quick examples.     
The particular functions appropriate for extracting and plotting the results will depend on the setup of the simulations, in particular whether new mutations occur during the simulations or whether they all mutations/clones in the simulation exist from the start.   
See the other tutorials for further details. 

---
You can see some information on the clones in the simulation.
This function returns a more readable version of the sim.clones_array which stores this information
```python
print(sim2.view_clone_info())
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>clone id</th>
      <th>label</th>
      <th>fitness</th>
      <th>generation born</th>
      <th>parent clone id</th>
      <th>last gene mutated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1.00</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1.02</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>1.04</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

----
The Muller plot shows the sizes of each clone over time. 

```python
import matplotlib.pyplot as plt

sim.muller_plot(figsize=(5, 5))
plt.show()
```
    
![png](1.Introduction_files/1.Introduction_15_0.png)

----
The population array stores the clone sizes. 
Each row is a unique clone. If new mutations occur, they start on a new row. 
It is a `scipy.sparse` array, so use `.toarray()` to convert to a numpy array
```python
print(sim.population_array.toarray())
```

    array([[100., 109., 115., 113., 115., 102.,  94.,  95.,  93.,  88.,  89.,
             96.,  92.,  89.,  84.,  90.,  90.,  93.,  87.,  90.,  92.,  92.,
             96.,  89., 103.,  99., 109.,  96., 102.,  99.,  96.,  94.,  98.,
             92.,  75.,  66.,  69.,  68.,  72.,  67.,  64.,  50.,  53.,  44.,
             45.,  35.,  29.,  25.,  16.,  16.,  21.,  22.,  23.,  22.,  21.,
             18.,  16.,  15.,  19.,  20.,  22.,  19.,  19.,  14.,  13.,  14.,
             14.,  14.,  15.,  12.,  10.,  10.,   9.,   7.,  11.,  11.,   9.,
              3.,   1.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
              0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
              0.,   0.],
           [100.,  89.,  81.,  70.,  77.,  83.,  88.,  96.,  96.,  98.,  87.,
             84.,  84.,  89.,  90.,  91.,  97.,  89.,  95., 108., 110., 113.,
            111., 114., 104., 103., 101., 101.,  90.,  89.,  87.,  78.,  78.,
             72.,  71.,  75.,  68.,  62.,  58.,  57.,  46.,  40.,  31.,  31.,
             31.,  32.,  31.,  36.,  40.,  36.,  37.,  34.,  25.,  24.,  23.,
             21.,  24.,  27.,  27.,  27.,  31.,  40.,  30.,  32.,  36.,  42.,
             40.,  40.,  35.,  35.,  33.,  32.,  39.,  38.,  31.,  36.,  34.,
             38.,  34.,  39.,  34.,  37.,  34.,  31.,  29.,  34.,  29.,  25.,
             25.,  23.,  23.,  19.,  14.,  20.,  14.,  11.,  13.,  15.,  15.,
             15.,  11.],
           [100., 102., 104., 117., 108., 115., 118., 109., 111., 114., 124.,
            120., 124., 122., 126., 119., 113., 118., 118., 102.,  98.,  95.,
             93.,  97.,  93.,  98.,  90., 103., 108., 112., 117., 128., 124.,
            136., 154., 159., 163., 170., 170., 176., 190., 210., 216., 225.,
            224., 233., 240., 239., 244., 248., 242., 244., 252., 254., 256.,
            261., 260., 258., 254., 253., 247., 241., 251., 254., 251., 244.,
            246., 246., 250., 253., 257., 258., 252., 255., 258., 253., 257.,
            259., 265., 260., 266., 263., 266., 269., 271., 266., 271., 275.,
            275., 277., 277., 281., 286., 280., 286., 289., 287., 285., 285.,
            285., 289.]])

----
The times in the simulation (which correspond to the columns in sim.population_array) can be accessed
```python
print(sim.times)
```

    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ,
            2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,
            4.5 ,  4.75,  5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25,  6.5 ,
            6.75,  7.  ,  7.25,  7.5 ,  7.75,  8.  ,  8.25,  8.5 ,  8.75,
            9.  ,  9.25,  9.5 ,  9.75, 10.  , 10.25, 10.5 , 10.75, 11.  ,
           11.25, 11.5 , 11.75, 12.  , 12.25, 12.5 , 12.75, 13.  , 13.25,
           13.5 , 13.75, 14.  , 14.25, 14.5 , 14.75, 15.  , 15.25, 15.5 ,
           15.75, 16.  , 16.25, 16.5 , 16.75, 17.  , 17.25, 17.5 , 17.75,
           18.  , 18.25, 18.5 , 18.75, 19.  , 19.25, 19.5 , 19.75, 20.  ,
           20.25, 20.5 , 20.75, 21.  , 21.25, 21.5 , 21.75, 22.  , 22.25,
           22.5 , 22.75, 23.  , 23.25, 23.5 , 23.75, 24.  , 24.25, 24.5 ,
           24.75, 25.  ])

-----
And various other plotting functions are available. These will be shown in other tutorials.
```python
sim.plot_average_fitness_over_time()
plt.show()
```

![png](1.Introduction_files/1.Introduction_18_0.png)
    
