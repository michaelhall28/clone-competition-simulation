# Defining Initial Clone Fitness

The fitness of an initial clone is defined using the `fitness_array`.  

If `fitness_array` is defined, it must contain a fitness value for each unique initial clone.   
If no `fitness_array` is provided, all initial clones will be given a fitness value of 1.  

The fitness of mutations that are generated during a simulation is more complex, and described in other tutorial guides.  


-----
With no fitness defined, all of the clones are given a fitness of 1

```python
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters
import numpy as np
p = Parameters(
    algorithm='Moran',
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_size_array=np.ones(16))
)
s = p.get_simulator()
s.run_sim()
print(s.view_clone_info())
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
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

-----
Give each initial clone a random fitness value between 0.5 and 1.5

```python
from clone_competition_simulation import FitnessParameters

p = Parameters(
    algorithm='Moran', 
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_size_array=np.ones(16)),
    fitness=FitnessParameters(fitness_array=np.random.uniform(0.5, 1.5, size=16))
)
s = p.get_simulator()
s.run_sim()
print(s.view_clone_info())
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
      <td>0.901536</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1.402680</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>1.113649</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0.895848</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0.598618</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>0.896367</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
      <td>1.229735</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
      <td>1.271452</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0</td>
      <td>1.052849</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
      <td>0.687008</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0</td>
      <td>0.930168</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0</td>
      <td>1.255376</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0</td>
      <td>0.902447</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0</td>
      <td>1.493341</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0</td>
      <td>0.900351</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0</td>
      <td>1.392611</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

-----
The same principle works for clones defined using a grid

```python
p = Parameters(
    algorithm='Moran2D',
    times=TimeParameters(max_time=10, division_rate=1), 
    population=PopulationParameters(initial_grid=np.arange(16).reshape(4, 4), cell_in_own_neighbourhood=False),
    fitness=FitnessParameters(fitness_array=np.random.uniform(0.5, 1.5, size=16))
)
s = p.get_simulator()
s.run_sim()
print(s.view_clone_info())
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
      <td>0.710121</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0.566186</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0.640329</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0.805205</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0.952062</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>0.847487</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0</td>
      <td>0.770136</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
      <td>0.673199</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0</td>
      <td>0.777511</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
      <td>1.117545</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0</td>
      <td>0.977799</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0</td>
      <td>0.576575</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0</td>
      <td>1.301780</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0</td>
      <td>1.220930</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0</td>
      <td>0.714362</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0</td>
      <td>1.433839</td>
      <td>0</td>
      <td>-1</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>
