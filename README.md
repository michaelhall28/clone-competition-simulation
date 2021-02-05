# clone-competition-simulation
Python3 simulations of clone competition during ongoing mutagenesis.

## Installation
To install, clone the git repository  
`git clone https://github.com/michaelhall28/clone-competition-simulation.git`

Dependencies can be installed using conda (https://docs.conda.io). If installing dependencies using another method, be aware that this code may not work with old versions of Scipy because of changes to scipy.sparse (tested for scipy 1.5.2).  
`cd clone-competition-simulation`  
`conda env create -f environment.yml`  
`conda activate competition`

and install the code in this repository  
`pip install -e .`

## Running simulations

First, the parameters for the simulation are defined. The Parameters class checks that the parameters are appropriate for the chosen algorithm.
e.g.
```
from clone_competition_simulation.parameters import Parameters
from clone_competition_simulation.fitness_classes import Gene, UniformDist, MutationGenerator

# Define the effect of mutations that appear during the simulation
mutation_generator = MutationGenerator(genes=[Gene('example_gene', UniformDist(1, 2), synonymous_proportion=0.5)],
                                        combine_mutations='multiply')

p = Parameters(algorithm='WF2D', grid_shape=(100, 100),
                mutation_generator=mutation_generator,
                mutation_rates=0.01, max_time=20,
                print_warnings=False, division_rate=1,
                cell_in_own_neighbourhood=True)

```

Then the simulation can be initialised and run from the parameter object

```
s = p.get_simulator()
s.run_sim()
s.muller_plot()
```
