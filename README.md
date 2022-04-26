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

There are Jupyter Notebooks containing guides to the various features of the code in the Tutorials directory.

## Algorithms
There are 5 algorithms that can be run.

Non-spatial algorithms:
- "Branching". A branching process based on the single progenitor model from Clayton, Elizabeth, et al. "A single type of progenitor cell maintains normal epidermis." Nature 446.7132 (2007): 185-189.
- "Moran". A Moran-style model. At each simulation step, one cell dies and another cell divides, maintaining the overall population.  
- "WF". A Wright-Fisher style model. At each simulation step an entire generation of cells is produced from the previous generation.

2D algorithms:
- "Moran2D". A Moran-style model constrained to a 2D hexagonal grid. At each simulation step, one cell dies and a *cell from an adjacent location in the grid* divides, maintaining the overall population.
- "WF2D". A Wright-Fisher style model constrained to a 2D hexagonal grid. At each simulation step an entire generation of cells is produced from the previous generation, where cell parents must be from the local neighbourhood in the grid.
