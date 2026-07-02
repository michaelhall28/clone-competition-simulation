# Run Config File

As well as defining all arguments explicitly in the `Parameters` object, you can provide a yaml config file 
containing parameters. This could be useful if using similar parameters across multiple scripts.  
The parameters from the yaml file can be overwritten/augmented with the arguments passed to the `Parameters` object. 


## Example config file

An example file is included in the repository (`example_run_config.yml`). 
This includes many of the options that can be defined in the file.  
You do not need to include sections in the file, just what you need. 

```yaml
algorithm: "Moran"
times:
  division_rate: 1.2
  max_time: 10
  samples: 5
population:
  initial_size_array: [100, 100]
  cell_in_own_neighbourhood: false
  population_limit: 201
fitness:
  initial_fitness_array: [1, 2]
  mutation_rates: 0
  initial_mutant_gene_array: [Gene1, Gene2]
  fitness_calculator: 
    genes:
      - name: Gene1
        mutation_distribution: 
          cls: NormalDist
          var: 1.1
          mean: 1.2
        synonymous_proportion: 0.5
        weight: 1
      - name: Gene2
        mutation_distribution: 
          cls: FixedValue
          value: 1.1
        synonymous_proportion: 0.8
        weight: 2
    combine_mutations: multiply
    multi_gene_array: true
    combine_array: add
    mutation_combination_class:
      cls: BoundedLogisticFitness
      a: 1.2
      b: 2.3
    epistatics: 
      - name: Epi1
        gene_names: [Gene1, Gene2]
        fitness_distribution:
          cls: ExponentialDist
          mean: 1.5
          offset: 1.1
labels:
  initial_label_array: [1, 2]
  label_times: [3, 6]
  label_frequencies: [0.04, 0.1]
  label_values: [3, 4]
  label_fitness: [2, 3] 
  label_genes: [Gene1, Gene2]
treatment:
  treatment_timings: [2, 5]
  treatment_effects: [[1, 0.5, 1.2], [1, 0.8, 0.3]]
  treatment_replace_fitness: true
differentiated_cells:
  r: 0.1
  gamma: 1.4
  stratification_sim_proportion: 0.99
plotting:
  figsize: [10, 8]
  plot_colour_maps:
    colour_rules:
      - rule_filter:
          - clone_feature: label
            value: 1
          - clone_feature: last_mutated_gene
            value: Gene1
        colourmap: viridis
      - rule_filter:
          - clone_feature: initial
            value: true
        colourmap: Reds
    all_clones_noisy: true
    use_fitness: true
tmp_store: tmp1.pickle
```

## Specifying the config file

You can specify the file path:
```python
from clone_competition_simulation import Parameters

p = Parameters(run_config_file="/path/to/example_run_config.yml")
```


Or set the `CCS_RUN_CONFIG` environment variable:

```python
import os
from clone_competition_simulation import Parameters
os.environ["CCS_RUN_CONFIG"] = "/path/to/example_run_config.yml"
p = Parameters()
```

## Combining parameters from the config file with init parameters

Arguments passed when initiating a `Parameters` object will take priority over the parameters from the config file.

```python
from clone_competition_simulation import Parameters

p = Parameters(run_config_file="/path/to/example_run_config.yml")

print(p.algorithm)
```
    Algorithm.MORAN


```python
from clone_competition_simulation import Parameters

p = Parameters(
    run_config_file="/path/to/example_run_config.yml",
    algorithm="Branching"
)

print(p.algorithm)
```
    Algorithm.BRANCHING


## Limitations

Not all parameters can be defined in the config file. 

### No custom functions

Functions in the fitness calculator have to be selected from 
pre-built options using strings:

- mutation distribution functions for genes and epistatics - use keys from `PREDEFINED_DISTRIBUTIONS`
- combine_mutations - use keys from `FITNESS_COMBINATION_FUNCTIONS`
- combine_array - use keys from `GENE_COMBINATION_FUNCTIONS`
- the mutation combination class - use keys from `PREDEFINED_TRANSFORMATIONS`

To see the available options, those function dictionaries can be imported via
```python
from clone_competition_simulation import (
  PREDEFINED_DISTRIBUTIONS, 
  FITNESS_COMBINATION_FUNCTIONS,
  GENE_COMBINATION_FUNCTIONS, 
  PREDEFINED_TRANSFORMATIONS
)
```
----

Colour maps defined in the config file have to be selected from those in 
matplotlib.cm (use the colormap name in the config). 

### No partial overwriting of nested objects

When overwriting config options using the Parameters initialisation, 
you can update direct arguments of the parameter classes (
  `FitnessParameters`, `TimeParameters` etc.), but you cannot partially 
update arguments of arguments. 

E.g. `FitnessParameters.fitness_calculator`:
You can entirely overwrite the FitnessCalculator, but you cannot 
keep some of the config FitnessCalculator parameters and replace others.

The other case this applies to is `PlottingParameters.plot_colour_maps`

```python
from clone_competition_simulation import (
  Parameters, 
  FitnessParameters, 
  FitnessCalculator, 
  PlottingParameters, 
  PlotColourMaps
)

p = Parameters(
    run_config_file="/path/to/example_run_config.yml",
    fitness=FitnessParameters(
      fitness_calculator=FitnessCalculator(
        # All the FitnessCalculator parameters here will be used, 
        # and none from the config
        ...
      )
    ), 
    plotting=PlottingParameters(
      plot_colour_maps=PlotColourMaps(
        # All the PlotColourMaps parameters here will be used, 
        # and none from the config
        ...
      )
    )
)
```







