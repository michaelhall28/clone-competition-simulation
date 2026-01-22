# Run Config File

As well as defining all arguments explicitly in the `Parameters` object, you can provide a yaml config file 
containing parameters. This could be useful if using similar parameters across multiple scripts.  
The parameters from the yaml file can be overwritten/augmented with the arguments passed to the `Parameters` object. 


## Example config file

An example file is included in the repository (`example_run_config.yml`). This includes a few of the 
options that can be defined in the file.  
```yaml
algorithm: "WF"
times:
  division_rate: 1
  max_time: 10
  samples: 100
population:
  initial_cells: 100
fitness:
  fitness_array: 1  # For initial clones. Integers be converted to a list
  mutation_rates: 0
  initial_mutant_gene_array: -1  # For the genes of initial clones. Integers be converted to a list. -1 means not associated with a gene.
labels:
  label_array: 0   # For initial clones. Integers be converted to a list
treatment:
  treatment_effect: 1
differentiated_cells:
  stratification_sim_percentile: 1
plotting:
  figsize: [10, 10]
```

Currently, it is not possible to define Genes or MutationGenerators using the config file. 

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
    Algorithm.WF


```python
from clone_competition_simulation import Parameters

p = Parameters(run_config_file="/path/to/example_run_config.yml", algorithm="Moran")

print(p.algorithm)
```
    Algorithm.MORAN