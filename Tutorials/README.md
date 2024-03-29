## Tutorials

This directory contains a series of Jupyter notebooks demonstrating how to run
the simulations in this repository and view the results.

### Notebooks

These are roughly in order of simple to complicated, and useful to less useful.


#### 1.Introduction
- A guide to the basics of running a simulation and viewing the results

#### 2.Algorithms
- A brief description of the various algorithms that can be run

#### 3.BasicOptions
- Showing warnings during parameter setup
- Printing simulation progress
- Altering division rate, simulation length and sampling times

#### 4.DefiningInitialCloneSizes
- A guide to the various methods of defining the initial clones in a simulation

#### 5.DefiningInitialCloneFitness
- A very brief guide to the `fitness_array` argument of the Parameters class

#### 6.LineageTracingSimulations
- How to set up simulations of lineage tracing experiments
- This notebook also explains how to introduce inheritable labels into simulations

#### 7.Mutations
- How to run simulations with randomly occurring mutations
- The basics of defining "genes" and the fitness effects of mutations
- dN/dS ratios

#### 8.Mutations2
- How to extract mutant clone sizes from simulations
- Muller plots
- Incomplete moment plots

#### 9.Mutations3
- More complex methods to define clone fitness
- How the fitness effects of multiple mutations are combined
- Multiple genes and epistatic effects
- Diminishing returns

#### 10.Simulating Biopsies
- Counting clone sizes in biopsies taken from 2D simulations
- Simulating sequencing with a lower limit of detection and fixed or random read depth
- Merging mutations found in multiple biopsies
- dN/dS calculation on the "sequencing" results

#### 11.Animation
- Creating videos of the simulations
- Colouring using the fitness of cells instead of the clone

#### 12.Colours
- How to define colours for the Muller plots and animations
- Setting colours based on clone properties, such as fitness, genes mutated and labels
- Changing colour schemes

#### 13.EarlyStoppingConditions
- Ending a simulation once a certain condition has been reached instead of running until the max_time
- A few examples showing how to define the end conditions

#### 14.DifferentiatedCells
- Simulating differentiated cells for the Branching, Moran and Moran2D algorithms
- These have no effect on the actions of proliferating cells and have no physical position in the Moran2D
- How to get proliferative cell only and total basal clone sizes

#### 15.Treatments
- Simulating changes to the competitive environment, e.g. the application of drugs
- How to apply treatments affecting initial clones (e.g. for lineage tracing) or for acquired mutations.

#### 16.StoringAndLoadingSimulations
- How to save and load simulation objects using pickle
