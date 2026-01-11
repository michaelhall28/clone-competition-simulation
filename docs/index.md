## Clone Competition Simulation

Python3 simulations of clone competition during ongoing mutagenesis.

This documentation contains a series of guides demonstrating how to run
the simulations in this repository and view the results.

### Guides

These are roughly in order of simple to complicated, and useful to less useful.

#### [Installation](Installation.md)
- Requirements and installation

#### [Introduction](Introduction.md)
- A guide to the basics of running a simulation and viewing the results

#### [Algorithms](Algorithms.md)
- A brief description of the various algorithms that can be run

#### [Simulation Length](SimulationLength.md)

- Altering division rate, simulation length and sampling times
- Printing simulation progress

#### [Defining Initial Clone Sizes](DefiningInitialCloneSizes.md)
- A guide to the various methods of defining the initial clones in a simulation

#### [Defining Initial Clone Fitness](DefiningInitialCloneFitness.md)
- A very brief guide to the `fitness_array` argument of the Parameters class

#### [Lineage Tracing Simulations](LineageTracingSimulations.md)
- How to set up simulations of lineage tracing experiments
- How to introduce inheritable labels into simulations

#### [Mutations](Mutations.md)
- How to run simulations with randomly occurring mutations
- The basics of defining "genes" and the fitness effects of mutations
- dN/dS ratios

#### [Mutations2](Mutations2.md)
- How to extract mutant clone sizes from simulations
- Muller plots
- Incomplete moment plots

#### [Mutations3](Mutations3.md)
- More complex methods to define clone fitness
- How the fitness effects of multiple mutations are combined
- Multiple genes and epistatic effects
- Diminishing returns

#### [Simulating Biopsies](SimulatingBiopsies.md)
- Counting clone sizes in biopsies taken from 2D simulations
- Simulating sequencing with a lower limit of detection and fixed or random read depth
- Merging mutations found in multiple biopsies
- dN/dS calculation on the "sequencing" results

#### [Animation](Animation.md)
- Creating videos of the simulations
- Colouring using the fitness of cells instead of the clone

#### [Colours](Colours.md)
- How to define colours for the Muller plots and animations
- Setting colours based on clone properties, such as fitness, genes mutated and labels
- Changing colour schemes

#### [Early Stopping Conditions](EarlyStoppingConditions.md)
- Ending a simulation once a certain condition has been reached instead of running until the max_time
- A few examples showing how to define the end conditions

#### [Differentiated Cells](DifferentiatedCells.md)
- Simulating differentiated cells for the Branching, Moran and Moran2D algorithms
- These have no effect on the actions of proliferating cells and have no physical position in the Moran2D
- How to get proliferative cell only and total basal clone sizes

#### [Treatments](Treatments.md)
- Simulating changes to the competitive environment, e.g. the application of drugs
- How to apply treatments affecting initial clones (e.g. for lineage tracing) or for acquired mutations.

#### [Storing And Loading Simulations](StoringAndLoadingSimulations.md)
- How to save and load simulation objects using pickle
