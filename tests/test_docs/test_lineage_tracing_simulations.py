"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
import matplotlib.pyplot as plt
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters
from clone_competition_simulation import LabelParameters, FitnessParameters
from clone_competition_simulation import (
    FitnessParameters, 
    Gene, 
    FitnessCalculator, 
    NormalDist, 
    PlottingParameters,
    PLOT_COLOURS_EXAMPLE1
)

def test_lineage_tracing():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.ones(10000))
    )
    s = p.get_simulator()
    s.run_sim()

    fig, ax=plt.subplots(figsize=(5, 5))
    s.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=True, legend_label='Moran', legend_label_fit='SPM trend',
                                                fit_plot_kwargs={'c': 'r', 'linestyle': '--'}, ax=ax
                                                    )
    plt.legend()

    s.plot_surviving_clones_for_non_mutation()


    s.plot_clone_size_distribution_for_non_mutation()

    s.plot_clone_size_distribution_for_non_mutation(t=5, as_bar=True)

    s.plot_clone_size_scaling_for_non_mutation(times=[1, 2, 5, 10])

    p = Parameters(
        algorithm='Branching',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.ones(10000))
    )
    branch = p.get_simulator()
    branch.run_sim()

    p = Parameters(
        algorithm='WF',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.ones(10000))
    )
    wf = p.get_simulator()
    wf.run_sim()

    p = Parameters(
        algorithm='WF', 
        times=TimeParameters(max_time=10, division_rate=2),  # Double division rate
        population=PopulationParameters(initial_size_array=np.ones(10000))
    )
    wf_double = p.get_simulator()
    wf_double.run_sim()


    fig, ax=plt.subplots(figsize=(5, 5))
    s.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='Moran', ax=ax)
    branch.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='Branching', ax=ax)
    wf.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='WF', ax=ax)
    wf_double.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='WF double speed', ax=ax)
    plt.legend()

    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_grid=np.arange(10000).reshape(100, 100), cell_in_own_neighbourhood=False)
    )
    moran2D = p.get_simulator()
    moran2D.run_sim()

    p = Parameters(
        algorithm='WF2D',
        times=TimeParameters(max_time=10, division_rate=2),  # Double division rate here again
        population=PopulationParameters(initial_grid=np.arange(10000).reshape(100, 100), cell_in_own_neighbourhood=False)
    )
    wf2d = p.get_simulator()
    wf2d.run_sim()

    fig, ax=plt.subplots(figsize=(5, 5))
    s.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='Moran non-spatial', ax=ax)
    moran2D.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='Moran2D', ax=ax)
    wf2d.plot_mean_clone_size_graph_for_non_mutation(show_spm_fit=False, legend_label='WF2D', ax=ax)
    plt.legend()

    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(
            initial_size_array=np.concatenate([
                np.array([9950]),   # The wild type cells tracked as one big clone
                np.ones(50)]   # 50 single-cell clones
        )),
        fitness=FitnessParameters(
            initial_fitness_array=np.concatenate([
                [1],   # Wild type cells with fitness 1
                np.full(50, 1.3)  # The next 50 clones given fitness 1.3
                ])
        ),
        labels=LabelParameters(
            initial_label_array=np.concatenate([
                [0],  # Wild type clone labelled with 0
                np.ones(50)  # Mutant clones labelled with 1
                ])
        )
    )
    moran_nn = p.get_simulator()
    moran_nn.run_sim()

    # The clones with fitness>1 grow faster than the neutral clones. 
    fig, ax=plt.subplots(figsize=(5, 5))
    s.plot_mean_clone_size_graph_for_non_mutation(ax=ax, legend_label='Neutral', show_spm_fit=False)

    # Use the label option to plot only the clones labelled with 1. 
    moran_nn.plot_mean_clone_size_graph_for_non_mutation(label=1, legend_label='Non-neutral', ax=ax, show_spm_fit=False)

    plt.legend()

def test_lineage_tracing1():
    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )

    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=10000), # Start with 10000 wild type cells
        fitness=FitnessParameters(
            mutation_rates=0.002,  # Introduce a constant mutation rate.
            fitness_calculator=fitness_calculator
        ),
        labels=LabelParameters(
            label_times=3,  # Introduce a label part way through
            label_frequencies=0.01,  # label 1% of cells at random
            label_values=1,  # Use the value 1 for the label
        ),
        plotting=PlottingParameters(
            plot_colour_maps=PLOT_COLOURS_EXAMPLE1  # Set the colours (see Colours guide)
        )
    )
    moran_label = p.get_simulator()
    moran_label.run_sim()

    # The labelling events are marked with the black Xs and the clones are green
    # The other mutations are marked with red (non-synonymous) and blue (synonymous)
    moran_label.muller_plot(figsize=(7, 7))


    assert len(moran_label.get_labeled_population(label=1)) > 1


def test_lineage_tracing2():
    fitness_calculator = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(mean=1.1, var=0.1), 
                    synonymous_proportion=0.5)]
    )
    
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=10000), # Start with 10000 wild type cells
        fitness=FitnessParameters(
            mutation_rates=0.002,  # Introduce a constant mutation rate.
            fitness_calculator=fitness_calculator
        ),
        labels=LabelParameters(
            label_times=[3, 5],  # Introduce multiple labels
            label_frequencies=[0.01, 0.02],  # label 1% then 2% of cells at random
            label_values=[1, 2],  # Use different values for the labels
            label_fitness=[1.2, 1],  # The first label event increases cell fitness
        ),
        plotting=PlottingParameters(
            plot_colour_maps=PLOT_COLOURS_EXAMPLE1  # Set the colours (see Colours guide)
        )
    )
    moran_label2 = p.get_simulator()
    moran_label2.run_sim()

    # There are now two labelling events. 
    moran_label2.muller_plot(figsize=(7, 7))

    plt.plot(moran_label2.times, moran_label2.get_labeled_population(label=1), c='g', label='Label1')
    plt.plot(moran_label2.times, moran_label2.get_labeled_population(label=2), c='m', label='Label2')
    plt.ylabel('Population size')
    plt.xlabel('Time')

