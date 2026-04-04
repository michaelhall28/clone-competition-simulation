import pytest
import matplotlib.pyplot as plt
import numpy as np
from src.clone_competition_simulation.parameters import Algorithm, Parameters, TimeParameters, PopulationParameters, FitnessParameters


@pytest.fixture()
def simulation(mutation_generator):
    np.random.seed(0)
    parameters = Parameters(
        algorithm=Algorithm.MORAN2D,
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_cells=100, cell_in_own_neighbourhood=True), 
        fitness=FitnessParameters(mutation_generator=mutation_generator, mutation_rates=0.1)
    )
    sim = parameters.get_simulator()
    sim.run_sim()
    sim._get_colours(sim.clones_array)
    return sim


@pytest.fixture()
def simulation2(mutation_generator):
    """
    A simulation without mutations for certain plots. 
    """
    np.random.seed(0)
    parameters = Parameters(
        algorithm=Algorithm.MORAN2D,
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_grid=np.arange(100).reshape(10, 10), cell_in_own_neighbourhood=True), 
    )
    sim = parameters.get_simulator()
    sim.run_sim()
    sim._get_colours(sim.clones_array)
    return sim


@pytest.fixture()
def simulation3(mutation_generator):
    """
    A non-spatial simulation 
    """
    np.random.seed(0)
    parameters = Parameters(
        algorithm=Algorithm.MORAN,
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_size_array=[50, 50]), 
    )
    sim = parameters.get_simulator()
    sim.run_sim()
    sim._get_colours(sim.clones_array)
    return sim


@pytest.mark.mpl_image_compare
def test_muller_plot(simulation):
    fig, ax = plt.subplots()
    simulation.muller_plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_incomplete_moment(simulation):
    fig, ax = plt.subplots()
    simulation.plot_incomplete_moment(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_dnds(simulation):
    fig, ax = plt.subplots()
    simulation.plot_dnds(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_overall_population(simulation):
    fig, ax = plt.subplots()
    simulation.plot_overall_population(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_average_fitness_over_time(simulation):
    fig, ax = plt.subplots()
    simulation.plot_average_fitness_over_time(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_mean_clone_size_graph_for_non_mutation(simulation2):
    fig, ax = plt.subplots()
    simulation2.plot_mean_clone_size_graph_for_non_mutation(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_surviving_clones_for_non_mutation(simulation2):
    fig, ax = plt.subplots()
    simulation2.plot_surviving_clones_for_non_mutation(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_clone_size_distribution_for_non_mutation(simulation2):
    fig, ax = plt.subplots()
    simulation2.plot_clone_size_distribution_for_non_mutation(ax=ax)
    return fig

@pytest.mark.mpl_image_compare
def test_plot_clone_size_scaling_for_non_mutation(simulation2):
    fig, ax = plt.subplots()
    simulation2.plot_clone_size_scaling_for_non_mutation(ax=ax, times=[1, 5, 10])
    return fig


def test_animate1(simulation):
    simulation.animate("animation_output1.mp4")


def test_animate2(simulation3):
    simulation3.animate("animation_output2.mp4", grid_size=10)
