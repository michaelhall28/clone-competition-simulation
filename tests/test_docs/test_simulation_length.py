"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import matplotlib.pyplot as plt
import numpy as np

from clone_competition_simulation import (Parameters, PopulationParameters,
                                          TimeParameters)


def test_simulation_length():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=100)
    )


def test_simulation_length1():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=20, division_rate=0.5),
        population=PopulationParameters(initial_cells=100)
    )


def test_simulation_length2():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(1000))
    )  
    s = p.get_simulator()
    s.run_sim()

    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, division_rate=1.7),
        population=PopulationParameters(initial_size_array=np.ones(1000))
    )  
    s2 = p.get_simulator()
    s2.run_sim()

    s.plot_mean_clone_size_graph_for_non_mutation(
        show_spm_fit=False, legend_label=1)
    s2.plot_mean_clone_size_graph_for_non_mutation(
        ax=plt.gca(), show_spm_fit=False, legend_label=1.7)
    plt.legend(title='Division rate')


    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(simulation_steps=15000, division_rate=1),
        population=PopulationParameters(initial_size_array=np.ones(1000))
    )
    s15 = p.get_simulator()
    s15.run_sim()

    s.plot_mean_clone_size_graph_for_non_mutation(
        show_spm_fit=False, legend_label='Max time=10')
    s15.plot_mean_clone_size_graph_for_non_mutation(
        ax=plt.gca(), show_spm_fit=False, legend_label='15000 steps')
    plt.legend()


def test_simulation_length3():
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(times=[2,10,30], division_rate=1.6),
        population=PopulationParameters(initial_size_array=np.ones(1000))
    )
    s = p.get_simulator()
    s.run_sim()
    s.plot_surviving_clones_for_non_mutation()


def test_simulation_length4():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(max_time=10, samples=5, division_rate=1),
        population=PopulationParameters(initial_size_array=[500, 500])
    )
    s = p.get_simulator()
    s.run_sim()

    # There are 5 samples plus the initial condition. 
    assert len(s.times) == 6
    assert s.population_array.toarray().shape == (2, 6)


def test_simulation_length5():
    p = Parameters(
        algorithm='WF', 
        times=TimeParameters(samples=500, max_time=10, division_rate=1),
        population=PopulationParameters(initial_size_array=[500, 500])
    )
    s = p.get_simulator()
    s.run_sim()

    # There are 10 samples plus the initial condition. 
    assert len(s.times) == 11
    assert s.population_array.toarray().shape == (2, 11)


def test_simulation_length6():
    p = Parameters(
        algorithm='WF', 
        times=TimeParameters(times=[2.3, 4.3, 8.7], division_rate=1),
        population=PopulationParameters(initial_size_array=[500, 500])
    )
    s = p.get_simulator()
    np.testing.assert_array_equal(s.times, [2, 4, 9])


def test_simulation_length7():
    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(times=[2.3, 4.3, 8.7], division_rate=1),
        population=PopulationParameters(initial_size_array=[500, 500])
    )
    s = p.get_simulator()
    np.testing.assert_array_equal(s.times, [2.3, 4.3, 8.7])
