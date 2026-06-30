"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import matplotlib.cm as cm
import numpy as np

from clone_competition_simulation import (FitnessCalculator, FitnessParameters,
                                          Gene, Parameters,
                                          PopulationParameters, TimeParameters,
                                          UniformDist)


def test_animation():
    np.random.seed(2)
    initial_grid = np.concatenate([np.zeros(5000), np.ones(5000)]).reshape(100, 100)
    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=False)
    )
    s = p.get_simulator()

    s.animate('outfile1.mp4', figsize=(3, 3), dpi=300, bitrate=1000, fps=5)

    s.animate(
        'outfile2.mp4', figsize=(3, 3), dpi=300, bitrate=1000, fps=5, 
        fixed_label_text='My simulation', fixed_label_loc=(10, 40), 
        fixed_label_kwargs={'fontsize': 14, 'fontweight': 3}, 
        show_time_label=True, time_label_units='days', time_label_loc=(70, 5), 
        time_label_kwargs={'fontsize': 10, 'color': 'r'}, time_label_decimal_places=1
    )


def test_animation2():
    np.random.seed(2)
    initial_grid = np.concatenate([np.zeros(5000), np.ones(5000)]).reshape(100, 100)
    fit_calc = FitnessCalculator(
        genes=[Gene(name='Gene1', mutation_distribution=UniformDist(1, 1.5), synonymous_proportion=0)], 
        combine_mutations='add'
    )
    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(max_time=10, division_rate=1), 
        population=PopulationParameters(initial_grid=initial_grid, cell_in_own_neighbourhood=False),
        fitness=FitnessParameters(fitness_calculator=fit_calc, mutation_rates=0.1)
    )
    s = p.get_simulator()
    s.run_sim()

    s.animate('outfile3.mp4', fitness=True, fitness_cmap=cm.plasma, min_fitness=1,
            figsize=(4, 3), dpi=300, bitrate=1000, fps=5)


def test_animation3():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran',
        times=TimeParameters(max_time=5, division_rate=1), 
        population=PopulationParameters(initial_size_array=np.full(4, 4))
    )
    s = p.get_simulator()
    s.run_sim()

    s.animate('outfile4.mp4', grid_size=16,  
            figsize=(3, 3), dpi=100, bitrate=200, fps=5)
