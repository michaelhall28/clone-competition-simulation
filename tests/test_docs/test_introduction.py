"""
Check that the code from the documentation runs.  

Not checking simulation results (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
import matplotlib.pyplot as plt
from clone_competition_simulation import Parameters, PopulationParameters, TimeParameters, FitnessParameters

def test_introduction_example():
    np.random.seed(0)

    p = Parameters(
        algorithm='Moran',  # We will run a non-spatial Moran simulation. 
        times=TimeParameters(
            max_time=25,  # Run for 25 time units
            division_rate=1.4  # Set average division rate for all cells to 1.4 per time unit
        ), 
        population=PopulationParameters(  # Define the cell population
            initial_size_array=[100, 100, 100]  # There are three initial clones, with 100 cells in each
        ),
        fitness=FitnessParameters(  # Define the cell fitness
            initial_fitness_array=[1, 1.02, 1.04]    # Each clone has a different fitness value
        )
    )
    
    assert p.show_progress is True

    sim = p.get_simulator()
    sim.run_sim()

    sim2 = p.get_simulator()
    sim2.run_sim()

    assert not np.all(
       sim.population_array.toarray() == sim2.population_array.toarray()
    )
    
    sim3 = p.get_simulator()
    np.random.seed(0)
    sim3.run_sim()

    sim4 = p.get_simulator()
    np.random.seed(0)
    sim4.run_sim()


    np.testing.assert_array_equal(sim3.population_array.toarray(), 
                                         sim4.population_array.toarray())


    sim2.view_clone_info()

    sim.muller_plot(figsize=(5, 5))

    sim.population_array.toarray()

    sim.times

    sim.plot_average_fitness_over_time()

    
def test_introduction_example2():
    from rich.progress import track

    p = Parameters(
        algorithm='Moran', 
        times=TimeParameters(
            max_time=25,
            division_rate=1.4 
        ), 
        population=PopulationParameters( 
            initial_size_array=[100, 100, 100] 
        ),
        fitness=FitnessParameters( 
            initial_fitness_array=[1, 1.02, 1.04]   
        ), 
        show_progress=False  # Set this to hide the progress bar
    )

    # Tracking the progress of the entire set of simulations
    for i in track(range(5), description="Running simulations..."):
        np.random.seed(i)
        s = p.get_simulator()
        s.run_sim()