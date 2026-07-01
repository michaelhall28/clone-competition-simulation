"""
Check that the code from the documentation runs.  

Not checking simulation results in detail (that is done in other tests), 
just that changes to the code have not broken the documentation examples
"""
import numpy as np
from clone_competition_simulation import Parameters, TimeParameters, PopulationParameters
from clone_competition_simulation import biopsy_sample
from clone_competition_simulation import Biopsy
from clone_competition_simulation import Gene, FitnessCalculator, NormalDist, FitnessParameters
from clone_competition_simulation import FixedValue
import matplotlib.pyplot as plt
from clone_competition_simulation import get_vafs_for_all_biopsies
from clone_competition_simulation import get_sample_dnds


def test_simulating_biopsies():
    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D', 
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_grid=np.arange(100).reshape(10, 10), 
                                        cell_in_own_neighbourhood=False)
    )
    s = p.get_simulator()
    s.run_sim()

    # The final clone ids on the grid
    s.grid_results[-1]

    biopsy_sample(s.grid_results[-1], s, biopsy=None, remove_initial_clones=False)

    biopsy = Biopsy(
        origin=(3, 4),  # The grid coordinates of the first corner of the biopsy
        shape=(5, 3)  # The lengths of the sides of the biopsy
    )
    s.grid_results[-1][3:3+5, 4:4+3]

    biopsy_sample(s.grid_results[-1], s, biopsy=biopsy, remove_initial_clones=False)


def test_simulating_biopsies2():

    fit_calc = FitnessCalculator(
        genes=[Gene(name="Gene1", mutation_distribution=NormalDist(0.1), synonymous_proportion=0.5)],
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='Moran2D',
        times=TimeParameters(max_time=10, division_rate=1),
        population=PopulationParameters(initial_cells=100, 
                                        cell_in_own_neighbourhood=False), 
        fitness=FitnessParameters(
            mutation_rates=0.1, 
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()

    s.grid_results[-1]

    s.grid_results[-1][3:3+5, 4:4+3]

    biopsy = Biopsy(
        origin=(3, 4),  # The grid coordinates of the first corner of the biopsy
        shape=(5, 3)  # The lengths of the sides of the biopsy
    )

    biopsy_sample(s.grid_results[-1], s, biopsy, remove_initial_clones=False)

    biopsy_sample(s.grid_results[-1], s, biopsy)

    for clone_id in [21, 29, 30]:
        s.get_clone_descendants(clone_id)


def test_simulating_biopsies3():
    fit_calc = FitnessCalculator(
        genes=[
            Gene(name='Gene1', mutation_distribution=FixedValue(1.2), synonymous_proportion=0.5), 
            Gene(name='Gene2', mutation_distribution=FixedValue(1.4), synonymous_proportion=0.5), 
            Gene(name='Gene3', mutation_distribution=FixedValue(0.7), synonymous_proportion=0.5)
        ], 
        combine_mutations='add'
    )

    np.random.seed(0)
    p = Parameters(
        algorithm='WF2D',
        times=TimeParameters(max_time=200, division_rate=1),
        population=PopulationParameters(initial_cells=10000, 
                                        cell_in_own_neighbourhood=False), 
        fitness=FitnessParameters(
            mutation_rates=0.01, 
            fitness_calculator=fit_calc
        )
    )
    s = p.get_simulator()
    s.run_sim()

    exact_counts = biopsy_sample(s.grid_results[-1], s, biopsy=None)

    plt.hist(exact_counts.values(), bins=np.arange(0, max(exact_counts.values())+1, 50));
    plt.ylabel('Frequency')
    plt.xlabel('Cell count')

    vafs = get_vafs_for_all_biopsies(s, biopsies=None, heterozygous=False)

    plt.hist(vafs['vaf'], 
             bins=np.linspace(0, 0.5, 50));
    plt.xlabel('Variant allele fraction')
    plt.ylabel('Frequency')

    sequenced_vafs = get_vafs_for_all_biopsies(s, biopsies=None, detection_limit=5, coverage=100)

    biopsy = Biopsy(origin=(30, 40),  shape=(50, 30))

    get_vafs_for_all_biopsies(s, biopsies=[biopsy], detection_limit=5, coverage=100, heterozygous=False)

    get_vafs_for_all_biopsies(s, biopsies=[biopsy], detection_limit=5, coverage=100, heterozygous=True)


    biopsies = [
        Biopsy(origin=(0, 0), shape=(20, 20)),
        Biopsy(origin=(20, 0), shape=(20, 20)),
        Biopsy(origin=(0, 20), shape=(20, 20)),
        Biopsy(origin=(20, 20), shape=(20, 20))
    ]

    get_vafs_for_all_biopsies(s, biopsies, detection_limit=5, coverage=100)

    get_vafs_for_all_biopsies(s, biopsies, detection_limit=5, coverage=100, merge_clones=True)

    s.times[10]

    get_vafs_for_all_biopsies(s, biopsies, detection_limit=5, coverage=100, sample_num=10)

    get_vafs_for_all_biopsies(s, biopsies, detection_limit=5, binom_params=(100, 0.5))



    vafs = get_vafs_for_all_biopsies(s, biopsies=None)
    get_sample_dnds(vafs, s)
    for i in range(1, 4):
        gene = f"Gene{i}"
        get_sample_dnds(vafs, s, gene=gene)

    vafs = get_vafs_for_all_biopsies(s, biopsies=biopsies, detection_limit=4, coverage=500)
    get_sample_dnds(vafs, s)
    for i in range(1, 4):
        gene = f"Gene{i}"
        get_sample_dnds(vafs, s, gene=gene)
