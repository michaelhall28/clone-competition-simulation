import numpy as np
from clone_competition_simulation import (
    Parameters,
    TimeParameters,
    PopulationParameters,
    FitnessParameters,
    Gene,
    MutationGenerator,
    FixedValue,
    NormalDist,
    pickle_load,
    Algorithm
)
from tests.utilities import compare_single_res, convert_sim_to_standard_form


def test_pickling():
    genes = [Gene(name='neutral', mutation_distribution=FixedValue(1), synonymous_proportion=0.5),
             Gene(name='mild_driver', mutation_distribution=FixedValue(1.1), synonymous_proportion=0.5),
             Gene(name='random_driver', mutation_distribution=NormalDist(mean=1.01, var=0.05),
                  synonymous_proportion=0.8),
    ]
    mut_gen1 = MutationGenerator(multi_gene_array=False, genes=genes, combine_mutations='add')

    np.random.seed(0)
    p = Parameters(
        algorithm=Algorithm.MORAN,
        population=PopulationParameters(initial_cells=10000),
        times=TimeParameters(max_time=20, division_rate=1),
        fitness=FitnessParameters(mutation_rates=0.01, mutation_generator=mut_gen1)
    )

    s = p.get_simulator()
    s.run_sim()
    s.pickle_dump("test_pickling.pickle")

    s2 = pickle_load("test_pickling.pickle")

    s_dict = convert_sim_to_standard_form(s, Algorithm.MORAN)
    s2_dict = convert_sim_to_standard_form(s2, Algorithm.MORAN)

    for k, v in s_dict.items():
        compare_single_res(v, s2_dict[k])


