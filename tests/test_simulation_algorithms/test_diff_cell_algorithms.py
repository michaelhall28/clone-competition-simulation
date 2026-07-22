import numpy as np
import pytest

from src.clone_competition_simulation import (
    Parameters, 
    FitnessParameters, 
    TimeParameters, 
    PopulationParameters,
    FitnessCalculator, 
    LabelParameters, 
    TreatmentParameters, 
    DifferentiatedCellsParameters, 
    Gene, 
    EpistaticEffect, 
    FixedValue
)
from tests.test_fitness.test_fitness import epistatics

@pytest.fixture
def fitness_calculator():
    genes = [
        Gene(name='1', mutation_distribution=FixedValue(1), 
             synonymous_proportion=0.5), 
        Gene(name='2', mutation_distribution=FixedValue(2), 
             synonymous_proportion=0.4)
    ]
    epistatics = [
        EpistaticEffect(
            name='epi1', 
            gene_names=['1', '2'], 
            fitness_distribution=FixedValue(3)
        )
    ]
    return FitnessCalculator(
        genes=genes, 
        epistatics=epistatics, 
        multi_gene_array=True
    )

def test_labels():
    p = Parameters(
        algorithm="Moran", 
        population=PopulationParameters(
            initial_cells=100
        ), 
        times=TimeParameters(max_time=10, division_rate=1, samples=11), 
        labels=LabelParameters(
            label_times=[2],
            label_frequencies=[0.1],
            label_values=[1]
        ), 
        differentiated_cells=DifferentiatedCellsParameters(
            r=0.1, gamma=0.9, stratification_sim_proportion=0.6
        )
    )
    np.random.seed(0)
    sim = p.get_simulator()
    sim.run_sim()
    
    # Check that the diff cell population has been extended
    assert sim.diff_cell_population.shape == (10, 12)


def test_labels2(fitness_calculator):
    p = Parameters(
        algorithm="Moran", 
        population=PopulationParameters(
            initial_cells=100
        ), 
        times=TimeParameters(max_time=10, division_rate=1, samples=11), 
        fitness=FitnessParameters(
            fitness_calculator=fitness_calculator, 
            mutation_rates=0.01
        ),
        labels=LabelParameters(
            label_times=[2],
            label_frequencies=[0.1],
            label_values=[1]
        ), 
        differentiated_cells=DifferentiatedCellsParameters(
            r=0.1, gamma=0.9, stratification_sim_proportion=0.6
        )
    )
    np.random.seed(0)
    sim = p.get_simulator()
    sim.run_sim()

    assert sim.diff_cell_population.shape == (21, 12)
