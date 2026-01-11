from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.clone_competition_simulation.tissue_sampling.sim_sampling import (
    Biopsy,
    get_vafs_for_all_biopsies,
    get_sample_dnds,
    get_vafs,
    get_mutants_from_clone_number,
    biopsy_sample,
    small_detection_limit_nbin,
    small_detection_limit,
    _get_vaf_denominator
)


@dataclass
class MockGene:
    name: str

@dataclass
class MockMutationGenerator:
    genes: list[MockGene]

    @staticmethod
    def get_synonymous_proportion(gene_number: int) -> float:
        return 0.5

    @staticmethod
    def get_gene_number(gene: str) -> int:
        return 0


@dataclass()
class MockSim:
    grid: np.ndarray
    total_pop: int
    mutation_generator: MockMutationGenerator
    ns_muts: set[int]
    grid_results: np.ndarray
    clones_array: np.ndarray
    gene_mutated_idx = 0

    @staticmethod
    def get_clone_ancestors(clone_number: int) -> np.ndarray:
        # Mock, and say the clone lineage is simply 0->1->2->3->4 etc.
        # Includes the root node (-1) and initial clone.
        return np.arange(clone_number, -2, -1)



@pytest.fixture
def grid():
    return np.arange(100).reshape((10, 10))

@pytest.fixture
def sim(grid):
    return MockSim(
        grid=grid, total_pop=grid.size, grid_results= np.array([grid]),
        clones_array=np.tile(np.arange(4), 25).reshape(100, 1),
        mutation_generator=MockMutationGenerator(
            genes=[MockGene(name="A"), MockGene(name="B"), MockGene(name="C"), MockGene(name="D")],
        ),
        ns_muts=set(range(0, 100, 4)),
    )


@pytest.fixture
def clones():
    return {
        i: i for i in range(1, 11)
    }


def test_biopsies():
    with pytest.raises(ValueError) as excinfo:
        Biopsy(origin=(-1, 0), shape=(1, 1))
        assert "Biopsy origin coordinates must be non-negative" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        Biopsy(origin=(0, -1), shape=(1, 1))
        assert "Biopsy origin coordinates must be non-negative" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        Biopsy(origin=(0, 0), shape=(0, 0))
        assert "Biopsy dimensions must be positive" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        Biopsy(origin=(0, 0), shape=(-1, 0))
        assert "Biopsy dimensions must be positive" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        Biopsy(origin=(0, 0), shape=(0, -1))
        assert "Biopsy dimensions must be positive" in str(excinfo)



def test_get_vafs_for_all_biopsies(sim):
    biopsies = [
        Biopsy(origin=(0, 0), shape=(2, 2)),
        Biopsy(origin=(2, 1), shape=(1, 1)),
    ]
    vafs1 = get_vafs_for_all_biopsies(sim, biopsies, merge_clones=False)
    expected = pd.DataFrame(
        {
            'sample': [0]*11 + [1]* 21,
            'vaf': [0.375] + [0.25]*9 + [0.125] + [0.5] * 21,
            'gene': ['B'] + ['C', 'B', 'A', 'D'] * 2 + ['C', 'D', 'B'] + ['A', 'D', 'C', 'B'] * 5,
            'clone_id': np.concatenate([[1], np.arange(10, 1, -1), [11], np.arange(21, 0, -1)]),
            'ns': [False, False, False, True] * 2 + [False] + [False, False, False, True] * 5 +
                  [False, False, False]
        }
    )
    assert vafs1.equals(expected)

def test_get_vafs_for_all_biopsies2(sim):
    biopsies = [
        Biopsy(origin=(0, 0), shape=(2, 2)),
        Biopsy(origin=(2, 1), shape=(1, 1)),
    ]
    vafs1 = get_vafs_for_all_biopsies(sim, biopsies, merge_clones=True)
    expected = pd.DataFrame(
        {
            'gene': ['A'] * 5 + ['B'] * 6 + ['C'] * 5 + ['D'] * 5,
            'clone_id': np.concatenate([np.arange(4, 24, 4), np.arange(1, 25, 4), np.arange(2, 22, 4),
                                        np.arange(3, 23, 4)]),
            'vaf': [0.75, 0.75, 0.5, 0.5, 0.5, 0.875, 0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75,
                    0.5, 0.5, 0.75, 0.75, 0.625, 0.5, 0.5],
            'ns': [True] * 5 + [False] * 16
        }
    )

    pd.testing.assert_frame_equal(vafs1, expected)


@pytest.mark.parametrize("biopsy,detection_limit,coverage,binom_params,remove_initial,het,expected", [
    (Biopsy(origin=(0, 0), shape=(2, 2)),0,100,None,False,True,
     {0:0.51, 1:0.385, 10:0.26, 9:0.26, 8:0.26, 7:0.26, 6:0.26, 5:0.26, 4:0.26, 3:0.26, 2:0.26, 11:0.135}),
    (Biopsy(origin=(0, 0), shape=(2, 2)),None,None,None,False,True,{
    0:0.5, 1:0.375, 10:0.25, 9:0.25, 8:0.25, 7:0.25, 6:0.25, 5:0.25, 4:0.25, 3:0.25, 2:0.25, 11:0.125
    }),
    (Biopsy(origin=(0, 0), shape=(2, 2)),3,10,(10, 0.1),True,False,{
        1:0.85, 10:0.6, 9:0.6, 8:0.6, 7:0.6, 6:0.6, 5:0.6, 4:0.6, 3:0.6, 2:0.6, 11:0.35
    }),
])
def test_get_vafs(monkeypatch, grid, sim, biopsy, detection_limit, coverage, binom_params,
                  remove_initial, het, expected):
    with monkeypatch.context() as m:
        m.setattr(np.random, "binomial", lambda n, p: n * p + 1)
        m.setattr(np.random, "negative_binomial", lambda n, p: n)
        assert get_vafs(
            grid, sim, biopsy, detection_limit,
            coverage, binom_params, remove_initial, het) == expected

@pytest.mark.parametrize("biopsy,remove_initial,expected", [
    (Biopsy(origin=(0, 0), shape=(2, 2)), False, {0:4, 1:3, 2:2, 3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:1}),
    (Biopsy(origin=(0, 0), shape=(2, 2)), True, {1:3, 2:2, 3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:1}),
    (None, False, {i:100-i for i in range(100)}),
    (None, True, {i:100-i for i in range(1,100)}),
])
def test_biopsy_sample(grid, sim, biopsy, remove_initial, expected):
    assert biopsy_sample(
        grid, sim, biopsy=biopsy, remove_initial_clones=remove_initial
    ) == expected

def test_small_detection_limit(clones, monkeypatch: pytest.MonkeyPatch):
    expected = {
        i: (i + 1) / 10 for i in range(2, 11)
    }
    with monkeypatch.context() as m:
        m.setattr(np.random, "binomial", lambda n, p: n * p + 1)
        assert small_detection_limit(
            clones, coverage=10,
            limit=3,
            vaf_denominator=10,
        ) == expected


def test_small_detection_limit_nbin(clones, monkeypatch: pytest.MonkeyPatch):
    expected = {
        i: (i+1)/10 for i in range(2, 11)
    }
    with monkeypatch.context() as m:
        m.setattr(np.random, "negative_binomial", lambda n, p: n)
        m.setattr(np.random, "binomial", lambda n, p: n * p + 1)
        assert small_detection_limit_nbin(
            clones, binom_params=(10, 0.1),
            limit=3,
            vaf_denominator=10,
        ) == expected


@pytest.mark.parametrize("biopsy,het,expected", [
    (None, False, 100),
    (None, True, 200),
    (Biopsy(origin=(0, 0), shape=(1, 1)), False, 1),
    (Biopsy(origin=(0, 0), shape=(1, 1)), True, 2),
    (Biopsy(origin=(0, 0), shape=(10, 5)), False, 50),
    (Biopsy(origin=(0, 0), shape=(10, 5)), True, 100),
])
def test_get_vaf_denominator(sim, biopsy, het, expected):
    denom = _get_vaf_denominator(sim=sim, biopsy=biopsy, heterozygous=het)
    assert denom == expected



@pytest.mark.parametrize("clone_number,remove_initial,expected", [
    (0, True, np.array([])),
    (1, True, np.array([1])),
    (10, True, np.arange(10, 0, -1)),
    (0, False, np.array([0])),
    (1, False, np.array([1, 0])),
    (10, False, np.arange(10, -1, -1)),
])
def test_get_mutants_from_clone_number(sim, clone_number, remove_initial, expected):
    np.testing.assert_equal(
        get_mutants_from_clone_number(
            sim=sim, clone_number=clone_number, remove_initial_clones=remove_initial),
        expected
    )


def test_get_sample_dnds(sim):
    vafs = pd.DataFrame(
        {
            "clone_id": np.arange(10),
            "gene": ["A"] * 5 + ["B"] * 5,
        }
    )
    np.testing.assert_almost_equal(
        get_sample_dnds(vafs, sim=sim), 0.42857142857142855
    )
    np.testing.assert_almost_equal(
        get_sample_dnds(vafs, sim=sim, gene="A"),
        0.6666666666666666
    )
    np.testing.assert_almost_equal(
        get_sample_dnds(vafs, sim=sim, gene="B"),
        0.25
    )

