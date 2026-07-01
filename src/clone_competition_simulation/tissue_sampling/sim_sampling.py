"""
Functions to simulate the taking of biopsies and DNA sequencing
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator

from ..simulation_algorithms.base_2D_class import BaseHexagonalGridSim


class Biopsy(BaseModel):
    """Describe a rectangular biopsy region within a simulation grid.

    Parameters
    ----------
    origin : tuple[int, int]
        The zero-based (x, y) coordinate of the top-left corner of the biopsy in the
        simulation grid.
    shape : tuple[int, int]
        The width and height of the biopsy rectangle in grid units.

    Attributes
    ----------
    origin : tuple[int, int]
        The validated biopsy origin coordinates.
    shape : tuple[int, int]
        The validated biopsy dimensions.
    size : int
        The total number of grid cells contained in the biopsy (width * height).

    Notes
    -----
    The class validates that origin coordinates are non-negative and that shape
    dimensions are positive.
    """
    origin: tuple[int, int]
    shape: tuple[int, int]

    @field_validator("origin", mode="after")
    @classmethod
    def validate_origin(cls, v) -> tuple[int, int]:
        if v[0] < 0 or v[1] < 0:
            raise ValueError('Biopsy origin coordinates must be non-negative')
        return v

    @field_validator("shape", mode="after")
    @classmethod
    def validate_shape(cls, v) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError('Biopsy dimensions must be positive')
        return v

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    def slice_grid(self, grid: NDArray) -> NDArray:
        """Returns the section of the grid covered by this biopsy

        Parameters
        ----------
        grid : NDArray
            The simulation grid from which to extract the biopsy region. 2D NumPy array. 

        Returns
        -------
        NDArray
            The sliced grid corresponding to the biopsy region. 2D NumPy array.
        """
        x2, y2 = self.origin[0] + self.shape[0], self.origin[1] + self.shape[1]
        assert x2 <= grid.shape[0], (x2, grid.shape[0])
        assert y2 <= grid.shape[1], (y2, grid.shape[1])
        return grid[self.origin[0]:x2, self.origin[1]:y2]



def get_vafs_for_all_biopsies(sim: BaseHexagonalGridSim, biopsies: list[Biopsy],
                              detection_limit: int | None=None, coverage: int | None=None, merge_clones: bool=False,
                              sample_num: int | None=None, binom_params: tuple[int, float] | None=None,
                              remove_initial_clones: bool=True, heterozygous: bool=True) -> pd.DataFrame:
    """Simulate the sampling of a simulation using multiple biopsies.

    Parameters
    ----------
    sim : BaseHexagonalGridSim
        A completed simulation object.
    biopsies : list[Biopsy] | None
        The list of Biopsy objects defining biopsy position and size. If None, the entire grid is sampled.
    detection_limit : int | None
        Lower limit of detection; a mutant must appear in at least this many simulated reads to be counted.
    coverage : int | None
        Sequencing depth for fixed-coverage sampling. If None and ``binom_params`` is also None, perfect sampling is assumed.
    merge_clones : bool
        If True, merge detections of the same clone across multiple biopsies by clone id and gene.
    sample_num : int | None
        Index of the sample time from the simulation. Defaults to the final time point.
    binom_params : tuple[int, float] | None
        Parameters for ``numpy.random.negative_binomial`` when using negative-binomial sequencing depth.
    remove_initial_clones : bool
        If True, remove clones present at the start of the simulation from the output.
    heterozygous : bool
        If True, assume mutations are heterozygous, so a mutation present in all biopsy cells has maximum VAF 0.5.

    Returns
    -------
    pd.DataFrame
        Data frame containing sample index, VAF, gene name, and clone id for each detected clone.
    """
    if sample_num is None:
        sample_num = -1
    grid = sim.grid_results[sample_num]

    names = []
    vafs = []
    genes = []
    clone_ids = []

    if biopsies is not None:
        for i, biopsy in enumerate(biopsies):
            biopsy_vafs = get_vafs(grid, sim, biopsy, detection_limit=detection_limit, coverage=coverage,
                                   binom_params=binom_params, remove_initial_clones=remove_initial_clones,
                                   heterozygous=heterozygous)
            for clone, vaf in biopsy_vafs.items():
                names.append(i)
                vafs.append(vaf)
                genes.append(
                    sim.fitness_calculator.get_gene_name(
                        sim.clones_array[clone, sim.gene_mutated_idx]
                    )
                )
                clone_ids.append(clone)
    else:
        biopsy_vafs = get_vafs(grid, sim, None, detection_limit=detection_limit, coverage=coverage,
                               binom_params=binom_params, remove_initial_clones=remove_initial_clones,
                               heterozygous=heterozygous)
        for clone, vaf in biopsy_vafs.items():
            names.append(0)
            vafs.append(vaf)
            genes.append(
                sim.fitness_calculator.get_gene_name(
                    sim.clones_array[clone, sim.gene_mutated_idx]
                )
            )
            clone_ids.append(clone)

    df = pd.DataFrame({
        'sample': names,
        'vaf': vafs,
        'gene': genes,
        'clone_id': clone_ids
    })
    if merge_clones:
        df = df.groupby(['gene', 'clone_id']).agg('sum')
        df = df.drop('sample', axis=1)
        df = df.reset_index()

    if len(df) > 0:
        # Mark each mutation as synonymous or non-synonymous
        df['ns'] = df['clone_id'].isin(sim.ns_muts)

    return df


def get_vafs(grid: NDArray, sim: BaseHexagonalGridSim, biopsy: Biopsy | None, detection_limit: int | None=None,
             coverage: int | None=None, binom_params: tuple[int, float] | None=None,
             remove_initial_clones: bool=True, heterozygous: bool=True) \
        -> dict[int, float]:
    """Simulate the DNA sequencing of a single biopsy.

    Parameters
    ----------
    grid : NDArray
        Simulated grid of cells.
    sim : BaseHexagonalGridSim
        The simulation object associated with the grid.
    biopsy : Biopsy | None
        Biopsy object defining the position and size of the biopsy. If None, the entire grid is used.
    detection_limit : int | None
        Lower limit of detection; a mutant must appear in at least this many simulated reads to be counted.
    coverage : int | None
        Sequencing depth for fixed-coverage sampling. If None and ``binom_params`` is None, perfect sampling is used.
    binom_params : tuple[int, float] | None
        Parameters for ``numpy.random.negative_binomial`` when using negative-binomial sampling. Set to use variable coverage 
        instead of fixed coverage.
    remove_initial_clones : bool
        If True, remove clones present at the start of the simulation from the final result.
    heterozygous : bool
        If True, assume mutations are heterozygous, giving a maximum VAF of 0.5 for fully occupied biopsy cells. 
        If False, all mutants are assumed to be homozygous. 
        To make other assumptions, run the biopsy_sample function to get raw counts of mutant cells and then apply your own VAF conversion.

    Returns
    -------
    dict[int, float]
        Mapping from clone id to observed VAF.
    """
    clones = biopsy_sample(grid, sim, biopsy, remove_initial_clones=remove_initial_clones)
    vaf_denominator = _get_vaf_denominator(biopsy, heterozygous, sim)

    if coverage is not None:
        vafs = small_detection_limit(clones, coverage, detection_limit, vaf_denominator)
    elif binom_params is not None:
        vafs = small_detection_limit_nbin(clones, binom_params, detection_limit, vaf_denominator)
    else:
        vafs = {}
        for c, size in clones.items():
            vafs[c] = size / vaf_denominator

    return vafs


def biopsy_sample(grid: NDArray, sim: BaseHexagonalGridSim, biopsy: Biopsy | None, remove_initial_clones: bool=True) \
        -> dict[int, int]:
    """Count cells in each mutant clone within a biopsy sample.

    Parameters
    ----------
    grid : NDArray
        Simulated grid of cells.
    sim : BaseHexagonalGridSim
        The simulation object associated with the grid.
    biopsy : Biopsy | None
        Biopsy object defining the position and size of the biopsy. If None, the full grid is sampled.
    remove_initial_clones : bool
        If True, remove clones present at the start of the simulation from the final results.

    Returns
    -------
    dict[int, int]
        Mapping from clone id to cell count in the biopsy.
    """
    if biopsy is not None:
        biopsy_grid = biopsy.slice_grid(grid)
    else:
        biopsy_grid = grid

    sample_counts = defaultdict(int)
    for clone_id, cell_count in zip(*np.unique(biopsy_grid, return_counts=True)):
        mutants = get_mutants_from_clone_number(sim, clone_id, remove_initial_clones=remove_initial_clones)
        for m in mutants:
            sample_counts[m] += cell_count

    return sample_counts


def small_detection_limit(clones: dict[int, int], coverage: int, limit: int, vaf_denominator: float) -> dict[int, float]:
    """Convert clone cell counts to observed VAF using fixed coverage and binomial sampling.

    Parameters
    ----------
    clones : dict[int, int]
        Mapping from clone id to cell count.
    coverage : int
        Sequencing depth.
    limit : int
        Lower limit of detection; clones with fewer observed reads are excluded.
    vaf_denominator : float
        Value used to convert clone cell count into an expected VAF.

    Returns
    -------
    dict[int, float]
        Mapping from clone id to observed VAF for detected clones.
    """
    observed = {}
    for c, size in clones.items():
        vaf = size / vaf_denominator
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def small_detection_limit_nbin(clones: dict[int, int], binom_params: tuple[int, float], limit: int,
                               vaf_denominator: float) -> dict[int, float]:
    """Convert clone cell counts to observed VAF using negative binomial coverage and binomial sampling.

    Parameters
    ----------
    clones : dict[int, int]
        Mapping from clone id to cell count.
    binom_params : tuple[int, float]
        Parameters for ``numpy.random.negative_binomial`` that define sequencing depth variability.
    limit : int
        Lower limit of detection; clones with fewer observed reads are excluded.
    vaf_denominator : float
        Value used to convert clone cell count into an expected VAF.

    Returns
    -------
    dict[int, float]
        Mapping from clone id to observed VAF for detected clones.
    """
    observed = {}
    for c, size in clones.items():
        vaf = size / vaf_denominator
        coverage = np.random.negative_binomial(*binom_params)
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def _get_vaf_denominator(biopsy: Biopsy | None, heterozygous: bool, sim: BaseHexagonalGridSim) -> int:
    """Return the total number of copies of chromosomes in the sample, e.g. the copy number multiplied by the number of cells. 
    This serves as the denominator for calculating VAFs.

    Parameters
    ----------
    biopsy : Biopsy | None
        Biopsy sample defining the region size. If None, the full simulation population is used.
    heterozygous : bool
        If True, account for two chromosome copies per cell.
    sim : BaseHexagonalGridSim
        The simulation object from which to obtain population size.

    Returns
    -------
    int
        Total number of chromosome copies in the sample.
    """
    if biopsy is None:
        d = sim.total_pop
    else:
        d = biopsy.size   # The number of cells in the biopsy

    if heterozygous:
        # Double the "chromosome count" if heterozygous
        d *= 2

    return d


def get_mutants_from_clone_number(sim: BaseHexagonalGridSim, clone_number: int, remove_initial_clones: bool=True) \
        -> list[int]:
    """Return the mutant ids present in a clone.

    Parameters
    ----------
    sim : BaseHexagonalGridSim
        The simulation object.
    clone_number : int
        Clone id to inspect.
    remove_initial_clones : bool
        If True, exclude mutations present in initial clones from the returned list.

    Returns
    -------
    list[int]
        List of mutant ids present in the specified clone.
    """
    if remove_initial_clones:
        a = -2
    else:
        a = -1
    mutants = sim.get_clone_ancestors(clone_number)[:a]  # Remove root node and the initial clone (usually 0) if required
    return mutants


def get_sample_dnds(observed_vafs: pd.DataFrame, sim: BaseHexagonalGridSim, gene: str | None=None) -> float:
    """Calculate dN/dS for observed mutations in a sample.

    Parameters
    ----------
    observed_vafs : pd.DataFrame
        Data frame of observed mutations, typically the output of ``get_vafs_for_all_biopsies``.
    sim : BaseHexagonalGridSim
        The simulation object used to determine synonymous mutation proportions.
    gene : str | None
        Name of the gene for which to compute dN/dS. If None, include all genes.

    Returns
    -------
    float
        The dN/dS ratio, or ``numpy.nan`` if the expected synonymous count is zero.
    """
    if gene is not None:
        observed_vafs = observed_vafs[observed_vafs['gene'] == gene]

    ns = observed_vafs['clone_id'].isin(sim.ns_muts).sum()
    s = len(observed_vafs) - ns

    if gene is not None:
        gene_number = sim.fitness_calculator.get_gene_number(gene)
    else:
        gene_number = None

    expected_ns = s * (1 / sim.fitness_calculator.get_synonymous_proportion(
        gene_number
    ) - 1)
    try:
        dnds = ns / expected_ns
        return dnds
    except ZeroDivisionError as e:
        return np.nan
