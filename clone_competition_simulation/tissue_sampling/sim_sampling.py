"""
Functions to simulate the taking of biopsies and DNA sequencing
"""
import pandas as pd
import numpy as np
from collections import defaultdict


def get_vafs_for_all_biopsies(sim, biopsies, detection_limit=None, coverage=None, merge_clones=False, sample_num=None,
                              binom=False, binom_params=(None, None), remove_initial_clones=True,
                              heterozygous=True):
    """
    Simulate the sampling of a simulation using multiple biopsies.

    :param sim: A simulation object with the simulation completed.
    :param biopsies: A list of biopsies. Each biopsy is defined using a dictionary
    {'biopsy_origin': (x, y), 'biopsy_shape': (w, h)}
    :param coverage: Int. The depth of the simulated sequencing. Will be used if not None. If coverage is None and
    binom=False, will assume perfect sampling of all clones.
    :param detection_limit: Int. The lower limit of detection. I.e. mutant must appear in at least this number of
    simulated reads to be counted as detected.
    :param merge_clones: Bool. If True, will add the clone sizes if the same clone is detected in two different biopsies.
    If False, will count the same clone in two different biopsies as two separate clones.
    :param sample_num: Int. The index of the sample time from the simulation. By default, uses the final time point.
    :param binom: If True, and coverage is None, uses a negative binomial distribution to randomly simulate the
    sequencing depth instead of using a fixed coverage value.
    :param binom_params: The parameters (tuple) for numpy.random.negative_binomial if binom=True
    :param remove_initial_clones: If True, will remove any clones present at the start of the simulation from the
    final results.
    :param heterozygous: If True, will assume every mutation is on one of two copies of the chromosome. So a mutant in
    the entire biopsy will have a VAF of 0.5 instead of 1. If False, will assume the mutants are all homozygous.
    To make other assumptions (such as a different copy number per gene), run the biopsy_sample function instead to
    get the raw counts of mutant cells, which can then be processed further with the assumptions you need.
    :return: A pandas data frame, with sample, VAF, gene and clone id for each "detected" clone.
    """
    if sample_num is None:
        sample_num = -1
    grid = sim.grid_results[sample_num]

    mutant_gene_map = {i: int(clone[sim.gene_mutated_idx]) for i, clone in enumerate(sim.clones_array)}

    names = []
    vafs = []
    genes = []
    clone_ids = []

    if biopsies is not None:
        for i, biopsy in enumerate(biopsies):
            biopsy_vafs = get_vafs(grid, sim, biopsy, detection_limit=detection_limit, coverage=coverage, binom=binom,
                                   binom_params=binom_params, remove_initial_clones=remove_initial_clones,
                                   heterozygous=heterozygous)
            for clone, vaf in biopsy_vafs.items():
                names.append(i)
                vafs.append(vaf)
                genes.append(sim.mutation_generator.genes[mutant_gene_map[clone]].name)
                clone_ids.append(clone)
    else:
        biopsy_vafs = get_vafs(grid, sim, None, detection_limit=detection_limit, coverage=coverage, binom=binom,
                               binom_params=binom_params, remove_initial_clones=remove_initial_clones,
                               heterozygous=heterozygous)
        for clone, vaf in biopsy_vafs.items():
            names.append(0)
            vafs.append(vaf)
            genes.append(sim.mutation_generator.genes[mutant_gene_map[clone]].name)
            clone_ids.append(clone)

    if merge_clones:
        clones = {}
        for biopsy, vaf, gene, clone_id in zip(names, vafs, genes, clone_ids):
            if clone_id not in clones:
                clones[clone_id] = {'vaf': vaf, 'gene': gene, 'clone_id': clone_id}
            else:
                clones[clone_id]['vaf'] += vaf

        df = pd.DataFrame.from_dict(list(clones.values()))
    else:
        df = pd.DataFrame({
            'sample': names,
            'vaf': vafs,
            'gene': genes,
            'clone_id': clone_ids
        })

    if len(df) > 0:
        # Mark each mutation as synonymous or non-synonymous
        df = _apply_ns(df, sim)

    return df


def get_vafs(grid, sim, biopsy, detection_limit, coverage=None, binom=False,
             binom_params=(None, None), remove_initial_clones=True, heterozygous=True):
    """
    Simulate the DNA sequencing of a single biopsy.

    :param grid: Simulated grid of cells.
    :param sim: The simulation object associated with the grid.
    :param biopsy: Dictionary defining the position and size of the biopsy.
    {'biopsy_origin': (x, y), 'biopsy_shape': (w, h)}
    :param coverage: Int. The depth of the simulated sequencing. Will be used if not None. If coverage is None and
    binom=False, will assume perfect sampling of all clones.
    :param detection_limit: Int. The lower limit of detection. I.e. mutant must appear in at least this number of
    simulated reads to be counted as detected.
    :param binom: If True, and coverage is None, uses a negative binomial distribution to randomly simulate the
    sequencing depth instead of using a fixed coverage value.
    :param binom_params: The parameters (tuple) for numpy.random.negative_binomial if binom=True
    :param remove_initial_clones: If True, will remove any clones present at the start of the simulation from the
    final results.
    :param heterozygous: If True, will assume every mutation is on one of two copies of the chromosome. So a mutant in
    the entire biopsy will have a VAF of 0.5 instead of 1. If False, will assume the mutants are all homozygous.
    To make other assumptions (such as a different copy number per gene), run the biopsy_sample function instead to
    get the raw counts of mutant cells, which can then be processed further with the assumptions you need.
    :return: dictionary of clone_id: VAF
    """
    clones = biopsy_sample(grid, sim, biopsy, remove_initial_clones=remove_initial_clones)
    vaf_denominator = _get_vaf_denominator(biopsy, heterozygous, sim)

    if coverage is not None:
        vafs = small_detection_limit(clones, coverage, detection_limit, vaf_denominator)
    elif binom:
        vafs = small_detection_limit_nbin(clones, binom_params, detection_limit, vaf_denominator)
    else:
        vafs = {}
        for c, size in clones.items():
            vafs[c] = size / vaf_denominator

    return vafs


def biopsy_sample(grid, sim, biopsy, remove_initial_clones=True):
    """
    Find all number of cells in each mutant clone in the biopsy.
    :param grid: Simulated grid of cells.
    :param sim: The simulation object associated with the grid.
    :param biopsy: Dictionary defining the position and size of the biopsy.
    {'biopsy_origin': (x, y), 'biopsy_shape': (w, h)}
    :param remove_initial_clones: If True, will remove any clones present at the start of the simulation from the
    final results.
    :return: Dictionary,  clone_id: cell_count
    """
    if biopsy is not None:
        x, y = biopsy['biopsy_origin']
        edge_x, edge_y = biopsy['biopsy_shape']
        x2, y2 = x + edge_x, y + edge_y
        assert x2 <= grid.shape[0], (x2, grid.shape[0])
        assert y2 <= grid.shape[1], (y2, grid.shape[1])
        biopsy = grid[x:x2, y:y2]
    else:
        biopsy = grid

    sample_counts = defaultdict(int)
    for clone_id, cell_count in zip(*np.unique(biopsy, return_counts=True)):
        mutants = get_mutants_from_clone_number(sim, clone_id, remove_initial_clones=remove_initial_clones)
        for m in mutants:
            sample_counts[m] += cell_count

    return sample_counts


def small_detection_limit(clones, coverage, limit, vaf_denominator):
    """
    Converts clone size in cell number to a noisy VAF, assuming a fixed sequencing depth and binomial sampling.
    :param clones: Dictionary of clone_id: cell_count
    :param coverage: Int. Sequencing depth.
    :param limit: Int. Lower limit of detection.
    :param vaf_denominator: Float or Int. Number used to divide clone cell count in order to convert to VAF.
    :return: Dictionary. clone_id: VAF
    """
    observed = {}
    for c, size in clones.items():
        vaf = size / vaf_denominator
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def small_detection_limit_nbin(clones, binom_params, limit, vaf_denominator):
    """
    Converts clone size in cell number to a noisy VAF, assuming a negative binomial sequencing depth
    and binomial sampling.
    :param clones: Dictionary of clone_id: cell_count
    :param binom_params: Tuple. Parameters for numpy.random.negative_binomial for the sequencing depth.
    :param limit: Int. Lower limit of detection.
    :param vaf_denominator: Float or Int. Number used to divide clone cell count in order to convert to VAF.
    :return: Dictionary. clone_id: VAF
    """
    observed = {}
    for c, size in clones.items():
        vaf = size / vaf_denominator
        coverage = np.random.negative_binomial(*binom_params)
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def _get_vaf_denominator(biopsy, heterozygous, sim):
    """
    Get the total number of copies of the chromosomes in the sample, i.e. the copy number multiplied by the cell number
    :param heterozygous: Assumes the mutant is on one of two copies of the chromosome
    :return: Int.
    """
    if biopsy is None:
        d = sim.total_pop
    else:
        d = biopsy['biopsy_shape'][0] * biopsy['biopsy_shape'][1]   # The number of cells in the biopsy

    if heterozygous:
        # Double the "chromosome count" if heterozygous
        d *= 2

    return d


def get_mutants_from_clone_number(sim, clone_number, remove_initial_clones=True):
    """
    Each clone may contain many mutants. This returns a list of the mutations present in a particular clone (defined
    using the clone_id of the clone that they formed when they first appeared).
    :param sim: The simulation.
    :param clone_number: The clone_id.
    :param remove_initial_clones: If True, will remove any clones present at the start of the simulation from the
    final results.
    :return: List of int.
    """
    if remove_initial_clones:
        a = -2
    else:
        a = -1
    mutants = sim.get_clone_ancestors(clone_number)[:a]  # Remove root node and the initial clone (usually 0) if required
    return mutants


def _apply_ns(df, sim):
    """
    Adds a column to the dataframe of detected mutations to mark each as synonymous (False) or non-synonymous (True).
    :param df:
    :param sim:
    :return:
    """
    df['ns'] = df.apply(lambda x: _is_ns(x['clone_id'], sim), axis=1)
    return df


def _is_ns(clone_id, sim):
    if clone_id in sim.ns_muts:
        return True
    else:
        return False


def get_sample_dnds(observed_vafs, sim, gene=None):
    """
    Calculates a dN/dS ratio for a set of "observed" mutations.
    :param observed_vafs: Dataframe of mutations (the output of the get_vafs_for_all_biopsies function).
    :param sim: The simulation object.
    :param gene: The gene to calculate dN/dS for. If None, will use all VAFs regardless of gene.
    :return:
    """
    if gene is not None:
        observed_vafs = observed_vafs[observed_vafs['gene'] == gene]
    ns = 0
    s = 0

    for clone in observed_vafs['clone_id']:
        if clone in sim.ns_muts:
            ns += 1
        else:
            s += 1

    expected_ns = s * (1 / sim.mutation_generator.get_synonymous_proportion(
        sim.mutation_generator.get_gene_number(gene)
    ) - 1)
    try:
        dnds = ns / expected_ns
        return dnds
    except ZeroDivisionError as e:
        return np.nan
