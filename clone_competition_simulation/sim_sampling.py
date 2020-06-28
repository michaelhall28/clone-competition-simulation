import pandas as pd
import numpy as np
from collections import defaultdict


def get_vafs_for_all_biopsies(sim, biopsies, coverage, detection_limit, merge_clones=False, sample_num=None,
                              binom=False, binom_params=(None, None), remove_initial_clones=True):
    ### Return as a dataframe
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
            biopsy_vafs = get_vafs(grid, sim, biopsy, coverage, detection_limit, binom=binom,
                                       binom_params=binom_params, remove_initial_clones=remove_initial_clones)
            for clone, vaf in biopsy_vafs.items():
                names.append(i)
                vafs.append(vaf)
                genes.append(sim.mutation_generator.genes[mutant_gene_map[clone]].name)
                clone_ids.append(clone)
    else:
        biopsy_vafs = get_vafs(grid, sim, None, coverage, detection_limit, binom=binom,
                               binom_params=binom_params, remove_initial_clones=remove_initial_clones)
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
        df = _apply_ns(df, sim)

    return df


def get_vafs(grid, sim, biopsy, coverage, detection_limit, binom=False,
             binom_params=(None, None), remove_initial_clones=True):
    clones = biopsy_sample(grid, sim, biopsy, remove_initial_clones=remove_initial_clones)

    if coverage is not None:
        vafs = small_detection_limit(clones, coverage, detection_limit, biopsy)
    elif binom:
        vafs = small_detection_limit_nbin(clones, binom_params, detection_limit, biopsy)
    else:
        vafs = {}
        A = (biopsy['biopsy_shape'][0] * biopsy['biopsy_shape'][1]) * 2
        for c, size in clones.items():
            vafs[c] = size / A

    return vafs


def biopsy_sample(grid, sim, biopsy, remove_initial_clones=True):
    # Assume the biopsies are square
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
    for i in biopsy.reshape(-1):
        mutants = get_mutants_from_clone_number(sim, i, remove_initial_clones=remove_initial_clones)
        for m in mutants:
            sample_counts[m] += 1

    return sample_counts


def small_detection_limit(clones, coverage, limit, biopsy):
    observed = {}
    A = (biopsy['biopsy_shape'][0] * biopsy['biopsy_shape'][1]) * 2
    for c, size in clones.items():
        vaf = size / A
        if vaf > 0.5:
            print('Got vaf of', vaf)
            print(c, size, A)
            print()
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def small_detection_limit_nbin(clones, binom_params, limit, biopsy):
    observed = {}
    A = (biopsy['biopsy_shape'][0] * biopsy['biopsy_shape'][1]) * 2
    for c, size in clones.items():
        vaf = size / A
        if vaf > 0.5:
            print('Got vaf of', vaf)
            print(c, size, A)
            print()
        coverage = np.random.negative_binomial(*binom_params)
        c_obs = np.random.binomial(coverage, vaf)
        if c_obs >= limit:
            observed[c] = c_obs / coverage

    return observed


def get_mutants_from_clone_number(sim, clone_number, remove_initial_clones=True):
    if remove_initial_clones:
        a = -2
    else:
        a = -1
    mutants = sim.get_clone_ancestors(clone_number)[:a]  # Remove root node and the initial clone (usually 0) if required
    return mutants


def _apply_ns(df, sim):
    df['ns'] = df.apply(lambda x: _is_ns(x['clone_id'], sim), axis=1)
    return df


def _is_ns(clone_id, sim):
    if clone_id in sim.ns_muts:
        return True
    else:
        return False


def get_dnds(observed_vafs, sim, gene=None):
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