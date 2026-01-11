"""
Classes to control the colours used for Muller plots, plots the 2D grids and animations.
"""

import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from collections import namedtuple


class ColourMapError(Exception):
    pass


class ColourScale(object):
    """ColourScale for plotting clones and mutations"""
    possible_fields = ('label', 'ns', 'initial', 'last_mutated_gene', 'genes_mutated')

    def __init__(self, colourmaps, all_clones_noisy=False, name=None, use_fitness=False):
        """
        This establishes the rules for the colours of clones in the Muller plots, grid plots and animations of the
        simulations.
        Clones can be categorised using the labels, non-synonymous/synonymous, whether they existed at the start of
        the simulation of appeared later, and on the genes mutated in the clone.
        The colour can also depend on the clone fitness
        :param colourmaps: Either a function that takes a single number (the fitness) as an input and returns a colour
        (for example a matplotlib colormap), or a dictionary where the keys are namedtuples defining the clone category
        and the values are the functions from fitness to colour. See the tutorials or the examples in
        clone_competition_simulation.colourscales.
        :param all_clones_noisy: Will add a small amount of noise to the fitness so that different clones with the same
        fitness can be distinguished.
        :param name: For labelling the object
        :param use_fitness: The fitness will be passed to the colormap. Higher fitness will return higher value colours.
        """
        self.fields = None  # Will remain None if there is a single colourmap for all clones
        # Otherwise will be the order for fields of the dictionary keys

        self.colourmaps = colourmaps  # colourmap if only one for all clones, otherwise a dictionary
        self.check_colourmap_dict()
        self.all_noise = all_clones_noisy
        self.use_fitness = use_fitness
        self.name = name

    def check_colourmap_dict(self):
        if type(self.colourmaps) == dict:
            self.fields = next(iter(self.colourmaps))._fields
            for f in self.fields:
                if f not in self.possible_fields:
                    raise ColourMapError('Field {0} not one of the possible options: {1}'.format(f,
                                                                                                 self.possible_fields))
            for k in self.colourmaps.keys():
                if k._fields != self.fields:
                    raise ColourMapError('Keys in the colourmap are not consistent')

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            if self.fields is not None:
                return 'Indexed by {0}'.format(self.fields)
            else:
                return 'Single scale for all clones'

    def _get_colour(self, fitness, label, ns, initial, last_mutated_gene, genes_mutated):
        if self.fields is not None:
            # make the correct key and get the assigned colourmap for the clone type.
            key = []
            for f in self.fields:
                if f == 'label':
                    key.append(label)
                elif f == 'ns':
                    key.append(ns)
                elif f == 'initial':
                    key.append(initial)
                elif f == 'last_mutated_gene':
                    key.append(last_mutated_gene)
                elif f == 'genes_mutated':
                    key.append(genes_mutated)
            key = tuple(key)
            cs = self.colourmaps[key]
        else:
            cs = self.colourmaps

        if not self.use_fitness:  # Just use a random number to vary the colours given
            return cs(np.random.random())
        else:
            # Otherwise use fitness, with or without some noise.
            if self.all_noise:  # Add a bit of randomness to see synonymous mutations
                noise = np.random.uniform(-0.1, 0.1)
            else:
                noise = 0
            return cs(fitness + noise)


def random_colour(rate):
    return cm.gist_ncar(np.random.random())


def random_colour_alt(rate):
    return cm.nipy_spectral(np.random.random())


##### Some examples

def get_CS_random_colours_from_colourmap(colourmap):
    """
    Each clone will be given a random colour from the given colourmap.
    :param colourmap: a matplotlib colormap.
    :return:
    """

    def f(fitness):
        """This ignores the fitness and just returns a random colour from the colourmap"""
        return colourmap(np.random.random())

    cs = ColourScale(
        all_clones_noisy=False,
        colourmaps=f,
        name=colourmap.name
    )
    return cs


def get_default_random_colourscale():
    cs = get_CS_random_colours_from_colourmap(cm.gist_ncar)
    cs.name = 'random'
    return cs


# An example colourscale which plots label 0 cells as beige, label 1 cells as yellow/green, and label 2 cells as purple
Key1 = namedtuple('Key1', ['label' ])
COLOURSCALE_EXAMPLE1 = ColourScale(
    name='Single Green Mutant',
    all_clones_noisy=True,
    colourmaps={Key1(label=0): cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.YlOrBr).to_rgba,
                Key1(label=1): cm.Greens,
                Key1(label=2): cm.Purples
         }
)

# An example colourscale which plots non-synonymous clones as red and synonymous as blue
# Initial clones are beige
Key2 = namedtuple('Key2', ['ns', 'initial'])
COLOURSCALE_EXAMPLE2 = ColourScale(
    name='S vs NS',
    all_clones_noisy=True,
    colourmaps={Key2(ns=True, initial=False): cm.Reds,
         Key2(ns=False, initial=False ): cm.Blues,
         Key2(ns=False, initial=True): cm.ScalarMappable(norm=Normalize(vmin=0, vmax=3),
                                      cmap=cm.YlOrBr).to_rgba
         }
)