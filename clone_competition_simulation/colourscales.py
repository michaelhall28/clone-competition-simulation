import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from collections import namedtuple


class ColourMapError(Exception):
    pass


class ColourScale(object):
    """ColourScale for plotting clones and mutations"""
    possible_fields = ('label', 'ns', 'initial', 'last_mutated_gene', 'genes_mutated')

    def __init__(self, colourmaps, all_clones_noisy=False, name=None):
        self.fields = None  # Will remain None if there is a single colourmap for all clones
        # Otherwise will the order for fields of the dictionary keys
        self.colourmaps = colourmaps  # colourmap if only one for all clones, otherwise a dictionary
        self.check_colourmap_dict()
        self.all_noise = all_clones_noisy
        self.name = name  # For labelling the class

    def check_colourmap_dict(self):
        if type(self.colourmaps) == dict:
            self.is_dict = True
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

        if (not ns and not initial) or self.all_noise:  # Add a bit of randomness to see synonymous mutations
            noise = np.random.uniform(-0.1, 0.1)
        else:
            noise = 0
        return cs(fitness + noise)


def random_colour(rate):
    return cm.gist_ncar(np.random.random())


def random_colour_alt(rate):
    return cm.nipy_spectral(np.random.random())


##### Some examples
def get_colourscale_with_green_label_and_random_other(max_fitness):
    """Any type 1 clones are in green. Any type 0 clones are random non-green colours, apart from the original,
    which is beige.
    """
    Key = namedtuple('Key', ['clone_type', 'initial'])
    diff_ = max_fitness - 1
    cs = ColourScale(
        all_clones_noisy=False,
        colourmaps={
            Key(clone_type=1, initial=True): cm.Greens,
            Key(clone_type=1, initial=False): cm.ScalarMappable(norm=Normalize(vmin=1 - 5*diff_, vmax=max_fitness),
                                                           cmap=cm.Greens).to_rgba,
            Key(clone_type=0, initial=True): cm.ScalarMappable(norm=Normalize(vmin=0, vmax=3),
                                      cmap=cm.YlOrBr).to_rgba,
            Key(clone_type=0, initial=False): random_colour
            }
    )
    return cs


def get_random_colourscale(max_fitness):
    cs = ColourScale(
        all_clones_noisy=False,
        colourmaps=random_colour,
        name='random'
    )
    return cs


# An example colourscale which plots type 0 cells and type 1 cells as yellow/green
Key1 = namedtuple('Key1', ['label' ])
COLOURSCALE_EXAMPLE1 = ColourScale(
    name='Single Green Mutant',
    all_clones_noisy=True,
    colourmaps={Key1(label=0): cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.YlOrBr).to_rgba,
                Key1(label=1): cm.Greens
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