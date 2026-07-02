"""
Classes to control the colours used for Muller plots, plots the 2D grids and animations.
"""

from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Iterable, Self

import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from pydantic import BaseModel, Field, field_validator


def _convert_to_string_set(value: str | Iterable) -> set[str]:
    """Convert a string or iterable of strings to a set of strings

    For matching sets of gene names

    Args:
        value (str | Iterable): Gene name or names

    Returns:
        set[str]: Set of gene names
    """

    if isinstance(value, str):
        return {value}
    else:
        return set(value)


class CloneFeature(Enum):
    """Enum for clones features used to assign colours

    Each member has a value and a function to convert the feature 
    into the consistent type for matching. 
    """
    LABEL = 'label', float
    NS = 'ns', bool
    INITIAL = 'initial', bool
    LAST_MUTATED_GENE = 'last_mutated_gene', str
    GENES_MUTATED = 'genes_mutated', _convert_to_string_set

    def __new__(cls, value, type_converter: Callable[[Any], Any]) -> Self:
        """Apply the type_converter function to the enum members

        Parameters
        ----------
        value : 
            The enum value
        type_converter : Callable[[Any], Any]
            A function to convert a type

        Returns
        -------
        Self
            the member with the type_converter function
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj.type_converter = type_converter
        return obj


class FeatureValue(BaseModel):
    """Combines a feature (CloneFeature) with a value.

    Used to match clone features to a value. 
    E.g. if clone_feature=CloneFeature.LABEL and the value=1, 
    then clones with the label 1 will match, and all other clones
    won't. 
    """
    clone_feature: CloneFeature
    value: Any

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, v, info):
        """Convert the input value to the required format.  

        The type conversion will raise an error if an incompatible type is used as input
        """
        clone_feature = info.data['clone_feature']
        return clone_feature.type_converter(v)


class ColourRule(BaseModel):
    """Rule for matching clones with a colourmap

    Parameters
    ----------
    rule_filter: list[FeatureValue]
        List of feature values for a clone to match to apply this rule. 
    colourmap: Callable[[float], Any]
        A function that converts a float to a format matplotlib can interpret as a colour
    """
    rule_filter: list[FeatureValue] = Field(default_factory=list)
    colourmap: Callable[[float], Any] 

    def apply_filter(self, **kwargs) -> bool:   
        """Check that the information about the clone matches this set of filters

        Parameters
        ----------
        kwargs: dict[str, Any]
            Dictionary of feature names and values. Used to compare to the rule_filter. 
            Any features in the kwargs that are not in the rule_filter will be ignored. 

        Returns
        -------
        bool
            False if the rules exclude the clone. True otherwise.
        """
        for key, value in kwargs.items():
            for rule in self.rule_filter:
                if rule.clone_feature.value == key:
                    if value != rule.value:
                        return False
        return True


DEFAULT_COLOUR_RULE = ColourRule(
    rule_filter=[],   # No filters, i.e. apply to all clones. 
    colourmap=cm.gist_ncar   # Apply colours at random.
)


DEFAULT_BACKGROUND_COLOUR_RULE = ColourRule(
    rule_filter=[],  # No filters, i.e. apply to all clones. 
    # Use a range of beige colours to help other colours stand out. 
    colourmap=cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.YlOrBr).to_rgba
)


def default_noise_fn() -> float:
    return np.random.uniform(-0.1, 0.1)


class PlotColourMaps:
    """Assigns the colours for plotting clones and mutations"""

    def __init__(self, colour_rules: list[ColourRule] | None = None, 
                 all_clones_noisy=False, use_fitness=False, 
                 random_noise_fn: Callable[[], float]=default_noise_fn):
        """This establishes the rules for the colours of clones in the Muller plots, grid plots and animations of the
        simulations.

        Clones can be categorised using the labels, non-synonymous/synonymous, whether they existed at the start of
        the simulation of appeared later, and on the genes mutated in the clone.
        The colour can also depend on the clone fitness

        :param use_fitness: If true, clones with a higher fitness will return higher value colours. If False, fitness will not affect the 
         colour for the clone. 

        Parameters
        ----------
        colour_rules : list[ColourRule] | None, optional
            Am optional list of ColourRules. These will be applied in order. If a clone matches the criteria in the rule_filter, then the 
            colourmap from that rule will be applied. If not, the next colour rule will be checked. See the docs for examples. 
            If None, the DEFAULT_COLOUR_RULE will be applied. 
        all_clones_noisy : bool, optional
            If True, will add a small amount of noise to the fitness so that different clones with the same fitness can be distinguished. 
            Will not apply if use_fitness=False because in that case colours are already randomly drawn. 
            Warning: if clones have similar fitness values, using all_clone_noisy=True may break the higher-fitness -> higher colour 
            on the colourmap relationship. By default False.
        use_fitness : bool, optional
            If true, clones with a higher fitness will return higher value colours from the colourmap. If False, fitness will not affect the 
            colour for the clone. By default False.
        random_noise_fn : Callable[[], float], optional
            Function used to apply random noise to the clone fitness if all_clones_noisy=True and use_fitness=True, by default default_noise_fn.
        """
        if colour_rules is None:
            colour_rules = [DEFAULT_COLOUR_RULE]
        else:
            colour_rules.append(DEFAULT_BACKGROUND_COLOUR_RULE)

        # Add the default colour rule to the end of list to pick up any clones not otherwise assigned colours
        self.colour_rules = colour_rules
        self.all_noise = all_clones_noisy
        self.random_noise_fn = random_noise_fn
        self.use_fitness = use_fitness

    def _get_colour(self, fitness: float, label: float, ns: bool, 
                    initial: bool, last_mutated_gene: str | None, 
                    genes_mutated: set[str]) -> Any:
        """Gets a colour for a clone with the input arguments. 

        Finds the colourmap that matches the clone features, then gets a colour
        from that colourmap.

        Parameters
        ----------
        fitness : float
            The clone fitness. Can be used to assign the colour if self.use_fitness=True
        label : float
            The clone label
        ns : bool
            Whether the clone was formed by a non-synonymous mutation
        initial : bool
            Whether the clone existed from the start of the simulation
        last_mutated_gene : str, optional
            The name of the gene of the mutation which formed the clone
        genes_mutated : set[str]
            Set of gene names of genes mutated in the clone

        Returns
        -------
        Any
            Colour (as recognised by Matplotlib)
        """

        cs = self._get_colourmap(
            label, ns, initial, last_mutated_gene, genes_mutated
        )

        # Find the input value to pass to the colour function
        if not self.use_fitness:  # Just use a random number to vary the colours given
            value = np.random.random()
        else:
            # Otherwise use fitness, with or without some noise.
            if self.all_noise:  # Add a bit of randomness to see synonymous mutations
                noise = self.random_noise_fn()
            else:
                noise = 0
            value = fitness + noise
        
        return cs(value)
    
    def _get_colourmap(self, label: float, ns: bool, 
                       initial: bool, last_mutated_gene: str | None, 
                       genes_mutated: set[str]) -> Callable[[float], Any]:
        """Find the first colour rule that matches the clone features and return its colourmap

        Parameters
        ----------
        label : float
            The clone label
        ns : bool
            Whether the clone was formed by a non-synonymous mutation
        initial : bool
            Whether the clone existed from the start of the simulation
        last_mutated_gene : str | None
            The name of the gene of the mutation which formed the clone
        genes_mutated : set[str]
            Set of gene names of genes mutated in the clone

        Returns
        -------
        Callable[[float], Any]
            Function which takes a float as an argument and returns a colour 
            (in a format recognised by Matplotlib)

        Raises
        ------
        ValueError
            If no colour rule matches the clone features
        """
        
        # Select the colour rule that applies to this clone
        for rule in self.colour_rules:
            if rule.apply_filter(label=label, ns=ns, initial=initial, 
                              last_mutated_gene=last_mutated_gene, 
                              genes_mutated=genes_mutated):
                return rule.colourmap
        
        # Should not get here! The defaults should pick up all clones. 
        raise ValueError("Failed to find a colour map matching the clone features")
    

# An example colourscale which plots label 0 cells as beige, label 1 cells as yellow/green, and label 2 cells as purple
PLOT_COLOURS_EXAMPLE1 = PlotColourMaps(
    all_clones_noisy=True,
    colour_rules=[
        ColourRule(
            rule_filter=[FeatureValue(clone_feature=CloneFeature.LABEL, value=0)],
            colourmap=cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=cm.YlOrBr).to_rgba
        ), 
        ColourRule(
            rule_filter=[FeatureValue(clone_feature=CloneFeature.LABEL, value=1)],
            colourmap=cm.Greens
        ),
        ColourRule(
            rule_filter=[FeatureValue(clone_feature=CloneFeature.LABEL, value=2)],
            colourmap=cm.Purples
        )
    ]
)