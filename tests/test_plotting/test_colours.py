import matplotlib.cm as cm
import pytest
import numpy as np

from clone_competition_simulation.plotting.plot_colours import (
    DEFAULT_BACKGROUND_COLOUR_RULE, DEFAULT_COLOUR_RULE, CloneFeature,
    ColourRule, FeatureValue, PlotColourMaps)


def test_colour_rule():
    rule1 = ColourRule(
        rule_filter=[
            FeatureValue(
                clone_feature=CloneFeature.LABEL, 
                value=1
            ),
        ], 
        colourmap=lambda x: x 
    )

    assert rule1.apply_filter(label=1)
    assert rule1.apply_filter(label=1.0)
    assert not rule1.apply_filter(label=2)
    assert not rule1.apply_filter(label="1")


def test_colour_scale():
    
    rules = [
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.LABEL, 
                    value=1
                ), 
                FeatureValue(
                    clone_feature=CloneFeature.INITIAL, 
                    value=True
                )
            ], 
            colourmap=cm.Reds
        ), 
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.LABEL, 
                    value=2
                ), 
                FeatureValue(
                    clone_feature=CloneFeature.INITIAL, 
                    value=False
                )
            ], 
            colourmap=cm.Blues
        ), 
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.LABEL, 
                    value=2
                ), 
                FeatureValue(
                    clone_feature=CloneFeature.NS, 
                    value=True
                )
            ], 
            colourmap=cm.Greens
        )
    ]

    colourscale = PlotColourMaps(
        colour_rules=rules
    )

    # Doesn't match any rule. Should be the default background
    cm1 = colourscale._get_colourmap(
        label=0, ns=False, 
        initial=True, last_mutated_gene=None, genes_mutated=set())
    assert cm1 == DEFAULT_BACKGROUND_COLOUR_RULE.colourmap

    # Matches the first rule
    cm2 = colourscale._get_colourmap(
        label=1, ns=False, 
        initial=True, last_mutated_gene=None, genes_mutated=set())
    assert cm2 == cm.Reds

    # Matches the second rule
    cm3 = colourscale._get_colourmap(
        label=2, ns=False, 
        initial=False, last_mutated_gene=None, genes_mutated=set())

    assert cm3 == cm.Blues

    # Matches the 2nd and 3rd rule. Should return the colour from the 2nd
    cm4 = colourscale._get_colourmap(
        label=2, ns=True, 
        initial=False, last_mutated_gene=None, genes_mutated=set())
    assert cm4 == cm.Blues

    # Matches the 3rd rule
    cm5 = colourscale._get_colourmap(
        label=2, ns=True, 
        initial=True, last_mutated_gene=None, genes_mutated=set())
    assert cm5 == cm.Greens

    

def test_colour_scale2():
    
    rules = [
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.GENES_MUTATED, 
                    value=set()
                ),
            ], 
            colourmap=cm.Reds
        ), 
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.GENES_MUTATED, 
                    value={"Gene1"}
                ),
            ], 
            colourmap=cm.Blues
        ), 
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.GENES_MUTATED, 
                    value="Gene2"
                ),
            ], 
            colourmap=cm.Purples
        ), 
        ColourRule(
            rule_filter=[
                FeatureValue(
                    clone_feature=CloneFeature.GENES_MUTATED, 
                    value={"Gene1", "Gene2"}
                ),
            ], 
            colourmap=cm.Greens
        )
    ]

    colourscale = PlotColourMaps(
        colour_rules=rules
    )

    # No genes mutated. Matches first rule
    cm1 = colourscale._get_colourmap(
        label=0, ns=False, 
        initial=False, last_mutated_gene=None, genes_mutated=set())
    assert cm1 == cm.Reds

    # Matches the first rule
    cm2 = colourscale._get_colourmap(
        label=0, ns=False, 
        initial=False, last_mutated_gene=None, genes_mutated={"Gene1"})
    assert cm2 == cm.Blues

    # Matches the second rule
    cm3 = colourscale._get_colourmap(
        label=0, ns=False, 
        initial=False, last_mutated_gene=None, genes_mutated={"Gene2"})
    assert cm3 == cm.Purples

    # Matches the second rule
    cm4 = colourscale._get_colourmap(
        label=0, ns=False, 
        initial=False, last_mutated_gene=None, genes_mutated={"Gene1", "Gene2"})
    assert cm4 == cm.Greens


def test_colour_scale3():
    colourscale = PlotColourMaps()
    assert len(colourscale.colour_rules) == 1
    assert colourscale.colour_rules[0] == DEFAULT_COLOUR_RULE
    

@pytest.mark.parametrize(
    "colour,expected", [
        ('red', (1.0, 0.0, 0.0, 1.0)),
        ('#ffbb11', (1.0, 0.7333333333333333, 0.06666666666666667, 1.0)),
        ('tab:green', (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)),
    ] 
)
def test_colour_conversion(colour, expected):
    rules = [
        ColourRule(
            rule_filter=[
            ], 
            colourmap=lambda x: colour
        )
        
    ]

    colourscale = PlotColourMaps(
        colour_rules=rules
    )
    output_colour = colourscale._get_colour(
        fitness=1, label=0, ns=True, initial=True, 
        last_mutated_gene=None, genes_mutated=set()
    )
    np.testing.assert_almost_equal(
        output_colour, expected
    )