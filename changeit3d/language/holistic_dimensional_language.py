"""
Analyze the dataset w.r.t. to language that refers to basic dimensions: length, width, height.
"""

dimensional_adjectives = dict()
dimensional_adjectives['tall'] = ('tall -er', 'tall -est')
dimensional_adjectives['short'] = ('short -er', 'short -est')
dimensional_adjectives['wide'] = ('wide -er', 'wide -est')
dimensional_adjectives['long'] = ('long -er', 'long -est')
dimensional_adjectives['deep'] = ('deep -er', 'deep -est')
dimensional_adjectives['shallow'] = ('shallow -er', 'shallow -est')



def holistic_dimensional_expressions(adjective, comparative, superlative, noun_set=None):
    """
    Generate simple expressions that reflect dimensional reasoning: it is bigger
    :param adjective: an adjective, e.g,, tall
    :param comparative: a comparative form of the adjective, e.g., taller
    :param superlative: a superlative form of the adjective, e.g., tallest
    :param noun_set: optional, set of noun words that will be used in expressions: e.g., {table}
    :return: set
    Example:
        holistic_dimensional_expressions(tall, taller, tallest, noun_set={table, stool})
        ->

    """
    expressions = set()
    for x in [adjective, comparative, superlative]:
        expressions.update([f'{x}',
                            f'is {x}',
                            f'is the {x}',
                            f'is the {x} one',
                            f'it is {x}',
                            f'it is the {x}',
                            f'it is the {x} one',
                            f'the {x}',
                            f'the {x} one',
                            f'the target is {x}',
                            f'the target is the {x}',
                            f'the target is the {x} one',
                            f'the distractor is {x}',
                            f'the distractor is the {x}',
                            f'the distractor is the {x} one',
                            ])

    x = adjective
    expressions.update([f'more {x}', f'less {x}', f'most {x}', f'least {x}'])

    wise_ending_expressions = []
    for e in expressions:
        wise_ending_expressions.extend(
            [f'{e} depthwise',
             f'{e} in depth',
             f'{e} heightwise',
             f'{e} in height',
             f'{e} lengthwise',
             f'{e} in length',
             f'{e} widthwise',
             f'{e} in width',
             f'{e} widthways',
             f'{e} lengthways'
             ])
    expressions.update(wise_ending_expressions)

    noun_expressions = []
    if noun_set is not None:
        for noun in noun_set:
            for x in [adjective, comparative, superlative]:
                noun_expressions.extend([f'{x} {noun}',
                                         f'the {x} {noun}',
                                         f'is the {x} {noun}',
                                         f'it is the {x} {noun}',
                                         f'the {noun} is {x}',
                                         f'the {noun} is the {x}'
                                         ])

    expressions.update(noun_expressions)
    return expressions


def holistic_expressions_mask(df, utterance_column, adjectives_comp_sup=None, use_object_class_nouns=False, verbose=True):
    total_fraction = 0.0
    total_mask = None

    if adjectives_comp_sup is None:
        adjectives_comp_sup = dimensional_adjectives

    if use_object_class_nouns:
        object_class_nouns = df.source_object_class.unique().tolist()
        object_class_nouns.extend(['tub', 'bath', 'stool', 'shelf']) # TODO, add more synonyms.
        noun_set = object_class_nouns
    else:
        noun_set = None


    for key, values in adjectives_comp_sup.items():
        adjective = key
        comparative, superlative = values
        expressions = holistic_dimensional_expressions(adjective, comparative, superlative,
                                                       noun_set=noun_set)

        mask = df[utterance_column].apply(lambda x: x in expressions)
        mu = mask.mean().round(4)
        total_fraction += mu
        if verbose:
            print(adjective, mu)

        if total_mask is None:
            total_mask = mask
        else:
            total_mask |= mask
    if verbose:
        print('Overall', total_fraction)
    return total_mask