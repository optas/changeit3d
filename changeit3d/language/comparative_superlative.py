NLTK_COMP_SUP_SYMBOLS = ['JJS', 'RBS', 'JJR', 'RBR']

irregular_com_sup = {
    'best': ('good', 'est'),
    'better': ('good', 'er'),
    'least': ('less',  'est'),
    'farthest': ('far', 'est'),
    'farther': ('far', 'er'),
    'further': ('far', 'er'),
    'furthest': ('far', 'est'),
    'fatter': ('fat', 'er'),
    'fattest': ('fat', 'est'),
    'flatter': ('flat', 'er'),
    'flattest': ('flat', 'est'),
    'slimmer': ('slim', 'er'),
    'slimmest': ('slim', 'est'),
    'bigger': ('big', 'er'),
    'biggest': ('big', 'est'),
    'thinnest': ('thin', 'est'),
    'thinner': ('thin', 'er'),
    'funnest': ('fun', 'est'),
    'funniest': ('fun', 'est'),
    'littlest': ('little', 'est'),
    'littler': ('little', 'er'),
    }

# E.g., close -> closer, thus you can't simply remove -er, -est to find the stem
adjectives_ending_in_e = {'close', 'simple', 'square', 'strange', 'wide', 'large',
                          'dense', 'loose', 'nearer', 'simple', 'square'}


sup_comp_found_via_nltk_in_our_data = {
 'best',
 'better',
 'bigger',
 'biggest',
 'brighter',
 'broader',
 'broadest',
 'cleaner',
 'closer',
 'closest',
 'deeper',
 'deepest',
 'denser',
 'farther',
 'farthest',
 'fatter',
 'fattest',
 'fewer',
 'flatter',
 'fuller',
 'further',
 'greater',
 'greatest',
 'harder',
 'harsher',
 'higher',
 'highest',
 'larger',
 'largest',
 'leaner',
 'least',
 'lesser',
 'lighter',
 'longer',
 'longest',
 'looser',
 'lower',
 'lowest',
 'narrower',
 'narrowest',
 'nearer',
 'nearest',
 'older',
 'rougher',
 'rounder',
 'shallower',
 'sharper',
 'shorter',
 'shortest',
 'simpler',
 'sleeker',
 'slimmer',
 'slower',
 'smaller',
 'smallest',
 'smoother',
 'softer',
 'squarer',
 'squatter',
 'steeper',
 'stouter',
 'straighter',
 'stronger',
 'subtler',
 'taller',
 'tallest',
 'thicker',
 'thickest',
 'thinner',
 'thinnest',
 'tighter',
 'weaker',
 'wider',
 'widest',
 'lengthier',
 'pointier',
 'glossier',
 'bulkier',
 'boxier',
 'bumpier',
 'bushier',
 'chubbier',
 'chunkier',
 'clunkier',
 'crazier',
 'curvier',
 'dirtier',
 'easier',
 'emptier',
 'fancier',
 'shinier',
 'skinnier',
 'skinniest',
 'stockier',
 'sturdier',
 'tinier',
 'fluffier',
 'happier',
 'heavier',
 'heftier'
 'messier',
 'roomier',
}


def break_down_superlative_comparatives(tokens, all_conversions):
    new_tokens = []
    for token in tokens:
        if token in sup_comp_found_via_nltk_in_our_data:
            if token in irregular_com_sup:
                root, ending = irregular_com_sup[token]
                new_tokens.append(root)
                new_tokens.append('-' + ending)
            else:
                assert token.endswith('est') or token.endswith('er')
                ending = 'est' if token.endswith('est') else 'er'

                # simple -> simpler (i.e., adjective ending in -e)
                if token[:-(len(ending) - 1)] in adjectives_ending_in_e:
                    root = token[:-(len(ending) - 1)]

                # fancy->fancier
                elif token.endswith('i' + ending):
                    root = token.replace('i' + ending, 'y')

                # tall->taller (vanilla)
                else:
                    root = token.replace(ending, '')

                ending = '-' + ending
                new_tokens.append(root)
                new_tokens.append(ending)
                all_conversions.add((token, root, ending))
        else:  # no need to break it down
            new_tokens.append(token)
    return new_tokens


def merge_superlative_comparatives(tokens):
    raise NotImplementedError('Look at 1_ShapeTalk_basic_statistics.ipynb')