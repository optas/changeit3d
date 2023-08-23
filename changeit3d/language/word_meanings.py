view_dependent_hot_words = {'visible',
                            'see',
                            'seen',
                            'showing',
                            'righthand',
                            'lefthand',
                            'left',
                            'right',
                            'front'
                            }  # can add "back", if we exclude when used as a part (chairs, etc.)


color_dependent_hot_words = {'color', 'colour',
                             'shadow', 'dark', 'darker',
                             'white', 'black'}


shape_words = {
    'length', 'height', 'depth',
    'thin', 'thinner', 'thinnest',
    'fat', 'fatter', 'fattest',
    'large', 'larger', 'largest',
    'small', 'smaller', 'smallest',
    'big', 'bigger', 'biggest',
    'wide', 'wider', 'widest',
    'long', 'longer', 'longest',
    'tall', 'taller', 'tallest',
    'thick', 'thicker', 'thickest',
    'narrow', 'narrower', 'narrowest',
    'skinny', 'skinnier', 'skinniest',
    'round', 'rounder', 'roundest', 'rounded',
    'deep', 'deeper', 'deepest',
    'flat', 'flatter', 'flattest',
    'horizontal', 'vertical',
    'cube', 'cuboid', 'cubic', 'cylinder', 'cylindrical',
    'lump', 'angle',  'point', 'pointy',
    'plane', 'planar', 'surface',
    'pyramid', 'pyramidal', 'pyramids',
    'diamond',
    'circle', 'circular',
    'triangle', 'triangular',
    'cone', 'conical',
    'rectangle', 'rectangular',
    'sphere', 'spherical',
    'ellipse', 'ellipsoid',
    'elliptical', 'cylindrical', 'cylinder',
    'symmetry', 'symmetric', 'symmetrical',
    'curve', 'curvy', 'curvier', 'curviest', 'curved',
    'straight',
    'shallow', 'shallower', 'shallowest',
    'oval',  'semicircle', 'square', 'squarish',
    'box', 'boxy', 'bulky', 'bulkier', 'slim',  'slimmer', 'slimmest',
    'pentagon', 'hexagon',
    'octagon', 'parallelogram', 'quadrilateral',
    'rhombus', 'polygon', 'obtuse', 'obtusely', 'orthogonal'
}



# Note the word 'corner' can be also used to refer to a shape. We opt adding to the spatial words
# since this is a much more common use in NR3D.
spatial_prepositions = {'aboard', 'above', 'across', 'adjacent', 'against', 'ahead', 'along',
                        'alongside', 'amid', 'amidst', 'among', 'amongst', 'apart', 'around',
                        'aside', 'astride', 'at', 'away', 'behind', 'below', 'beneath', 'beside',
                        'between', 'beyond', 'by', 'down', 'inside', 'into',
                        'near', 'nearby', 'on', 'onto', 'opposite', 'out', 'outside', 'over',
                        'through', 'together', 'toward', 'under', 'underneath', 'up', 'upper', 'within'}
# left out: 'about', 'in', 'round'

spatial_words = {'far', 'farthest', 'furthest', 'nearest',
                 'holding', 'holds', 'supporting', 'supports',
                 'left', 'right', 'front', 'side',
                 'low', 'lower', 'lowest',
                 'center', 'corner', 'middle', 'closest'}

spatial_tokens = spatial_words.union(spatial_prepositions)
spatial_expressions = [['close', 'to'], ['next', 'to'], ['back', 'of']]


# Panos will put below in a nicer way.
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
            matches.append(pattern)
    return matches


def uses_spatial_reasoning(tokens):
    exact_word = sum([i in tokens for i in spatial_tokens]) > 0  # at least one word (exact match)
    if exact_word:
        return True
    for s in spatial_expressions:
        if len(subfinder(tokens, s)) > 0:
            return True
    return False


style_hotwords = ['traditional', 'traditionally',
                  'modern', 'artsy', 'antique', 'victorian', 'queen', 'royal',
                  'gothic', 'medieval', 'conventional',
                  'artistic',  'stylish', 'decorative',
                  'old', 'oldish', 'new', 'fancy', 'flamboyant',
                  'classic', 'classical', 'classy',
                  'complex',  'simple', 'vanilla', 'common', 'basic', 'plain', 'complicated',
                  'intricate', 'detailed', 'geometric',
                  'cheap', 'expensive', 'rich', 'elegant',
                  'excessive', 'uncommon', 'weird', 'strange', 'rare',
                  'boring',  'refined',  'ornate', 'gracefully', 'elaborate',
                  'wildly', 'faux', 'unusual', 'eccentric',
                  'quirky', 'abnormal', 'abnormally', 'aesthetic', 'aesthetically',
                  'aggressively', 'american', 'asian', 'french', 'chinese', 'retro',
                  'unusual', 'strange', 'unusually', 'rustic', 'roman',
                  'regular', 'ordered', 'orderly', 'ergonomic', 'ergonomically',
                  'ordinary', 'extraordinarily', 'extraordinary', 'extravagant',
                  'beautiful',  'beautifully',  'beauty',  'exotic',
                  'comfy', 'comically', 'comfortably', 'comfortable', 'comfort',
                  'relaxing', 'lazy', 'uncomfortable',
                  'ugly', 'neater', 'clean', 'neatly', 'neat', 'normal', 'normally', 'oriental',
                  'typical', 'style', 'design', 'fashion', 'funny', 'industrial',
                  # shall I include those two:
                  'symmetrical', 'asymmetrical',

                  # couches
                  'burrow', 'tuxedo', 'camel', 'chesterfield', 'cabriole', 'english', 'lawson', 'chaise', 'knol',
                  # chairs
                  'club', 'windsor', 'adirondack', 'dinning', 'cuddler', 'armchair', 'ottoman',
                  # lamps
                  'chandelier', 'empire'
                  # hats

                  # airplanes

                  # etc.
                  ]


style_hotwords = set(style_hotwords)