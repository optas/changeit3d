##
## An assortment of manual made decisions and corresponding data that will improve the quality of the
## ShapeTalk Dataset
##

digit_to_alpha = {'1': 'one',
                  '2': 'two',
                  '3': 'three',
                  '4': 'four',
                  '5': 'five',
                  '6': 'six',
                  '7': 'seven',
                  '8': 'eight',
                  '9': 'nine',
                  '10': 'ten',
                  '11': 'eleven',
                  '12': 'twelve',
                  '13': 'thirteen',
                  '14': 'fourteen',
                  '15': 'fifteen',
                  '16': 'sixteen',
                  '17': 'seventeen',
                  '18': 'eighteen',
                  '19': 'nineteen',
                  '20': 'twenty'
                  }


sentence_spelling_dictionary = dict()
## from ShapeGlot
sentence_spelling_dictionary['HAS TALLER LEGS AND IS A SQUARE DEAT.'] = 'HAS TALLER LEGS AND HAS A SQUARE SEAT.'
sentence_spelling_dictionary["it has a bumpy'ish back, and thicker legs"] = 'it has kind of a bumpy looking back and thicker legs'
sentence_spelling_dictionary['sits flat on the ground—looks like a seat from a car'] = 'sits flat on the ground. Looks like a seat from a car'
sentence_spelling_dictionary['odd, round chair, triangle leges'] = 'odd, round chair, triangle legs'
sentence_spelling_dictionary['heart shaped back and longest leges'] = 'heart shaped back and longest legs'
sentence_spelling_dictionary['Flat bottom with III in the back'] = 'flat bottom with three slats in the back'
sentence_spelling_dictionary['back is like III'] = 'back has three slats'
sentence_spelling_dictionary['IIII  FOUR'] = 'four slats'
sentence_spelling_dictionary['IIII'] = 'four slats'
sentence_spelling_dictionary['no arms, III*III'] = 'no arms, many slats and ornament in the back'

## from ShapeTalk
sentence_spelling_dictionary["The top's thinner."] = 'The top is thinner.'
sentence_spelling_dictionary["The top;'s thinner."] = 'The top is thinner.'
sentence_spelling_dictionary["the top's thinner."] = 'The top is thinner.'
sentence_spelling_dictionary["THe top's thinner."] = 'The top is thinner.'
sentence_spelling_dictionary["The top's thicker."] = 'The top is thicker.'
sentence_spelling_dictionary["the top's thicker."] = 'The top is thicker.'
sentence_spelling_dictionary["The top's thidcker."] = 'The top is thicker.'
sentence_spelling_dictionary["The top's less wide."] = 'The top is less wide.'
sentence_spelling_dictionary["The top's less wide.."] = 'The top is less wide.'
sentence_spelling_dictionary["The top's less widfe."] = 'The top is less wide.'
sentence_spelling_dictionary["The top's longer."] = 'The top is longer.'
sentence_spelling_dictionary["The top's more wide."] = 'The top is more wide.'
sentence_spelling_dictionary["THE TOP'S WIDER."] = 'The top is wider.'
sentence_spelling_dictionary["The top's wider."] = 'The top is wider.'
sentence_spelling_dictionary["the top's wider."] = 'The top is wider.'
sentence_spelling_dictionary["The top's less long."] = 'The top is less long.'
sentence_spelling_dictionary["THe top's less long."] = 'The top is less long.'
sentence_spelling_dictionary["The footprint's larger."] = 'The footprint is larger.'
sentence_spelling_dictionary["The footprint's larger"] = 'The footprint is larger'
sentence_spelling_dictionary["The footprint's smaller."] = 'The footprint is smaller.'
sentence_spelling_dictionary["The footpirnt's smaller."] = 'The footprint is smaller.'
sentence_spelling_dictionary["The .footprint's smaller."] = 'The footprint is smaller.'
sentence_spelling_dictionary["The footpritn's smaller."] = 'The footprint is smaller.'
sentence_spelling_dictionary["The skirt's taller."] = 'The skirt is taller.'
sentence_spelling_dictionary["THE SKIRT'S TALLER."] = 'The skirt is taller.'



# noinspection PyDictCreation
token_spelling_dictionary = {'taget': 'target',
                             'arget': 'target',
                             'trger': 'target',
                             'taraget': 'target',
                             'targer': 'target',
                             'thetarget': ['the', 'target'],
                             'thebtarget': ['the', 'target'],
                             'thetyarget': ['the', 'target'],
                             'thentarget': ['the', 'target'],
                             'distracter': 'distractor',
                             'distr4actor': 'distractor',
                             'edistractor': 'distractor',
                             'distravtor': 'distractor',
                             'distraactor': 'distractor',
                             'distractotr': 'distractor',
                             'disctractor': 'distractor',
                             'distarctor': 'distractor',
                             'draweers': 'drawers',
                             'hree': 'three',
                             'bottommost': ['bottom', 'most'],
                             'lenght': 'length',
                             'desgn': 'design',
                             'symetrical': 'symmetrical',
                             'shoter': 'shorter',
                             'talller': 'taller',
                             'retangular': 'rectangular',
                             'insid': 'inside',
                             'muchtaller': ['much', 'taller'],
                             'bedframes': ['bed', 'frames'],
                             'thesupports': ['the', 'supports'],
                             'watsebin': ['waste', 'bin'],
                             'andlower': ['and', 'lower'],
                             'curvedsupports': ['curved', 'supports'],
                             'topwithout': ['top', 'without'],
                             'streetlamp': ['street', 'lamp'],
                             'difrenet': 'different',
                             'lightbulb': ['light', 'bulb'],
                             'cylindricallegs': ['cylindrical', 'legs'],
                             'skinnieset': 'skinniest',
                             'thinnerlegs': ['thinner', 'legs'],
                             'legshave': ['legs', 'have'],
                             'iswidder': ['is', 'wider'],
                             'lesscurved': ['less', 'curved'],
                             'areshorter': ['are', 'shorter'],
                             'itsbody': ['its', 'body'],
                             'arebigger': ['are', 'bigger'],
                             'longerqit': ['longer', 'and', 'it'],
                             'nottapered': ['not', 'tapered'],
                             'thespout': ['the', 'spout'],
                             'estetchuan': 'escutcheon',
                             'pedistool': 'pedestal',
                             'pedastool': 'pedestal',
                             'bedframe': ['bed', 'frame'],
                             'higer': 'higher',
                             'wier': 'wider',
                             'widder': 'wider',
                             'harrower': 'narrower',
                             'lamo': 'lamp',
                             'windowframes': ['window', 'frames'],
                             'andthe': ['and', 'the'],
                             'twp': 'two',
                             '2nd': 'second',
                             'thestretcher': ['the', 'stretcher'],
                             'thereat': ['the', 'seat'],
                             '1the': 'the',
                             'togther': 'together',
                             '1ft': ['one', 'foot'],
                             '2ft': ['two', 'feet'],
                             'anad': 'and',
                             'arent': ['are', 'not'],
                             'bac': 'back',
                             'ahs': 'has',
                             'chesspiece': ['chess', 'piece'],
                             'bottlecap': ['bottle', 'cap'],
                             'fourdistict': ['four', 'distinct'],
                             'aand': 'and',
                             'aare': 'are',
                             'adn': 'and',
                             'almos': 'almost',
                             'alot': ['a', 'lot'],
                             'curvedsides': ['curved', 'sides'],
                             'upsidedown': ['upside', 'down'],
                             'thedistractor': ['the', 'distractor'],
                             'lgs': 'legs',
                             'fram': 'frame',
                             'backpillows': ['back', 'pillows'],
                             'backcushions': ['back', 'cushions'],
                             'doorhandles': ['door', 'handles'],
                             'streelamp': ['street', 'lamp'],
                             'targetare': ['target', 'are'],
                             'themouth': ['the', 'mouth'],
                             'tablespace': ['table', 'space'],
                             'emptydrawers': ['empty', 'drawers'],
                             'thetraget': ['the', 'target'],
                             'heptigon': 'heptagon',
                             'octoganol': 'octagonal',
                             'disitractor': 'distractor',
                             'distracator': 'distractor',
                             'distratcor': 'distractor',
                             'liketwo': ['like', 'two'],
                             'thetagrt': ['the', 'target'],
                             'haverectangular': ['have', 'rectangular'],
                             'toekick': ['toe', 'kick'],
                             'middle-most': 'middlemost',
                             'powercord': ['power', 'cord'],
                             'inidivudal': 'individual',
                             'pedetastools': 'pedestals',
                             'beneaththem': ['beneath', 'them'],
                             'havereinforced': ['have', 'reinforced'],
                             'sidestretcher': ['side', 'stretcher'],
                             'edgesprotruding': ['edges', 'protruding'],
                             'leverpoints': ['lever', 'points'],
                             'cylindershape': ['cylinder', 'shape'],
                             'clotheshangers': ['clothes', 'hangers'],
                             'supportbelow': ['support', 'below'],
                             'nonadjustable': ['non', 'adjustable'],
                             'morepointed': ['more', 'pointed'],
                             'targetdoes': ['target', 'does'],
                             'firepit': ['fire', 'pit'],
                             'wirecable': ['wire', 'cable'],
                             'targetchair': ['target', 'chair'],
                             'lawnchair': ['lawn', 'chair'],
                             'openspace': ['open', 'space'],
                             'thetargethas': ['the', 'target', 'has'],
                             'pullstring': ['pull', 'string'],
                             'thebottom': ['the', 'bottom'],
                             'crosssection': ['cross', 'section'],
                             'soundhole': ['sound', 'hole'],
                             'soundholes': ['sound', 'holes'],
                             'chairhandles': ['chair', 'handles'],
                             'thinnerand': ['thinner', 'and'],
                             'airplanehas': ['airplane', 'has'],
                             'targetplane': ['target', 'plane'],
                             'havedelta': ['have', 'delta'],
                             'verticalstabilizar': ['vertical', 'stabilizer'],
                             'wingsare': ['wings', 'are'],
                             'frontmost': ['front', 'most'],
                             'thefuselage': ['the', 'fuselage'],
                             'horozential': 'horizontal',
                             'lightnigbolt': ['lightning', 'bolt'],
                             'havepropellers': ['have', 'propellers'],
                             'itsrear': ['its', 'rear'],
                             'sharppoints': ['sharp', 'points'],
                             'strapholders': ['strap', 'holders'],
                             'rightside': ['right', 'side'],
                             'thickervisor': ['thicker', 'visor'],
                             'groetsch': 'gretsch',
                             'smallertrigger': ['smaller', 'trigger'],
                             'soudhole': ['sound', 'hole'],
                             'hasarmaments': ['has', 'armaments'],
                             'countetop': 'countertop',
                             'nonretractable': ['non', 'retractable'],
                             'arestraight': ['are', 'straight'],
                             'sidesextending': ['sides', 'extending'],
                             'roundtubes': ['round', 'tubes'],
                             'cylindricaland': ['cylindrical', 'and'],
                             'moretapered': ['more', 'tapered'],
                             'bodybis': ['body', 'is'],
                             'havewinglets': ['have', 'winglets'],
                             'benchseat': ['bench', 'seat'],
                             'amourments': 'armaments',
                             'crossrails': ['cross', 'rails'],
                             "doesnl't": ['does', 'not'],
                             'fatterfuselage': ['fatter', 'fuselage'],
                             'muchwider': ['much', 'wider'],
                             'targetairplane': ['target', 'airplane'],
                             'overallshape': ['overall', 'shape'],
                             'mouthprint': ['mouth', 'print'],
                             '90°': ['ninety', 'degree'],
                             '90': 'ninety',
                             'shapedthing': ['shaped', 'thing'],
                             'underneatheach': ['underneath', 'each'],
                             'tiltrotors': ['tilt', 'rotors'],
                             'oldschool': ['old', 'school'],
                             'thatbreach': ['that', 'breach'],
                             'forwardthe': ['forward', 'the'],
                             'faceshield': ['face', 'shield'],
                             'cornersits': ['corners', 'its'],
                             'labelholders': ['label', 'holders'],
                             'stairsteps': ['stair', 'steps'],
                             'pokeball': ['poke', 'ball'],
                             'basemolding': ['base', 'molding'],
                             'streetpole': ['street', 'pole'],
                             'edgetrim': ['edge', 'trim'],
                             'lighbulb': ['light', 'bulb'],
                             'fourleaf': ['four', 'leaf'],
                             'centerpost': ['center', 'post']
                             }


# this way we link the work "height" with "heightwise"
token_spelling_dictionary['heightwise'] = ['height', 'wise']
token_spelling_dictionary['lengthwise'] = ['length', 'wise']
token_spelling_dictionary['depthwise'] = ['depth', 'wise']
token_spelling_dictionary['widthwise'] = ['width', 'wise']
token_spelling_dictionary['shapewise'] = ['shape', 'wise']
token_spelling_dictionary['stylewise'] = ['style', 'wise']
token_spelling_dictionary['crosswise'] = ['cross', 'wise']


## TODO decide on more regularizations e.g, -ly, -ness, -like
## here I basically decide also on the tokenization, this is not kosher.
token_spelling_dictionary['plinthlike'] = ['plinth', 'like']
token_spelling_dictionary['plinthlike'] = ['plinth', 'like']
token_spelling_dictionary['plithlike'] = ['plinth', 'like']
token_spelling_dictionary['plinthelike'] = ['plinth', 'like']
token_spelling_dictionary['pinthlike'] = ['plinth', 'like']
token_spelling_dictionary['pillowlike'] = ['pillow', 'like']
token_spelling_dictionary['blocklike'] = ['block', 'like']
token_spelling_dictionary['plantlike'] = ['plant', 'like']
token_spelling_dictionary['rooflike'] = ['roof', 'like']
token_spelling_dictionary['bulblike'] = ['bulb', 'like']
token_spelling_dictionary['knoblike'] = ['knob', 'like']
token_spelling_dictionary['corklike'] = ['cork', 'like']
token_spelling_dictionary['midrails'] = ['mid', 'rails']
token_spelling_dictionary['multitool'] = ['multi', 'tool']
token_spelling_dictionary['multisided'] = ['multi', 'sided']
token_spelling_dictionary['hookshaped'] = ['hook', 'shaped']
token_spelling_dictionary['seethrough'] = ['see', 'through']
token_spelling_dictionary['lamphead'] = ['lamp', 'head']
token_spelling_dictionary['egglike'] = ['egg', 'like']
token_spelling_dictionary['leverlike'] = ['lever', 'like']


token_spelling_dictionary['rectanguarly'] = 'rectangularly'
token_spelling_dictionary['hemisphiercally'] = 'hemispherically'



## that is that the key words are actual correct words. However, in the context of our dataset
## they seems as typos of the corresponding values. This has partially being verified, hence the
## "with_some_uncertainty".
token_spelling_dictionary_with_some_uncertainty = {
    'targets': ['target', '\'s'],
    'flater': 'flatter',
    'skinner': 'skinnier',
    'arcing': 'arching',
    'backboarding': 'backboard',
}


# for convenience we treat the conversion of digit_to_alpha as spell-check
token_spelling_dictionary.update(digit_to_alpha)
token_spelling_dictionary.update(token_spelling_dictionary_with_some_uncertainty)


missing_from_glove_but_are_actual_words = {'nonagon', 'pointier', 'conically',
                                           'predrilled', 'downlighter',
                                           'handleless', 'dodecagon', 'torchiere',
                                           'middlemost', 'ladderback', 'frontwards', 'concavities',
                                           'backpiece', 'gretsch', 'guitarron', 'drawerless',
                                           'cuddler', 'swivelable', 'widthways', 'lengthways',
                                           'checkmarked', 'octothorpe', 'bottlenosed', 'ridgeless',
                                           'outreaching', 'overarches', 'stairstep'
                                           }

# 'width-wise', 'height-wise', 'length-wise',
# 'depth-wise', 'shape-wise', 'style-wise',
# 'cross-wise', 'pillow-like', 'plinth-like',


token_spelling_dictionary_human_clip=dict()
token_spelling_dictionary_human_clip['flipflops'] = ['flip', 'flops']
token_spelling_dictionary_human_clip['polkadots'] = ['polka', 'dots']

