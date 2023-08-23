"""
Per object-class part-word dictionaries.
    Different Granularities. E.g., Coarse vs. Fine-Grained

Note.
We note the singular of each noun here. Use an engine like inflate to extract the plural form.
E.g., pip install inflect https://pypi.org/project/inflect/
"""


##
# Part-name applicable to all/most shapes
##

human_body_inspired_part_names = {'body', 'foot', 'neck', 'joint', 'elbow', 'head', 'heel',
                                  'lip', 'butt', 'mouth', 'ear', 'foot', 'leg', 'hand',
                                  'toe', 'spine', 'shoulder', 'neck', 'arm'
                                  }

generic_part_names = {'base',  'button', 'stem',  'pedestal', 'divider', 'beam',
                      'stand', 'handle', 'frame', 'panel', 'top', 'skirting', 'skirt',
                      'sides', 'hinge', 'knob', 'lever', 'pull', 'cable',
                      'switch', 'spring', 'strut', 'footprint', 'compartment',
                      'clawfoot', 'shelve', 'drape', 'wall', 'rail',
                      'spindle', 'spindled', 'spindles', 'spindly', 'corner', 'bottom', 'edge',
                      'backboard'
                      }

generic_part_names |= human_body_inspired_part_names


##
# Chair
##
chair_coarse_dict = dict()
chair_coarse_dict['arm'] = {'arm', 'armrest', 'cup', 'holders', 'cupholders'}
chair_coarse_dict['leg'] = {'leg', 'legged', 'wheel', 'wheels', 'swivel', 'foot',
                            'base', 'footrest', 'roller', 'caster'}
chair_coarse_dict['seat'] = {'seat', 'sit', 'sitting', 'seater', 'seating'}
chair_coarse_dict['back'] = {'back', 'backed', 'backing', 'backside', 'backrest', 'backboard', 'backpiece',
                             'slat', 'slit', 'slot', 'bar', 'head', 'headrest'}

# other_parts: it is ambiguous which part exactly a reference that mentions them is about
# e.g, pole can talk about the back of a chair, or a pole looking leg.
chair_coarse_dict['other_parts'] = {'rest', 'rail', 'pole', 'track', 'stick', 'pillow', 'bar', 'post'}

chair_parts = set()
for major_part in chair_coarse_dict.keys():
    chair_parts.update(chair_coarse_dict[major_part])


##
# Bathtub
##
bathtub_parts = {'faucet', 'sprout', 'spout', 'rim', 'foot', 'bar', 'panel',
                 'gasket', 'overflow', 'drain', 'abutment', 'pedestal',
                 'rim', 'drain', 'pan', 'knob', 'headrest', 'shelf', 'shelving',
                 'bar', 'handle', 'rail', 'jet', 'abutment', 'lever,'
                 'hardware', 'stair', 'step', 'button', 'tap',
                 'seat', 'base', 'basin', 'leg'}

##
# Faucet
##
faucet_parts = {'faucet', 'aerator', 'body', 'handle', 'aerator', 'ring',
                'spout', 'lever', 'assembly', 'sprayer', 'escutcheon',
                'knob', 'control', 'hose', 'button', 'foot',
                'sprayer', 'base', 'head', 'rim', 'sink'}

##
# Display
##
display_parts = {'frame', 'screen', 'stand', 'neck', 'button',
                 'case', 'display', 'keyboard', 'pedestal',
                 'base', 'control', 'joystick',  'dial',
                 'monitor', 'back', 'speaker', 'body',
                 'port', 'camera', 'vent'}

##
# Cabinet
##
cabinet_parts = {'drawer', 'knob', 'rail', 'compartment', 'side',
                 'door', 'hinge', 'top', 'back', 'carcass',
                 'leg', 'handle', 'knob', 'base', 'shelving', 'shelf',
                 'wheel', 'foot', 'pull', 'cupboard', 'latch'}

##
# Lamp
##
lamp_parts = {'base', 'chain', 'pole', 'fixture', 'arm',
              'head', 'leg', 'shade', 'lid', 'bar', 'neck', 'cable',
              'lever', 'plug', 'switch', 'bulb', 'wire', 'body', 'socket',
              'mount', 'cover', 'stem', 'tube', 'rod', 'cord', 'tubing',
              'post', 'plate', 'shelf', 'finial', 'harp', 'switch', 'rod',
              'joint', 'spring', 'strut', 'pedestal'}
# > 'light', 'lamp'

##
# Vase
##
vase_parts = {'neck', 'shoulder', 'lip', 'mouth', 'base', 'body',
              'foot', 'leg', 'flower', 'plant', 'lid', 'leaf', 'wheel',
              'handle', 'rim', 'head', 'bottom', 'opening', 'pedestal'}

##
# Flowerpot
##
flowerpot_parts = {'mouth', 'body', 'foot', 'plant', 'base', 'lip', 'leg',
                   'leaf', 'flower', 'weed', 'lid', 'cover', 'opening',
                   'hanger', 'rim', 'pedestal'}

##
# Table
##
table_parts = {'leg', 'top', 'drawer', 'apron', 'footrest', 'stretcher',
               'shelf', 'shelving', 'shelve', 'divider', 'net', 'wheel',
               'handle', 'knob', 'foot', 'feet', 'arm', 'base', 'pedestal',
               'cubby', 'cabinet', 'compartment', 'support', 'bracing',
               'door', 'garter', 'tabletop'}

##
# Bed
##
bed_parts = {'headboard', 'pillow', 'nightstand', 'headrest', 'mattress',
             'blanket', 'footboard', 'bar', 'step', 'foot', 'panel',
             'stair', 'wheel', 'frame', 'drawer', 'slat', 'bunk', 'leg',
             'shelf', 'shelving', 'ladder', 'canopy', 'bolster', 'rail',
             'door', 'cabinet', 'cubby'}

##
# Bench
##
bench_parts = {'slat', 'back', 'leg', 'arm', 'seat', 'foot', 'handle',
               'handrail', 'drawer', 'base', 'rail', 'cushion', 'divider'}
bench_parts |= chair_parts


##
# Bookshelf
##
bookshelf_parts = {'shelf', 'frame', 'book', 'cubicle', 'desk', 'foot', 'feet',
                   'shelving', 'cupboard', 'cubby', 'base', 'upright',
                   'divider', 'door', 'compartment', 'slat', 'spindle',
                   'drawer', 'back', 'wheel', 'leg', 'ladder', 'molding',
                   'cabinet', 'handle', 'knob'}

##
# Dresser
##
dresser_parts = {'cupboard', 'door', 'cabinet', 'molding', 'lock', 'top',
                 'latch', 'shelf', 'shelving', 'leg', 'handle', 'foot',
                 'drawer', 'divider', 'hardware', 'pull', 'cubicle',
                 'knob', 'panel', 'cubby', 'compartment', 'wheel', 'back',
                 'base'}

##
# Knife
##
knife_parts = {'tip', 'handle', 'blade', 'butt',
               'guard', 'spine', 'bolster', 'grip', 'hand-guard'}


##
# Sofa
##
sofa_parts = {'cushion', 'slat', 'pillow', 'leg', 'back', 'arm',
              'backrest', 'foot', 'seat', 'swivel', 'armrest'}
sofa_parts |= chair_parts

##
# Trash bin
##
trashbin_parts = {'lid', 'cover', 'foot', 'top', 'handle', 'hole', 'leg',
                  'base', 'bar', 'cage', 'drawer', 'container', 'body',
                  'mouth', 'rim', 'wheel', 'roof', 'stripe', 'lip', 'opening'}

##
# Clock
##
clock_parts = {'hand', 'dial', 'pendulum', 'face', 'case'}

##
# Mug
##
mug_parts = {'handle', 'lip', 'bottom', 'body', 'base'}

##
# Bottle
##
bottle_parts = {'mouth', 'neck', 'finish', 'shoulder', 'body', 'orifice', 'heel', 'ring', 'base'}

##
# Skateboard
##
skateboard_parts = {'board', 'wheels', 'nose', 'tail', 'axle', 'deck', 'truck', 'bearings'}

##
# Bag
##
bag_parts = {'handle', 'strap', 'zipper', 'piping', 'tab', 'foot', 'body', 'wheels'}

##
# Cap/Hat
##
cap_parts = {'brim', 'crown', 'visor', 'button', 'eyelets', 'sweatband', 'closure', 'panels', 'band', 'creases',
             'bill'}

##
# Person-specific
##
person_parts = {'eye', 'face', 'lip', 'jaw', 'hair'}

##
# Scissors
##
scissors_parts = {'blade', 'point', 'tip', 'screw', 'handle', 'stopper', 'silencer', 'pivot'}


##
# Pistol
##
pistol_parts = {'sight', 'slide', 'hammer', 'grip', 'muzzle', 'magazine', 'trigger', 'guard', 'barrel'
                'frame', 'lever', 'safety', 'release', 'strap', 'cylinder', 'silencer'}

##
# Airplane
##
airplane_parts = {'engine', 'wheel', 'winglet', 'stabilizer', 'wing', 'aileron', 'propeller',
                  'flap', 'fuselage', 'elevator', 'cabin', 'tail', 'tailplane', 'cockpit', 'spinner',
                  'fin', 'rudder', 'armament', 'armour', 'weapon', 'wingtip'}
# + landing gear 'nose', 'tip'?
                  
##
# Guitar
##
guitar_parts = {'neck', 'buttons', 'headstock', 'bout', 'fret', 'rosette', 'pickguard', 'bridge',
                'soundboard', 'fretboard', 'nut', 'peg', 'pickup', 'knob',
                }
#+ sound hole, head, pick guard, tone control, volume control, MORE


##
# Bowl
##
bowl_parts = {'neck', 'rim', 'base'}

##
# Plant
##
plant_parts = {'leaf', 'stem', 'flower', 'vase', 'vein', 'root'}

##
# Helmet
##
helmet_parts = {'brim', 'chin', 'bill', 'handle', 'crown', 'shell', 'strap', 'chinstrap', 'gorget', 'fastener', 'visor'}



part_names_all_classes = dict()
part_names_all_classes['generic'] = generic_part_names
part_names_all_classes['human_body'] = human_body_inspired_part_names
part_names_all_classes['airplane'] = airplane_parts
part_names_all_classes['bathtub'] = bathtub_parts
part_names_all_classes['bag'] = bag_parts
part_names_all_classes['bed'] = bed_parts
part_names_all_classes['bench'] = bench_parts
part_names_all_classes['bookshelf'] = bookshelf_parts
part_names_all_classes['bottle'] = bottle_parts
part_names_all_classes['bowl'] = bowl_parts
part_names_all_classes['cap'] = cap_parts
part_names_all_classes['cabinet'] = cabinet_parts
part_names_all_classes['clock'] = clock_parts
part_names_all_classes['chair'] = chair_parts
part_names_all_classes['dresser'] = dresser_parts
part_names_all_classes['display'] = display_parts
part_names_all_classes['faucet'] = faucet_parts
part_names_all_classes['flowerpot'] = flowerpot_parts
part_names_all_classes['guitar'] = guitar_parts
part_names_all_classes['helmet'] = helmet_parts
part_names_all_classes['knife'] = knife_parts
part_names_all_classes['lamp'] = lamp_parts
part_names_all_classes['mug'] = mug_parts
part_names_all_classes['person'] = human_body_inspired_part_names | person_parts
part_names_all_classes['pistol'] = pistol_parts
part_names_all_classes['plant'] = plant_parts
part_names_all_classes['skateboard'] = skateboard_parts
part_names_all_classes['scissors'] = scissors_parts
part_names_all_classes['sofa'] = sofa_parts
part_names_all_classes['table'] = table_parts
part_names_all_classes['trashbin'] = trashbin_parts
part_names_all_classes['vase'] = vase_parts




#TODO NOT USED but it could
bigram_part_words = dict()
bigram_part_words['knife'] = {'cutting edge', 'hand guard'}
bigram_part_words['faucet'] = {'shower head'}
bigram_part_words['display'] = {'track pad'}  # apparently some displays include a computer setup
bigram_part_words['pistol'] = {'lanyard loop'}
bigram_part_words['scissors'] = {'finger rest', 'cutting edge', 'finger hole'}