"""
Describe frequent expressions of ShapeTalk for categories used in editing experiments.
  - sort expressions of ShapeTalk per frequency, per class.
  - pick the 10 most frequent expressions ignoring expressions that are identical in meaning:
     e.g., it is taller, vs. the target is taller

Note. look analysis/2_ShapeTalk_specialized_language_usage_IAN.ipynb
"""

frequent_expressions = dict()

# ----------
# chair
# ----------
# the legs are thin -er                              0.006034
# the legs are short -er                             0.005325
# the seat is thick -er                              0.005071
# the legs are thick -er                             0.004830
# the seat is thin -er                               0.004442
# it has arms                                        0.004161
# the legs are long -er                              0.003612
# the seat is wide -er                               0.003104
# it does not have arms                              0.002783
# it has four legs                                   0.002716
# the target has arm rests                           0.002221
# the seat is small -er                              0.002154
# it has thin -er legs                               0.002020
# there are no arm rests                             0.001913
# the legs are tall -er                              0.001793
# CUMULATIVE-FRACTION:                               0.052179


frequent_expressions['chair'] = \
    {'the legs are thin -er',
     'the legs are short -er',
     'the seat is thick -er',
     'the legs are thick -er',
     'the seat is thin -er',
     'it has arms',
     'the legs are long -er',
     'the seat is wide -er',
     'it does not have arms',
     'it has four legs'
     }

# ----------
# table
# ----------
# it is tall -er                                     0.008924
# the top is thin -er                                0.008710
# the top is thick -er                               0.008081
# it is short -er                                    0.008036
# the legs are thick -er                             0.005833
# it has four legs                                   0.005586
# the top of the target is narrow -er                0.005350
# the legs are thin -er                              0.005271
# it is long -er                                     0.004642
# the target is short -er                            0.004271 (repeats above, exclude it from top-10)
# the target is tall -er                             0.004260 (repeats above, exclude it from top-10)
# the legs are short -er                             0.004091
# the top is less wide                               0.003102
# the top of the target is wide -er                  0.002821
# its legs are thin -er                              0.002619
# CUMULATIVE-FRACTION:                               0.081596


frequent_expressions['table'] = \
    {'it is tall -er',
     'the top is thin -er',
     'the top is thick -er',
     'it is short -er',
     'the legs are thick -er',
     'it has four legs'
     'the top of the target is narrow -er',
     'the legs are thin -er',
     'it is long -er',
     'the legs are short -er',
     }


# ----------
# lamp
# ----------
# it is tall -er                                     0.006042
# it is short -er                                    0.005894
# the base is small -er                              0.005226
# the shade is wide -er                              0.003781
# the target is short -er                            0.003429 (repeats above, exclude it from top-10)
# the target is tall -er                             0.003299 (repeats above, exclude it from top-10)
# the shade is small -er                             0.003225
# the base is wide -er                               0.003058
# the shade is short -er                             0.002928
# the base is large -er                              0.002928
# the base is thick -er                              0.002910
# the pole is thin -er                               0.002836
# the pole is thick -er                              0.002706
# it has a thin -er pole                             0.002595
# it has a square base                               0.002539
# CUMULATIVE-FRACTION:                               0.053395

frequent_expressions['lamp'] = \
    {'it is tall -er',
     'it is short -er',
     'the base is small -er',
     'the shade is wide -er',
     'the shade is small -er',
     'the base is wide -er',
     'the shade is short -er',
     'the base is large -er',
     'the base is thick -er',
     'the pole is thin -er'
     }


# ----------
# airplane
# ----------
# it does not have a propeller                       0.004747
# it has four engines                                0.004477
# it has a propeller                                 0.004207
# the fuselage is long -er                           0.003960
# it has two engines                                 0.003960
# it has a long -er fuselage                         0.003847 (repeats above, exclude it from top-10)
# it has winglets                                    0.003397
# the fuselage is short -er                          0.003375
# it does not have wheels                            0.003330
# it has a small -er wingspan                        0.003285
# it has wheels                                      0.003195
# it is a fighter jet                                0.003172
# it does not have propellers                        0.003105
# it has propellers                                  0.003105
# it has a short -er fuselage                        0.003037
# CUMULATIVE-FRACTION:                               0.054196

frequent_expressions['airplane'] = \
    {'it does not have a propeller',
     'it has four engines ',
     'it has a propeller',
     'the fuselage is long -er',
     'it has two engines',
     'it has winglets',
     'the fuselage is short -er',
     'it does not have wheels',
     'it has a small -er wingspan',
     'it has wheels'
     }


# tokenize based on space
for shape_class in frequent_expressions:
    frequent_expressions[shape_class] = \
        [sent.split() for sent in frequent_expressions[shape_class]]
