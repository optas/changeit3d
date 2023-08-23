"""
ModelNet.
"""

model_net_40_categories = {
    'airplane',
    'bathtub',
    'bed',
    'bench',
    'bookshelf',
    'bottle',
    'bowl',
    'car',
    'chair',
    'cone',
    'cup',
    'curtain',
    'desk',
    'door',
    'dresser',
    'flower_pot',
    'glass_box',
    'guitar',
    'keyboard',
    'lamp',
    'laptop',
    'mantel',
    'monitor',
    'night_stand',
    'person',
    'piano',
    'plant',
    'radio',
    'range_hood',
    'sink',
    'sofa',
    'stairs',
    'stool',
    'table',
    'tent',
    'toilet',
    'tv_stand',
    'vase',
    'wardrobe',
    'xbox'
}

# The ModelNet classes below contain objects that are extremely similar on average
# to the corresponding ones in ShapeNet classes and should be considered as the same class
modelnet_to_shapenet_categories = {
    'flower_pot': 'pot',
    'vase': 'jar'
}

