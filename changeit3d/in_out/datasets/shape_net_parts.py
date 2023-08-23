"""
TODO. NOT fully finished, BUT for its minimal usage for ShapeTalk is OK.
"""

from torch.utils.data import Dataset
import numpy as np
import os.path as osp

class ShapeNetParts(Dataset):
    r"""The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing 17,775 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.
    Code adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html"""

    url = ('https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')

    # number of maximum parts annotates in each shape instance per class
    n_parts = {
        'airplane': 4,
        'bag': 2,
        'cap': 2,
        'car': 4,
        'chair': 4,
        'earphone': 3,
        'guitar': 3,
        'knife': 2,
        'lamp': 4,
        'laptop': 2,
        'motorbike': 6,
        'mug': 2,
        'pistol': 3,
        'rocket': 3,
        'skateboard': 3,
        'table': 3
    }

    @staticmethod
    def synid_to_shape_category_maps(self):
        """
        e.g.,  '03797390'-> 'mug'
        :return:
        """
        syn_to_cat_file = osp.join(self.root_dir, 'synsetoffset2category.txt')
        synid_to_shape_cat = dict()
        with open(syn_to_cat_file) as fin:
            for line in fin:
                category, synsetid = line.strip().split()
                synid_to_shape_cat[synsetid] = category.lower()
        shape_cat_to_synid = {v: k for k, v in synid_to_shape_cat.items()}
        return synid_to_shape_cat, shape_cat_to_synid

    def __init__(self, root_dir, split_df, number_of_points):
        super(ShapeNetParts, self).__init__()
        self.root_dir = root_dir
        self.number_of_points = number_of_points
        self.synid_to_shape_cat, self.shape_cat_to_synid = self.synid_to_shape_category_maps()

        pc_data = []
        meta_data = []

        for c, m, in zip(split_df.shape_class, split_df.model_name):
            syn_id = self.category_ids[c]
            pts_file = osp.join(self.root_dir, syn_id, 'points', m + '.pts')
            segmentation_file = osp.join(self.root_dir, syn_id, 'points_label', m + '.seg')
            point_set = np.loadtxt(pts_file).astype(np.float32)
            segmentation = np.loadtxt(segmentation_file).astype(np.int64)
            pc_data.append()

        self.pc_data = pc_data


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:, [0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        pass


# what part does each point-cloud assigned integer means in plain english TODO-FINISH those in empty strings ''
idx_to_segmentation_part_name = {
    'airplane': {1: 'body', 2: 'wings', 3: 'rudder', 4: 'engines'},
    'bag': {1: 'handle', 2: 'body'},
    'cap': {1: 'crown', 2: 'visor'},
    'car': {1: '', 2: '', 3: '', 4: ''},
    'chair': {1: 'back', 2: 'seat', 3: 'leg', 4: 'arm'},
    'earphone': {1: '', 2: '', 3: ''},
    'guitar': {1: 'headstock', 2: 'neck', 3: 'body'},
    'knife': {1: 'blade', 2: 'handle'},
    'lamp': {1: 'floor-base', 2: 'shade', 3: 'canopy', 4: 'body'},
    'laptop': {1: '', 2: ''},
    'motorbike': {1: '', 2: '', 3: '', 4: '', 5: '', 6: ''},
    'mug': {1: 'handle', 2: 'body'},
    'pistol': {1: '', 2: '', 3: ''},
    'rocket': {1: '', 2: '', 3: ''},
    'skateboard': {1: 'wheels', 2: 'deck', 3: 'truck'},
    'table': {1: 'top', 2: 'leg', 3: None}  # 3=> inconsistent: sometimes shelves sometimes connectors...
}


canonical_part_names_to_relevant_words = dict()
canonical_part_names_to_relevant_words['chair'] = {'leg': ['leg', 'legged', 'wheel', 'wheels', 'swivel', 'foot', 'base', 'footrest', 'roller', 'caster'],
                                                   'back': ['back', 'backed', 'backing', 'backside', 'backrest', 'backboard', 'backpiece',
                                                            'slat', 'slats', 'slit', 'slits', 'slot', 'slots', 'bars', 'head', 'headrest'],                                                   
                                                   'arm': ['arm', 'arms', 'armrest', 'armrests', 'cupholder', 'cupholders'],
                                                   'seat': ['seat', 'sit', 'seating', 'sitting', 'seater']}

canonical_part_names_to_relevant_words['table'] = {'leg': ['leg', 'legs', 'feet', 'foot', 'footrest', 'stretcher', 'base', 'pedestal'],
                                                   'top': ['top', 'apron', 'tabletop']}

canonical_part_names_to_relevant_words['lamp'] = {'floor-base': ['base'],
                                                  'shade': ['shade', 'lampshade'],
                                                  'canopy': ['canopy'],
                                                  'body': ['body', 'pole']}

canonical_part_names_to_relevant_words['airplane'] = {'body': ['body', 'fuselage'],
                                                      'wings': ['wing', 'wings', 'flaps', 'wingspan', 'winglets'],
                                                      'rudder': ['rudder', 'elevator', 'stabilizer', 'fin'], # In this dataset, the "rudder" (label=3) includes all of them
                                                      'engines': ['engine', 'engines']
                                                      }

canonical_part_names_to_relevant_words['knife'] = {'blade': ['blade'], 
                                                   'handle': ['handle', 'grip', 'handgrip']}

canonical_part_names_to_relevant_words['bag'] = {'body': ['body'], 
                                                 'handle': ['handle', 'grip', 'handgrip']}

canonical_part_names_to_relevant_words['cap'] = {'crown': ['crown'], 
                                                 'visor': ['visor', 'brim']}

