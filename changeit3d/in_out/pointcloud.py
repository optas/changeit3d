"""
Dataset oriented routines to handle pointclouds of shapes.

Originally created at 2/16/21, for Python 3.x
2022 Panos Achlioptas (https://optas.github.io)

TODO. Make dependency of point-cloud subsampling and random_seed to be such that the batch_size does not matter.
      https://github.com/pytorch/pytorch/issues/5059
"""

import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial
from .datasets.shape_talk import model_to_file_name
from ..utils.basics import parallel_apply


class PointcloudDataset(Dataset):
    def __init__(self, pointclouds, part_masks=None, model_classes=None, model_metadata=None, pc_transform=None):
        """
        :param pointclouds: iterable of N point-clouds, each being K x 3 points (floats). Typically, this is
         a numpy-array (or a list of size N).
        :param part_masks: part-labels for each provided point of each pointcloud. Assumes same order as
        `pointlouds`.
        :param model_metadata: pandas dataframe storing metadata indicating e.g., the names/classes of the provided
        point-clouds.
        """
        super(PointcloudDataset, self).__init__()
        self.pointclouds = pointclouds
        self.part_masks = part_masks
        self.model_classes = model_classes
        self.model_metadata = model_metadata
        self.pc_transform = pc_transform

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, index):
        pc = self.pointclouds[index]
        if self.pc_transform is not None:
            pc = self.pc_transform(pc)

        part_mask = []
        if self.part_masks is not None:
            part_mask = self.part_masks[index]

        model_class = []
        if self.model_classes is not None:
            model_class = self.model_classes.iloc[index]

        model_metadata = []
        if self.model_metadata is not None:
            model_metadata = self.model_metadata.iloc[index].to_dict()

        return {'pointcloud': pc,
                'part_mask': part_mask,
                'model_class': model_class,
                'model_metadata': model_metadata,
                'index': index}



def pc_loader_from_npz(npz_filename, swap_xy_axis=False, assert_preprocessed=False,
                       only_pc=True, n_samples=None, random_seed=None, pc_dtype=np.float32, part_label_dtype=np.int32):

    pc = np.load(npz_filename)['pointcloud'].astype(pc_dtype)

    if swap_xy_axis:
        pc = swap_axes_of_pointcloud(pc, [0, 2, 1])

    if assert_preprocessed:
        assert is_centered_in_unit_sphere(pc)

    if n_samples is not None:
        pc, selected_idx = uniform_subsample(pc, n_samples=n_samples, random_seed=random_seed)
        
    result = dict()
    result['pc'] = pc

    if only_pc:
        return pc

    result['part_ids'] = np.load(npz_filename)['part_ids']
    
    if result['part_ids'] is not None: # has parts
        result['part_ids'] = result['part_ids'].astype(part_label_dtype)
        
        if n_samples is not None:
            result['part_ids'] = result['part_ids'][selected_idx]
            
    return result


def simple_pc_transform(pc, n_samples=None, random_seed=None, scale_pc=False):
    if n_samples is not None:
        pc = uniform_subsample(pc, n_samples=n_samples, random_seed=random_seed)[0]
        if scale_pc:
            pc = center_in_unit_sphere(pc)
    return pc


def prepare_vanilla_pointcloud_datasets(args):
    """ Good if you want to train a simple PC-AE or PC-Classifier.
    :param args:
    :return:
    """
    print('Loading ALL pc-data in memory. This can make the deep-net I/O faster but it is optional as it might have a '
          'large memory footprint.')

    split_df = pd.read_csv(args.split_file)

    if len(args.restrict_shape_class) > 0:
        restriction = set(list(args.restrict_shape_class))
        print('Restricting training to shape classes:', restriction)
        split_df = split_df[split_df.object_class.isin(restriction)].copy()
        split_df.reset_index(inplace=True, drop=True)

    if args.debug:
        print('Debugging! will only keep up to 1K point-clouds.')
        split_df = split_df.sample(min(len(split_df), 1000), random_state=args.random_seed)
        split_df.reset_index(inplace=True, drop=True)

    split_df['file_name'] = model_to_file_name(split_df, 'model_name', 'object_class', 'dataset', args.data_dir, '.npz')
    assert split_df['file_name'].apply(osp.exists).all(), 'files/models in the split file should exist on the hard drive!'

    # Add class-label-to-idx
    all_train_classes = split_df[split_df.split == 'train']['object_class'].unique()
    all_train_classes = sorted(all_train_classes)

    if not hasattr(args, 'n_classes') or args.n_classes is None:
        args.n_classes = len(all_train_classes)
    else:
        if args.n_classes != len(all_train_classes):
            raise ValueError('The number of object classes of the split-file are not equal to the user-provided one.')

    class_name_to_idx = {name: i for i, name in enumerate(all_train_classes)}
    # not train examples that do not fall into the train classes will be assigned -1:
    split_df = split_df.assign(object_class_int=split_df.object_class.apply(lambda x: class_name_to_idx.get(x, -1)))

    datasets = dict()
    example_cnt = 0
    for split in ['train', 'test', 'val']:
        split_data = split_df[split_df.split == split].copy()
        split_data.reset_index(drop=True, inplace=True)
        split_pcs = np.array(parallel_apply(split_data.file_name, partial(pc_loader_from_npz, only_pc=True)))
        pc_sampling_random_seed = None
        if args.deterministic_point_cloud_sampling or split in ['test', 'val']:
            pc_sampling_random_seed = args.random_seed
        
        scale_in_u_sphere = False
        if hasattr(args, "scale_in_u_sphere") and args.scale_in_u_sphere:            
            scale_in_u_sphere = True
                            
        dataset = PointcloudDataset(pointclouds=split_pcs,
                                    model_classes=split_data.object_class_int,
                                    model_metadata=split_data,
                                    pc_transform=partial(simple_pc_transform,
                                                         n_samples=args.n_pc_points,
                                                         random_seed=pc_sampling_random_seed,
                                                         scale_pc=scale_in_u_sphere
                                                         ))
                
        datasets[split] = dataset
        print(f'Number of {split} examples: {len(dataset)}')
        example_cnt += len(dataset)
    print(f'Total number of examples: {example_cnt}')
    print(f'Total number of training classes: {len(all_train_classes)}')
    return datasets, class_name_to_idx


def prepare_pointcloud_dataloaders(datasets, args):
    data_loaders = dict()
    for split, dataset in datasets.items():
        seed = None
        if args.deterministic_point_cloud_sampling or split in ['test', 'val']:
            seed = args.random_seed

        batch_size = args.batch_size
        data_loaders[split] = DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=split == 'train',
                                         num_workers=args.num_workers,
                                         worker_init_fn=lambda x: np.random.seed(seed=seed))

    return data_loaders


def deterministic_data_loader(data_loader, **kwargs):

    default_bsize = data_loader.batch_size
    default_n_workers = data_loader.num_workers
    default_worker_init_fn = data_loader.worker_init_fn

    batch_size = kwargs.get('batch_size', default_bsize)
    n_workers = kwargs.get('n_workers', default_n_workers)
    worker_init_fn = kwargs.get('n_workers', default_worker_init_fn)

    print('Used bsize, n-workers:', batch_size, n_workers)

    result = DataLoader(data_loader.dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=n_workers,
                        worker_init_fn=worker_init_fn)

    return result


##
# Simple Pointcloud Utilities
##


def uniform_subsample(points, n_samples, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)
    return points[idx], idx


def swap_axes_of_pointcloud(pointcloud, permutation):
    """
    :param pointcloud: 2-dimensional numpy/torch array: N-points x 3
    :param permutation: a permutation of [0,1,2], e.g., [0,2,1]
    :return:
    """
    v = pointcloud
    nv = len(pointcloud)
    vx = v[:, permutation[0]].reshape(nv, 1)
    vy = v[:, permutation[1]].reshape(nv, 1)
    vz = v[:, permutation[2]].reshape(nv, 1)
    pointcloud = np.hstack((vx, vy, vz))
    return pointcloud


def swap_axes_of_pointcloud_batch(pointcloud, permutation):
    """
    :param pointcloud: B x N-points x 3
    :param permutation: a permutation of [0,1,2], e.g., [0,2,1]
    :return:
    """

    x = pointcloud[:, :, permutation[0]]
    y = pointcloud[:, :, permutation[1]]
    z = pointcloud[:, :, permutation[2]]

    new_pc = [x, y, z]

    if type(pointcloud) in [np.ndarray, np.array]:
        pointcloud = np.stack(new_pc, 2)
    else:
        pointcloud = torch.cat(new_pc)

    return pointcloud


def center_in_unit_sphere(pc, in_place=True):
    if not in_place:
        pc = pc.copy()

    for axis in range(3):  # center around each axis
        r_max = np.max(pc[:, axis])
        r_min = np.min(pc[:, axis])
        gap = (r_max + r_min) / 2.0
        pc[:, axis] -= gap

    largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= largest_distance
    return pc


def is_centered_in_unit_sphere(pc, epsilon=1e-6):
    largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if abs(largest_distance - 1) > epsilon:
        return False
    if any(abs(np.max(pc, 0) + np.min(pc, 0)) > 1e-6):
        return False
    return True


def rotate_z_axis_by_degrees(pc, theta, clockwise=True):
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t, 0],
                                [sin_t, cos_t, 0],
                                [0, 0, 1]], dtype=pc.dtype)
    if not clockwise:
        rotation_matrix = rotation_matrix.T

    return pc.dot(rotation_matrix)
