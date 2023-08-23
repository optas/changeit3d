"""
Originally created at 5/24/21, for Python 3.x
2022 Panos Achlioptas
"""

import pandas as pd
import os.path as osp


SHAPE_TALK_CLASSES = [
    'airplane',
    'bag',
    'bathtub',
    'bed',
    'bench',
    'bookshelf',
    'bottle',
    'bowl',
    'cabinet',
    'cap',
    'chair',
    'clock',
    'display',
    'dresser',
    'faucet',
    'flowerpot',
    'guitar',
    'helmet',
    'knife',
    'lamp',
    'mug',
    'person',
    'pistol',
    'plant',
    'scissors',
    'skateboard',
    'sofa',
    'table',
    'trashbin',
    'vase']


# describe the classes of objects that where mixed together across ShapeNet, ModelNet, PartNet
# to create the corresponding class in ShapeTalk
# Naming Convention followed for the values of the below dictionary is: (ShapeNet, ModelNet, PartNet)
shapetalk_class_mixing = dict()
shapetalk_class_mixing['airplane'] = ('airplane', 'airplane', None)
shapetalk_class_mixing['bag'] = ('bag', None, 'bag')
shapetalk_class_mixing['bathtub'] = ('bathtub', 'bathtub', None)
shapetalk_class_mixing['bed'] = ('bed', 'bed', None)
shapetalk_class_mixing['bench'] = ('bench', 'bench', None)
shapetalk_class_mixing['bookshelf'] = ('bookshelf', 'bookshelf', None)
shapetalk_class_mixing['bottle'] = ('bottle', None, None)  # MN "bottle" are duplicates of SN
shapetalk_class_mixing['bowl'] = ('bowl', 'bowl', 'bowl')
shapetalk_class_mixing['cabinet'] = ('file', None, None)
shapetalk_class_mixing['cap'] = ('cap', None, 'hat')
shapetalk_class_mixing['chair'] = ('chair', 'chair', None)
shapetalk_class_mixing['clock'] = ('clock', None, None)
shapetalk_class_mixing['display'] = ('monitor', 'monitor', None)
shapetalk_class_mixing['dresser'] = ('cabinet', 'dresser', None)
shapetalk_class_mixing['faucet'] = ('faucet', None, None)
shapetalk_class_mixing['flowerpot'] = ('pot', 'flower_pot', None)
shapetalk_class_mixing['guitar'] = ('guitar', None, None)  # MN guitar are duplicates of SN
shapetalk_class_mixing['helmet'] = ('helmet', None, None)
shapetalk_class_mixing['knife'] = ('knife', None, None)
shapetalk_class_mixing['lamp'] = ('lamp', 'lamp', None)
shapetalk_class_mixing['mug'] = ('mug', None, None)  # "cup" exists in MN but all duplicates and/or bad alignments
shapetalk_class_mixing['person'] = (None, 'person', None)
shapetalk_class_mixing['pistol'] = ('pistol', None, None)
shapetalk_class_mixing['plant'] = (None, 'plant', None)
shapetalk_class_mixing['scissors'] = (None, None, 'scissors')
shapetalk_class_mixing['skateboard'] = ('skateboard', None, None)
shapetalk_class_mixing['sofa'] = ('sofa', 'sofa', None)
shapetalk_class_mixing['table'] = ('table', ['table', 'desk'], None)  # for tables: in MN we used both table/desk classes, to be similar with SN's "table" category
shapetalk_class_mixing['trashbin'] = ('can', None, None)
shapetalk_class_mixing['vase'] = ('jar', 'vase', None)


def original_class_name(shapetalk_class_name, dataset):
    possibilities = shapetalk_class_mixing[shapetalk_class_name]
    if dataset == 'ShapeNet':
        return possibilities[0]
    elif dataset == 'ModelNet':
        return possibilities[1]
    elif dataset == 'PartNet':
        return possibilities[2]
    else:
        raise ValueError('Unknown dataset')


def add_image_files(shape_talk_df, top_img_dir, ending='.png'):
    for model_type in ['source', 'target']:
        new_column = model_type + '_' + 'image_file_name'
        shape_talk_df[new_column] = model_to_file_name(shape_talk_df,
                                                       f'{model_type}_model_name',
                                                       f'{model_type}_object_class',
                                                       f'{model_type}_dataset',
                                                       top_dir=top_img_dir,
                                                       ending=ending)
    return shape_talk_df


def add_pointcloud_files(shape_talk_df, top_pc_dir, ending='.npz'):
    for model_type in ['source', 'target']:
        new_column = model_type + '_' + 'pointcloud_file_name'
        filenames = model_to_file_name(shape_talk_df,
                                       f'{model_type}_model_name',
                                       f'{model_type}_object_class',
                                       f'{model_type}_dataset',
                                       top_dir=top_pc_dir,
                                       ending=ending)
        shape_talk_df = shape_talk_df.assign(**{new_column: filenames})
    return shape_talk_df


def model_to_file_name(df, model_name_col, object_class_col, dataset_col, top_dir, ending):
    """
    :param df:
    :param model_name_col: (string)
    :param object_class_col: (string)
    :param dataset_col: (string)
    :param top_dir: (string)
    :param ending: (string)
    :return:
    """
    def join_parts(row):
        return osp.join(top_dir, row[object_class_col], row[dataset_col], row[model_name_col] + ending)
    return df.apply(join_parts, axis=1)


def model_to_uid(df, model_name_col, object_class_col, dataset_col):
    return df[object_class_col] + '/' + df[dataset_col] + '/' + df[model_name_col]


def expand_df_from_descriptions_to_utterances(df, max_utters=5):
    # expand df rows to contain separately the utterances/saliencies
    df_expanded = []
    for i in range(max_utters):
        current = df.copy()
        current = current.assign(saliency=pd.Series([i] * len(current)))

        for drop_salieny in set(range(max_utters)).difference([i]):            
            current = current.drop(columns=f'utterance_{drop_salieny}')

        current = current.rename(columns={f'utterance_{i}': 'utterance'})
        df_expanded.append(current)

    df_expanded = pd.concat(df_expanded, ignore_index=True)
    df_expanded.reset_index(inplace=True, drop=True)

    none_mask = df_expanded.utterance.isna()
    none_mask |= df_expanded.utterance.apply(lambda x: x is None)

    df_expanded = df_expanded[~none_mask]  # remove None/VOID "utterances"
    df_expanded.reset_index(inplace=True, drop=True)

    return df_expanded