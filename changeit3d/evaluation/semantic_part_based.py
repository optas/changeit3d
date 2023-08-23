import numpy as np
from changeit3d.in_out.datasets.shape_net_parts import canonical_part_names_to_relevant_words, idx_to_segmentation_part_name



def sentence_to_its_referred_canonical_parts(tokens, canonical_names):
    """   ["a", "chair" "with" "fine", "looking", "legs"] -> "leg"
    :param tokens:
    :param canonical_names:
    :return:
    """
    tokens = set(tokens)
    result = set()
    for key, val in canonical_names.items():
        if len(tokens.intersection(set(val))) > 0:
            result.add(key)
    return result


def mark_part_reference_in_sentences(sentences_tokenized, shape_classes,
                                     canonical_part_names=canonical_part_names_to_relevant_words,
                                     idx_to_segmentation_part_name=idx_to_segmentation_part_name,
                                     part_dataset='shape_net_parts'):

    if part_dataset != 'shape_net_parts':
        raise NotImplementedError()
        
    if len(sentences_tokenized) != len(shape_classes):
        raise ValueError()
    
    referred_parts = []
    referred_parts_idx = []
    for tokens, shape_class in zip(sentences_tokenized, shape_classes):
        canonical_part_of_class = canonical_part_names[shape_class]
        ref_parts_names = sentence_to_its_referred_canonical_parts(tokens, canonical_part_of_class)
        referred_parts.append(ref_parts_names)
        if idx_to_segmentation_part_name is not None:
            idx_to_name = idx_to_segmentation_part_name[shape_class]
            name_to_idx = {j: i for i, j in idx_to_name.items()}
            referred_parts_idx.append([name_to_idx[p] - 1 for p in ref_parts_names])  # -1 since enumeration of
                                                                                      # part_name_to_idx starts from 1 for shape_net_parts
    return referred_parts, referred_parts_idx


def masks_of_referred_parts_for_pcs(pointclouds, predicted_parts, rows_to_part_idx):
    n_points = pointclouds.shape[1]
    masks = np.ones((len(pointclouds), n_points), dtype=bool)
    for row, part_idx in rows_to_part_idx.items():
        white_mask = np.ones(n_points, dtype=bool)
        for p in part_idx:
            white_mask &= predicted_parts[row] != p
        masks[row] = white_mask
    return masks
