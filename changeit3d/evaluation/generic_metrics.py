import torch
import numpy as np
import torch.nn.functional as F

from ..losses.chamfer import chamfer_loss
from ..utils.basics import iterate_in_chunks


def chamfer_dists(original_shapes, transformed_shapes, bsize, device="cuda"):
    n_items = len(original_shapes)
    all_dists = []
    assert n_items == len(transformed_shapes)
    for locs in iterate_in_chunks(range(n_items), bsize):
        o = torch.Tensor(original_shapes[locs]).to(device)
        t = torch.Tensor(transformed_shapes[locs]).to(device)        
        mu = chamfer_loss(o, t, swap_axes=False, reduction='mean')
        all_dists.append(mu.cpu())
    all_dists = torch.cat(all_dists)
    return torch.mean(all_dists).item(), all_dists


def chamfer_dists_on_masked_pointclouds(pointclouds_a, pointclouds_b, masks_a, masks_b, bsize, device='cuda'):
    """
    :param pointclouds_a: np.array, B x pc-points x 3
    :param pointclouds_b: np.array, B x pc-points x 3
    :param masks_a: np.array, B x pc-points, boolean mask carrying a white list (1s) of points that matter for the
        computation of the Chamfer, per pointcloud for pointclouds_a. 0s are ignored.
    :param masks_b: same as ``masks_a`` for pointclouds_b
    :param bsize:
    :param device:
    :return:

    """
    n_items = len(pointclouds_a)
    all_dists = []
    assert n_items == len(pointclouds_b) == len(masks_a) == len(masks_b)
    for locs in iterate_in_chunks(range(n_items), bsize):
        a = torch.Tensor(pointclouds_a[locs]).to(device)
        b = torch.Tensor(pointclouds_b[locs]).to(device)

        ma = ~masks_a[locs]
        mb = ~masks_b[locs]
        a[ma, :] = 10e6     # assign a large value to ensure that this points will be matched with themselves
        b[mb, :] = 10e6

        cd_a, cd_b = chamfer_loss(a, b, swap_axes=False, reduction=None)

        cd_a = cd_a.cpu().numpy()
        cd_b = cd_b.cpu().numpy()

        a = cd_a * masks_a[locs]  # zero-out-non whitelisted points/dists
        b = cd_b * masks_b[locs]

        n_a_points = masks_a[locs].sum(1)
        n_b_points = masks_b[locs].sum(1)

        mu = (a.sum(1) + b.sum(1)) / (n_a_points + n_b_points)  # weighted sum
        all_dists.append(mu)
    all_dists = np.hstack(all_dists)
    return np.mean(all_dists), all_dists


@torch.no_grad()
def get_clf_probabilities(clf, shapes,
                          clf_feed_key=None, clf_res_key=None,
                          channel_last=True, bsize=128, device="cuda"):
    all_probs = list()
    n_items = len(shapes)
    for locs in iterate_in_chunks(range(n_items), bsize):
        b_shapes = torch.Tensor(shapes[locs]).to(device)

        if channel_last:
            b_shapes.transpose_(2, 1)

        if clf_feed_key is not None:
            b_shapes = {clf_feed_key: b_shapes}

        logits = clf(b_shapes)

        if clf_res_key is not None:
            logits = logits[clf_res_key]

        p = F.softmax(logits, 1).cpu()
        all_probs.append(p)
    all_probs = torch.cat(all_probs)
    return all_probs


def difference_in_probability(original_probabilities, transformed_probabilities, gt_class_labels):
    ## we could also apply this over the entire domain of all classes via KL, EMD etc.
    diff_in_prob = torch.abs(transformed_probabilities - original_probabilities)
    return diff_in_prob[torch.arange(len(diff_in_prob)), gt_class_labels].mean().item()
