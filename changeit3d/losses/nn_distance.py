import torch


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2


def chamfer_loss(pc_a, pc_b, swap_axes=False, return_separately=True):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :return: B floats, indicating the chamfer distances
    # Note: this is 10x slower than the chamfer_loss in losses/chamfer.py BUT this plays also in CPU (the
    other does not).
    """
    if swap_axes:
        pc_a = pc_a.transpose(-1, -2).contiguous()
        pc_b = pc_b.transpose(-1, -2).contiguous()
    dist_a, _, dist_b, _ = nn_distance(pc_a, pc_b)
    dist = dist_a.mean(1) + dist_b.mean(1)  # reduce separately, sizes of points can be different
    if return_separately:
        return dist, dist_a.mean(1), dist_b.mean(1)
    else:
        return dist