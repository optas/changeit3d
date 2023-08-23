import torch
import warnings

try:
    from .ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
    chamfer_raw = dist_chamfer_3D.chamfer_3DDist()
    efficient_chamfer = True
except:
    raise
    # warnings.warn('A cuda-based (efficient) chamfer implementation is not installed/found! Using a native pytorch '
    #               'implementation which was NOT used for the experiments of this paper.')
    # from .nn_distance import chamfer_loss as chamfer_raw
    # efficient_chamfer = False


def chamfer_loss(pc_a, pc_b, swap_axes=False, reduction='mean'):
    """Compute the chamfer loss for batched pointclouds.
        :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
        :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
        :return: B floats, indicating the chamfer distances when reduction is mean, else un-reduced distances
    """

    n_points_a = pc_a.shape[1]
    n_points_b = pc_b.shape[1]

    if swap_axes:
        pc_a = pc_a.transpose(-1, -2).contiguous()
        pc_b = pc_b.transpose(-1, -2).contiguous()

    if efficient_chamfer:
        dist_a, dist_b, _, _ = chamfer_raw(pc_a, pc_b)
    else:
        _, dist_a, dist_b = chamfer_raw(pc_a, pc_b)

    if reduction == 'mean':
        # reduce separately, sizes of points can be different
        dist = ((n_points_a * dist_a.mean(1)) + (dist_b.mean(1) * n_points_b)) / (n_points_a + n_points_b)
    elif reduction is None:
        return dist_a, dist_b
    else:
        raise ValueError('Unknown reduction rule.')
    return dist


if __name__ == '__main__':
    pca = torch.rand(10, 2048, 3).cuda()
    pcb = torch.rand(10, 4096, 3).cuda()
    a, b = chamfer_loss(pca, pcb, reduction=None)
    print(a.shape, b.shape)
    l = chamfer_loss(pca, pcb)
    print(l.shape)