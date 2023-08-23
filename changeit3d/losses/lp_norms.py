import torch


def safe_l2(tensor, dim=-1, epsilon=1e-16, keepdim=False, reduce=None):
    """ Even at zero it will return ~epsilon.
    Reminder: l2_norm has no derivative at 0.0.
    :param tensor:
    :param dim:
    :param epsilon:
    :param keepdim:
    :param reduce:
    :return:
    """

    squared_norm = torch.sum(torch.pow(tensor, 2), dim=dim, keepdim=keepdim)
    norm = torch.sqrt(squared_norm + epsilon)

    if reduce is None:
        return norm
    elif reduce == "mean":
        return torch.mean(norm)
    else:
        raise ValueError()
