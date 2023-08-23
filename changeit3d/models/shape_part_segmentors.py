import torch
from ..external_tools.part_segmentation_pytorch.pointnet.model import PointNetDenseCls
from ..in_out.datasets.shape_net_parts import ShapeNetParts

##
## shape_net_parts: i.e., use the code of Fei trained with Eric's shapenetcore parts
##

@torch.no_grad()
def shape_net_parts_segmentor_inference(segmentor, dataloader, bcn_format=True, device='cuda'):
    segmentor.eval()
    all_predictions = []
    for batch in dataloader:
        pc = batch['pointcloud'].to(device)

        if bcn_format:
            pc = pc.transpose(2, 1).contiguous()

        prediction_logits = segmentor(pc)[0]
        prediction = torch.argmax(prediction_logits, -1)
        all_predictions.append(prediction.cpu())
    return torch.cat(all_predictions).numpy()


def load_shape_net_parts_segmentor(segmentor_file, shape_class, feature_transform=False):
    n_parts = ShapeNetParts.n_parts[shape_class]
    part_segmentor = PointNetDenseCls(k=n_parts, feature_transform=feature_transform)
    part_segmentor.load_state_dict(torch.load(segmentor_file))
    return part_segmentor