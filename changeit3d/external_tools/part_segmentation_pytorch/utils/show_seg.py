from __future__ import print_function
#from show3d_balls import showpoints
import os
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import sys
user = 'ir0'
if user =='ir0':
    home_dir = '/home/ir0/data/language_faders/language_faders/external/language_changes/language_changes/external_tools/part_segmentation_pytorch'
    sys.path.append(home_dir)

from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx
print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
out_vis = os.path.join(BASE_DIR, 'seg_viz')
os.makedirs(out_vis, exist_ok=True)
np.savetxt(os.path.join(out_vis,opt.class_choice+f'_{idx}_pred.txt'),np.hstack((point_np,pred_color)))
np.savetxt(os.path.join(out_vis,opt.class_choice+f'_{idx}_gt.txt'),np.hstack((point_np,gt)))


#print(pred_color.shape)
#showpoints(point_np, gt, pred_color)
