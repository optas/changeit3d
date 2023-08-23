from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def load_part_segmentator(best_model_name):
    ''' Load and return segmentation model
        args:
            best_model_name : the name of the model to use (str)
    '''
    state_dict = torch.load(best_model_name)
    classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
    classifier.load_state_dict(state_dict)
    return classifier


def test_segmentation(best_model_name, pc):
    ''' Inference of part segmentator for a given point cloud
        returns the predicted values
        args:
            best_model_name : the name of the model to use (str)
            pc : the point cloud to process (N,3)
        output: 
            pred_choice : part prediction per point (N,1) 
    '''
    classifier = load_part_segmentator(best_model_name)
    classifier.eval()

    pc = pc.transpose(1, 0).contiguous()
    pc = Variable(pc.view(1, pc.size()[0], pc.size()[1]))
    pred, _, _ = classifier(pc)
    pred_choice = pred.data.max(2)[1]
    return pred_choice


def save_pred_gt_pcs(pc, pred, gt_seg, filename):
    ''' Export in txt format the original point cloud with colors 
        for ground truth and predicted part segmentation
        args:
            pc : the point cloud (N,3)
            pred : the part prediction per point (N,1)
            gt_seg : the ground truth part label per point (N,1)
            filename : the name to use for exporting the files (str)
    '''
    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    gt_color = cmap[gt_seg.numpy() - 1, :]
    pred_color = cmap[pred.numpy()[0], :]
    BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
    out_vis = os.path.join(BASE_DIR, 'seg_viz')
    os.makedirs(out_vis, exist_ok=True)
    np.savetxt(os.path.join(out_vis,filename+'_pred.txt'),np.hstack((pc,pred_color)))
    np.savetxt(os.path.join(out_vis,filename+'_gt.txt'),np.hstack((pc,gt_color)))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='seg/best_seg_model_Chair.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='/home/ir0/data/language_faders/language_faders/external/language_changes/language_changes/external_tools/part_segmentation_pytorch/data', help='dataset path')
    parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')
    parser.add_argument('--vis', action='store_true', default=False, help='if true exports pc for prediction and groundtruth')

    opt = parser.parse_args()
    print(opt)

    d = ShapeNetDataset(
        root=opt.dataset,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)

    idx = np.random.randint(len(d))
    print("model %d/%d" % (idx, len(d)))
    pc, gt_seg = d[idx]
    print(pc.size(), gt_seg.size())

    if len(opt.model)==0:
        opt.model='seg/best_seg_model_'+opt.class_choice+'.pth'
    pred = test_segmentation(opt.model, pc)
    
    #export visualizations
    if opt.vis:
        print("Exporting visualizations")
        pc_np = pc.numpy()
        filename=opt.class_choice+f'_{idx}'
        save_pred_gt_pcs(pc, pred, gt_seg, filename)