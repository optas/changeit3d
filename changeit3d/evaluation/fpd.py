"""Calculate Frechet Pointcloud Distance referened by Frechet Inception Distance."
    [ref] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
    github code  : (https://github.com/bioinf-jku/TTUR)
    paper        : (https://arxiv.org/abs/1706.08500)
    
** Code adapted from: https://github.com/jtpils/TreeGAN/blob/master/evaluation/FPD.py **
"""

import types    
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from scipy.linalg import sqrtm
from ..utils.basics import iterate_in_chunks


def get_activations(pointclouds, model, feature_ids, batch_size=100, device=None, verbose=True, channel_fist=True):    
    model, features = model.collect_features(device, feature_ids=feature_ids)
    
    if channel_fist:
        pointclouds = pointclouds.transpose(1, 2)

    for i, pointcloud_batch in enumerate(iterate_in_chunks(pointclouds, batch_size)):
        if verbose:
            print(f'Computing feature maps batch {i}')

        if device is not None:
            pointcloud_batch = pointcloud_batch.to(device)

        pointcloud_batch = {'pointcloud': pointcloud_batch}
        model(pointcloud_batch)  # will store features

    features = [np.vstack(f) for f in features.values()]  # merge across batches
    features = np.hstack(features)  # concatenate across feature maps/channels
    if verbose:
        print('Shape of collected feature maps is', features.shape)
    return features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    ssdiff = np.sum(diff**2.0)

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def calculate_activation_statistics(pointclouds, model, feature_ids, batch_size=100, device=None, verbose=False):
    act = get_activations(pointclouds, model, feature_ids, batch_size, device, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


@torch.no_grad()
def calculate_fpd(pointclouds_set_a, pointclouds_set_b, pretrained_model_file, feature_ids=['fc2'], 
                  batch_size=100, device="cuda", architecture="pointnet_with_default_params", verbose=False):    
    """Calculates the FPD of two pointclouds given a pre-trained model
    
    Note: currently works only when pre-trained model was built with the DEFAULT arch. parameters (args) of 
    returned by changeit3d.in_out.arguments.parse_train_test_pc_clf_arguments.     
    """
    model = torch.load(pretrained_model_file).to(device)
    
    # equip model with "named" feature-extraction capabilities    
    if architecture == "pointnet_with_default_params":
        model.collect_features = types.MethodType(collect_features_on_pnet_classifier_built_with_default_params, model)
    else:
        raise NotImplementedError("For a different PC architecture you have to implement the feature extractor yourself.")
    
    m1, s1 = calculate_activation_statistics(pointclouds_set_a, model, feature_ids, batch_size, device, verbose=verbose)
    m2, s2 = calculate_activation_statistics(pointclouds_set_b, model, feature_ids, batch_size, device, verbose=verbose)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


@torch.no_grad()
def collect_features_on_pnet_classifier_built_with_default_params(self, device, feature_ids=['maxpool', 'fc1', 'fc2']):
    """
    Add forward hooks that store the feature output at specific locations of the network.
    The network ("self") is expected to be build with the default parameters of:
                changeit3d.in_out.arguments.parse_train_test_pc_clf_arguments()
    Args:
        device:
        feature_ids:
    Returns:
    """

    model = self.to(device).eval()
    features = defaultdict(list)

    if 'maxpool' in feature_ids:        
        def hook_1(module, hook_input, hook_output):
            out = F.max_pool1d(hook_output, kernel_size=hook_output.shape[-1]).squeeze_(-1)
            features['maxpool'].append(out.cpu().numpy())
        self.encoder.pre_pooling_feature.register_forward_hook(hook_1)

    if 'fc1' in feature_ids:
        def hook_2(module, hook_input, hook_output):
            features['fc1'].append(hook_output.cpu().numpy())
        self.decoder[0].register_forward_hook(hook_2)   # fc1 (linear) is at location [0] of decoder: fc/bnorm/relu

    if 'fc2' in feature_ids:
        def hook_3(module, hook_input, hook_output):
            features['fc2'].append(hook_output.cpu().numpy())
        self.decoder[3].register_forward_hook(hook_3)  # fc2 (linear) is at location [3] of decoder: fc/bnorm/relu

    return model, features