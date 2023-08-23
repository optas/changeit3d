"""
Functions to define different neural-models.

Originally created at 4/22/21, for Python 3.x
Panos Achlioptas (https://optas.github.io/)
"""

import os.path as osp
import warnings
from torch import nn
from .mlp import MLP
from .point_net import PointNet
from .pointcloud_classifier import PointcloudClassifier
from .pointcloud_autoencoder import PointcloudAutoencoder
from .listening_oriented import default_lstm_encoder
from .basic_ops_as_modules import ReLU
from .changeit3d_net import LatentDirectionFinder
from .monolithic_changeit3d import MonolithicChangeIt3D
from .listening_oriented import TransformerModel, TransformerModelFeature
from ..in_out.basics import read_saved_args, load_state_dicts
from ..in_out.changeit3d_net import load_pickled_shape_latent_codes
from ..language.vocabulary import Vocabulary


##
# PC-AE model
##

def describe_pc_ae(args):
    # Make an AE.
    if args.encoder_net == 'pointnet':
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()
        
    if args.decoder_net == 'mlp':
        ae_decoder = MLP(in_feat_dims=encoder_latent_dim,
                         out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
                         b_norm=False)

    model = PointcloudAutoencoder(ae_encoder, ae_decoder)
    return model


def load_pretrained_pc_ae(model_file):
    config_file = osp.join(osp.dirname(model_file), 'config.json.txt')
    pc_ae_args = read_saved_args(config_file)    
    pc_ae = describe_pc_ae(pc_ae_args)
    
    if osp.join(pc_ae_args.log_dir, 'best_model.pt') != osp.abspath(model_file):
        warnings.warn("The saved best_model.pt in the corresponding log_dir is not equal to the one requested.")

    best_epoch = load_state_dicts(model_file, model=pc_ae)
    print(f'Pretrained PC-AE is loaded at epoch {best_epoch}.')
    return pc_ae, pc_ae_args


##
# PC-CLF model
##

def pnet_classifier_default_encoder():
    """Default encoder construction as described in original paper (modulo the lack of feature transform) """
    encoder = PointNet(init_feat_dim=3, conv_dims=[64, 64, 128, 128, 1024])
    return encoder


def pnet_classifier_default_decoder(n_classes):
    """Default decoder construction as described in original paper (modulo the lack of feature transform) """
    decoder = nn.Sequential(
        nn.Linear(1024, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
    )
    return decoder


def describe_pc_clf(args):
    # Make a PC-shape CLF.

    # encoder
    if args.encoder_net == 'pointnet':
        if args.encoder_conv_layers is None: # default parameters
            clf_encoder = pnet_classifier_default_encoder()
            encoding_feat_dim = 1024
        else:
            clf_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
            encoding_feat_dim = args.encoder_conv_layers[-1]

    else:
        raise NotImplementedError

    # decoder
    if args.decoder_fc_neurons is None: # default parameters
        clf_decoder = pnet_classifier_default_decoder(args.n_classes)
    else:
        clf_decoder = MLP(in_feat_dims=encoding_feat_dim,
                          out_channels=args.decoder_fc_neurons + [args.n_classes],
                          b_norm=False)

    model = PointcloudClassifier(clf_encoder, clf_decoder)
    
    print('\nArchitecture of Classifier\n')
    print(model, '\n')
    return model



##
# ChangeIt Models and Ablations
##

def ablations_changeit3d_net(vocab, shape_latent_dim, ablation_version, self_contrast=True):                 
    d_lang_model= 128        
    in_dim = d_lang_model + shape_latent_dim
            
    editor = MLP(in_dim, [256, shape_latent_dim, shape_latent_dim, shape_latent_dim], b_norm=True, remove_final_bias=True)
    stimulus_encoder = MLP(shape_latent_dim, [shape_latent_dim, shape_latent_dim])    
    closure = ReLU()
            
    if ablation_version == 'decoupling_mag_direction':
        magnitude_encoder = MLP(in_dim, [256, 128, 64, 1], closure=closure)
        unit_normalize_direction = True
    elif ablation_version == 'coupled':        
        unit_normalize_direction = False
        magnitude_encoder = None    
    else:
        raise ValueError('ablation version of ChangeIt3D not understood.')

    print('Doing ST ablation', ablation_version, 'with self contrast', self_contrast)

    nhead = 2
    d_hid = 128
    nlayers = 2
    language_dropout = 0.2
    language_model = TransformerModel(len(vocab), d_lang_model, nhead, d_hid, nlayers, language_dropout)
    language_encoder = TransformerModelFeature(language_model)
    
    model = LatentDirectionFinder(language_encoder,
                                  stimulus_encoder,
                                  editor,
                                  magnitude_unit=magnitude_encoder,
                                  unit_normalize_direction=unit_normalize_direction,
                                  self_contrast=self_contrast)

    return model


def monolithic_alternative_to_changeit3d(n_pc_points, vocab):
    ## you can obviously play-around with the architecture's params, below I re-use the basic structure (arhcitecture's parameters) used by PC-AE & our LSTM-based listeners
     
    
    shape_encoder_dim = 256    
    shape_encoder = PointNet(init_feat_dim=3, conv_dims=[32, 64, 64, 128, shape_encoder_dim])    
    

    word_embedding_dim = lstm_n_hidden = 128
    word_dropout = 0.10
    # for multi-utter listening experiments LSTMs perform betten than Transformers
    language_encoder = default_lstm_encoder(vocab,                            
                                            word_embedding_dim,
                                            lstm_n_hidden,
                                            word_dropout=word_dropout)
    
    
    fusion_dim = 128    
    fuser = MLP(in_feat_dims=lstm_n_hidden+shape_encoder_dim, out_channels=[256, 128, fusion_dim])
    
    decoder_fc_neurons = [256, 256, 512]
    decoder = MLP(in_feat_dims=fusion_dim, out_channels=decoder_fc_neurons + [n_pc_points * 3], b_norm=False)
    
    model = MonolithicChangeIt3D(visual_encoder=shape_encoder, 
                                 language_encoder=language_encoder,
                                 fuser=fuser,
                                 decoder=decoder, 
                                 n_pc_points=n_pc_points)
    
    return model


def load_pretrained_changeit3d_net(checkpoint_file, shape_latent_dim=None, vocab=None):
    config_file = osp.join(osp.dirname(checkpoint_file), 'config.json.txt')    
    args = read_saved_args(config_file)
    
    if shape_latent_dim is None:
        shape_latent_dim = load_pickled_shape_latent_codes(args) 
    
    if vocab is None:
        vocab = Vocabulary.load(args.vocab_file)
                
    model = ablations_changeit3d_net(vocab, shape_latent_dim, args.shape_editor_variant, args.self_contrast)
    
    best_epoch = load_state_dicts(checkpoint_file, model=model, map_location="cpu")
    model = model.eval()
    
    return model, best_epoch, args
