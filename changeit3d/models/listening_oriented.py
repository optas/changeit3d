"""
Originally created at 2/17/21, for Python 3.x 
Panos Achlioptas (https://optas.github.io/)
"""

import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .lstm_encoder import LSTMEncoder
from .mlp import MLP
from .dgcnn import DGCNN
from .point_net import PointNet
from ..utils.stats import AverageMeter


class ContextFreeListener(nn.Module):
    def __init__(self, language_encoder, stimulus_encoder, classifier_head, lstm_based=True, geo_encoder_uses_lang=False):
        super().__init__()
        self.language_encoder = language_encoder
        self.stimulus_encoder = stimulus_encoder
        self.classifier_head = classifier_head
        self.geo_encoder_uses_lang = geo_encoder_uses_lang
        self.lstm_based = lstm_based

    def transformer_based_forward(self, tokens, stimuli):                
        logits = []        
        n_stimuli = stimuli.shape[1]

        lang_feat = self.language_encoder(tokens.t())  # seq_len, batch_size, d_model            
        lang_feat = lang_feat.max(0)[0]
                        
        for i in range(n_stimuli):
            visual_input = stimuli[:, i]            
            visual_feat = self.stimulus_encoder(visual_input)            
            feat = torch.cat([visual_feat, lang_feat], 1)
            logit_i = self.classifier_head(feat)                
            logits.append(logit_i)
        return torch.cat(logits, 1)

    def lstm_based_forward(self, tokens, stimuli):
        logits = []
        n_stimuli = stimuli.shape[1]
        for i in range(n_stimuli):
            grounding_input = stimuli[:, i]
            if self.geo_encoder_uses_lang:
                grounding_feat = self.stimulus_encoder(grounding_input, tokens)  # e.g., when using attention.
            else:
                grounding_feat = self.stimulus_encoder(grounding_input)
            grounded_lang_feat = self.language_encoder(tokens, grounding_feat)
            logit_i = self.classifier_head(grounded_lang_feat)
            logits.append(logit_i)
        return torch.cat(logits, 1)

    def forward(self, tokens, stimuli):
        """
        Stumuli: B x len(stimuli) x stimulus_dims
        """
        if self.lstm_based:
            return self.lstm_based_forward(tokens, stimuli)
        else:
            return self.transformer_based_forward(tokens, stimuli)
            

class ContextAwareListener(nn.Module):    
    def __init__(self, language_encoder, stimulus_encoder, context_encoder, classifier_head, lstm_based=False):
        super().__init__()
        self.language_encoder = language_encoder
        self.stimulus_encoder = stimulus_encoder
        self.classifier_head = classifier_head
        self.context_encoder = context_encoder
        self.lstm_based = lstm_based

    def lstm_based_forward(self, tokens, stimuli):
        n_stimuli = stimuli.shape[1]                
        vis_feats = []        
        for i in range(n_stimuli):
            grounding_feat = self.stimulus_encoder(stimuli[:, i])            
            vis_feats.append(grounding_feat)
        
        assert n_stimuli == 2
        context_feats = []
        for i in [1, 0]:
            context_feats.append(self.context_encoder(vis_feats[i]))
                        
        logits = []
        for vf, cf in zip(vis_feats, context_feats):
            fused_feat = torch.cat([vf, cf], 1)            
            final_feat = self.language_encoder(tokens, fused_feat)                    
            logits.append(self.classifier_head(final_feat))
                
        return torch.cat(logits, 1)

    def transformer_based_forward(self, tokens, stimuli):                
        n_stimuli = stimuli.shape[1]

        vis_feats = []        
        for i in range(n_stimuli):
            grounding_feat = self.stimulus_encoder(stimuli[:, i])            
            vis_feats.append(grounding_feat)        
        
        assert n_stimuli == 2
        context_feats = []
        for i in [1, 0]:
            context_feats.append(self.context_encoder(vis_feats[i]))
        
        lang_feat = self.language_encoder(tokens.t())  # seq_len, batch_size, d_model            
        lang_feat = lang_feat.max(0)[0]

        logits = []
        for vf, cf in zip(vis_feats, context_feats):
            fused_feat = torch.cat([vf, cf, lang_feat], 1)            
            logits.append(self.classifier_head(fused_feat))
        
        return torch.cat(logits, 1)

    def forward(self, tokens, stimuli):
        """
        Stumuli: B x len(stimuli) x stimulus_dims
        """
        if self.lstm_based:
            return self.lstm_based_forward(tokens, stimuli)
        else:
            return self.transformer_based_forward(tokens, stimuli)


class PositionalEncoding(nn.Module):
    """
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerModelFeature(nn.Module):
    def __init__(self, transformer_model, pool='maxpool'):
        super().__init__()
        self.model = transformer_model
        self.pool = pool
    
    def forward(self, tokens):
        lang_feat = self.model(tokens.t())
        if self.pool == "maxpool":
            lang_feat = lang_feat.max(0)[0]
        return lang_feat
        


##                      ##
##   Useful Functions   ##
##                      ##

def weight_regularization_loss(listener):
    reg_loss = 0.0
    for p in listener.named_parameters():
        name = p[0]
        if 'weight' in name and ('stimulus_encoder' in name) or ('classifier_head' in name):
            reg_loss += p[1].norm(2)
    return reg_loss


def single_epoch_train(listener, train_loader, criterion, optimizer, device='cuda', **kwargs):
    reg_gamma = kwargs.get('reg_gamma', 0)    
    entropy_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    listener.train()
    for batch in train_loader:
        stimuli = batch['stimulus'].to(device)
        targets = batch['label'].to(device)
        tokens = batch['tokens'].to(device)
        b_size = len(targets)

        logits = listener(tokens, stimuli)
        preds = torch.argmax(logits, 1)

        xentropy_loss = criterion(logits, targets)
        reg_loss = 0
        if reg_gamma > 0:
            reg_loss = reg_gamma * weight_regularization_loss(listener)

        total_loss = xentropy_loss + reg_loss

        listener.zero_grad()
        total_loss.backward()
        optimizer.step()

        entropy_loss_meter.update(xentropy_loss.item(), b_size)
        guess_correct = torch.sum(preds == targets).double()
        accuracy_meter.update(guess_correct.double() / b_size, b_size)

    result = dict()
    result['accuracy'] = accuracy_meter.avg
    result['entropy_loss'] = entropy_loss_meter.avg
    return result


@torch.no_grad()
def evaluate_listener(listener, dataloader, device='cuda', return_logits=False):
    listener.eval()
    running_corrects = 0
    all_logits = []
    for batch in dataloader:
        stimuli = batch['stimulus'].to(device)
        targets = batch['label'].to(device)
        tokens = batch['tokens'].to(device)
        logits = listener(tokens, stimuli)
        if return_logits:
            all_logits.append(logits.cpu())
        preds = torch.argmax(logits, 1)
        running_corrects += torch.sum(preds == targets)
    n_examples = len(dataloader.dataset)
    accuracy = running_corrects.double() / n_examples
    accuracy = float(accuracy.cpu().squeeze().numpy())

    result = dict()
    result['accuracy'] = accuracy

    if return_logits:
        result['logits'] = torch.cat(all_logits).numpy()
    return result


def default_lstm_encoder(vocab, word_embedding_dim, lstm_n_hidden, init_c=None, init_h=None,
                         word_dropout=0.0, pooling='max'):

    word_embedding = nn.Embedding(len(vocab), word_embedding_dim, padding_idx=vocab.pad)
    wp = [nn.Linear(word_embedding_dim, word_embedding_dim), nn.ReLU()]

    if word_dropout > 0:
        wp.append(nn.Dropout(word_dropout))

    word_projection = nn.Sequential(*wp)

    model = LSTMEncoder(n_input=word_embedding_dim,
                        n_hidden=lstm_n_hidden,
                        word_embedding=word_embedding,
                        word_transformation=word_projection,
                        eos_symbol=vocab.eos,
                        feature_type=pooling,
                        init_h=init_h,
                        init_c=init_c)
    return model


##                                            ##
##  Ablation of latent-based listening Models ##
##                                            ##

def ablation_model_one(vocab, shape_latent_dim, context_aware=False):
    # transformer-based-encoder
    
    # vision: 
    visual_dropout = 0.3
    visual_n_hidden = 128
    visual_encoder = MLP(shape_latent_dim, [visual_n_hidden, visual_n_hidden], dropout_rate=visual_dropout)
    
    context_n_hidden = 0
    if context_aware:
        context_n_hidden = 64

    # language: 
    d_model= 128
    nhead = 2
    d_hid = 128
    nlayers = 2
    language_dropout = 0.3
    language_encoder = TransformerModel(len(vocab), d_model, nhead, d_hid, nlayers, language_dropout)
    
    # final head classifier
    clf_head_dropout = 0
    clf_mlp_neurons = [200, 100, 50, 1]
    classifier_head = MLP(visual_n_hidden + d_model + context_n_hidden, clf_mlp_neurons, dropout_rate=clf_head_dropout, remove_final_bias=True)    
    
    if context_aware:                        
        context_encoder = MLP(visual_n_hidden, [context_n_hidden, context_n_hidden], dropout_rate=visual_dropout)
        model = ContextAwareListener(language_encoder, visual_encoder, context_encoder, classifier_head, lstm_based=False)
    else:
        model = ContextFreeListener(language_encoder, visual_encoder, classifier_head, lstm_based=False)
    return model


def ablation_model_two(vocab, shape_latent_dim, context_aware=False):
    #lstm-based-encoder   
    word_embedding_dim = 128
    lstm_n_hidden = 128
    word_dropout = 0.1
    visual_n_hidden = 128
    visual_dropout = 0.2    
    clf_head_dropout = 0
    context_n_hidden = 0
    clf_mlp_neurons = [100, 50, 1]
    
    if context_aware:
        context_n_hidden = 64

    language_encoder = default_lstm_encoder(vocab,
                                            word_embedding_dim,
                                            lstm_n_hidden,
                                            word_dropout=word_dropout,
                                            init_c=nn.Linear(visual_n_hidden + context_n_hidden, lstm_n_hidden),
                                            init_h=nn.Linear(visual_n_hidden + context_n_hidden, lstm_n_hidden)
                                            )

    visual_encoder = MLP(shape_latent_dim, [visual_n_hidden, visual_n_hidden], dropout_rate=visual_dropout)
    classifier_head = MLP(lstm_n_hidden, clf_mlp_neurons, dropout_rate=clf_head_dropout, remove_final_bias=True)

    if context_aware:                
        context_encoder = MLP(visual_n_hidden, [context_n_hidden, context_n_hidden], dropout_rate=visual_dropout)
        model = ContextAwareListener(language_encoder, visual_encoder, context_encoder, classifier_head)
    else:        
        model = ContextFreeListener(language_encoder, visual_encoder, classifier_head)
    return model



##                                                    ##
##  Ablation of RAW pointcloud-based listening Models ##
##                                                    ##

class PointNetEncoder(nn.Module):
    def __init__(self, encoder_conv_layers, visual_n_hidden, visual_dropout):        
        super(PointNetEncoder, self).__init__()
        self.encoder_conv_layers = encoder_conv_layers        
        self.visual_n_hidden = visual_n_hidden
        self.visual_dropout = visual_dropout        
        self.pnet = PointNet(init_feat_dim=3, conv_dims=self.encoder_conv_layers)                         
        self.mlp = MLP(self.encoder_conv_layers[-1], [self.visual_n_hidden, self.visual_n_hidden], dropout_rate=self.visual_dropout)
    
    def __call__(self, pcs, channel_first=True):
        if channel_first:
            pcs = pcs.transpose(2, 1) # bring channel (xyz) dim before points
            
        pooled_feat = self.pnet(pcs)        
        return self.mlp(pooled_feat)


def ablation_raw_pointnet(vocab, context_aware=False):
    ### pointnet based pc-encoder
    ### transformer based language-encoder
    
    # vision: 
    encoder_conv_layers = [32, 32, 64, 128, 128, 128]
    visual_n_hidden = 128
    visual_dropout = 0.3
    visual_encoder = PointNetEncoder(encoder_conv_layers, visual_n_hidden, visual_dropout)
    
    context_n_hidden = 0
    if context_aware:
        context_n_hidden = 64

    # language: 
    d_model= 128
    nhead = 2
    d_hid = 128
    nlayers = 2
    language_dropout = 0.3
    language_encoder = TransformerModel(len(vocab), d_model, nhead, d_hid, nlayers, language_dropout)
    
    # final head classifier
    clf_head_dropout = 0
    clf_mlp_neurons = [200, 100, 50, 1]
    classifier_head = MLP(visual_n_hidden + d_model + context_n_hidden, clf_mlp_neurons, dropout_rate=clf_head_dropout, remove_final_bias=True)    
    
    if context_aware:                        
        context_encoder = MLP(visual_n_hidden, [context_n_hidden, context_n_hidden], dropout_rate=visual_dropout)
        model = ContextAwareListener(language_encoder, visual_encoder, context_encoder, classifier_head, lstm_based=False)
    else:
        model = ContextFreeListener(language_encoder, visual_encoder, classifier_head, lstm_based=False)
    return model


class DGCNNEncoder(nn.Module):
    def __init__(self, in_dim, dgcnn_out_dim, dgcnn_intermediate_feat_dim, k_neighbors, n_hidden, dropout):
        super(DGCNNEncoder, self).__init__()                    
        self.dgcnn = DGCNN(in_dim, intermediate_feat_dim=dgcnn_intermediate_feat_dim, out_dim=dgcnn_out_dim, k_neighbors=k_neighbors)
        self.mlp = MLP(dgcnn_out_dim, [n_hidden, n_hidden], dropout_rate=dropout)

    def forward(self, xyz):
        out = self.dgcnn(xyz)
        return self.mlp(out)
    
    
def ablation_raw_dgcnn(vocab, context_aware=False):        
    ### dgcnn++ based pc-encoder
    ### transformer based language-encoder
        
    # vision:     
    visual_n_hidden = 128    
    visual_dropout = 0.1     
    dgcnn_intermediate_feat_dim = [32, 32, 128]  # [64, 64, 128, 256] (higher performing but slower to train)
    dgcnn_out_dim = 256
    k_neighbors = 12
    visual_encoder = DGCNNEncoder(3, dgcnn_out_dim, dgcnn_intermediate_feat_dim, k_neighbors, visual_n_hidden, visual_dropout)
        
    context_n_hidden = 0
    if context_aware:
        context_n_hidden = 64

    # language: 
    d_model= 128
    nhead = 2
    d_hid = 128
    nlayers = 2
    language_dropout = 0.3
    language_encoder = TransformerModel(len(vocab), d_model, nhead, d_hid, nlayers, language_dropout)
    
    # final head classifier
    clf_head_dropout = 0
    clf_mlp_neurons = [200, 100, 50, 1]
    classifier_head = MLP(visual_n_hidden + d_model + context_n_hidden, clf_mlp_neurons, dropout_rate=clf_head_dropout, remove_final_bias=True)    
    
    if context_aware:                        
        context_encoder = MLP(visual_n_hidden, [context_n_hidden, context_n_hidden], dropout_rate=visual_dropout)
        model = ContextAwareListener(language_encoder, visual_encoder, context_encoder, classifier_head, lstm_based=False)
    else:
        model = ContextFreeListener(language_encoder, visual_encoder, classifier_head, lstm_based=False)
    return model