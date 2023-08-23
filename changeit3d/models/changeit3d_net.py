import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch import nn

from ..losses.lp_norms import safe_l2
from ..utils.stats import AverageMeter


def safe_log(x, eps=1e-12):
    eps = torch.ones_like(x) * eps
    x_with_eps = torch.max(torch.cat([x, eps], 1), -1).values
    return torch.log(x_with_eps)


class LatentDirectionFinder(nn.Module):
    def __init__(self,
                 language_encoder,
                 visual_encoder,
                 editor_unit,
                 context_free=True,
                 magnitude_unit=None,
                 unit_normalize_direction=True,
                 self_contrast=True
                 ):
        super().__init__()

        self.language_encoder = language_encoder
        self.visual_encoder = visual_encoder
        self.editor_unit = editor_unit
        self.context_free = context_free
        self.unit_normalize_direction = unit_normalize_direction
        self.magnitude_unit = magnitude_unit
        self.self_contrast = self_contrast
        
        if not context_free:
            raise NotImplementedError('Not in this version.')

    def context_free_edit(self, tokens, visual_stimulus):
        z_lang = self.language_encoder(tokens)
        
        if self.visual_encoder is not None:
            z_vision = self.visual_encoder(visual_stimulus)
            edit_in = torch.cat([z_lang, z_vision], axis=1)
        else:
            edit_in = z_lang

        edit_latent = self.editor_unit(edit_in)
        
        if self.unit_normalize_direction:
            edit_latent = F.normalize(edit_latent, dim=1)

        if self.magnitude_unit is not None:
            magnitude = self.magnitude_unit(edit_in)
            edit_latent *= magnitude
        else:
            magnitude = torch.zeros(len(tokens))

        return edit_latent, magnitude

    def transform(self, tokens, visual_stimulus):
        edit_latent, _ = self.context_free_edit(tokens, visual_stimulus)
        return visual_stimulus + edit_latent

    def __call__(self, tokens, visual_stimulus):
        if self.context_free:
            return self.context_free_edit(tokens, visual_stimulus)
                            
    def single_epoch_train(self, pretrained_listener, dataloader, criterion, optimizer, gamma=0, adaptive_id_penalty="", device="cuda"):  
        self.train()
        np.random.seed()
        pretrained_listener.eval()  # we do not train this net.

        listening_meter = AverageMeter()
        identity_meter = AverageMeter()
        total_loss_meter = AverageMeter()
                                        
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            distractor = batch['stimulus'][:, 0].to(device)                        
            target_labels = batch['label'].to(device)            
            assert (target_labels == 1).all()
            
            optimizer.zero_grad()
            
            edit_latent, _ = self(tokens, distractor)
            transformed_distractor = distractor + edit_latent

            if self.self_contrast:
                context = torch.stack([distractor, transformed_distractor], dim=1)
            else:
                # compare with the actual ground-truth item for which the language is compatible/referential
                target = batch['stimulus'][:, 1].to(device)
                context = torch.stack([target, transformed_distractor], dim=1)

            logits = pretrained_listener(tokens, context)
                        
            # compute losses            
            listening_loss = criterion(logits, target_labels)
            total_loss = listening_loss
                                               
            if gamma > 0:
                if adaptive_id_penalty != "":
                    if adaptive_id_penalty == "distractor_listening_prob":                                 
                        with torch.no_grad():                            
                            distractor_probs = torch.softmax(logits[:, 0])                            
                        identity_loss = gamma * distractor_probs * safe_l2(edit_latent, dim=[1], reduce=None)
                        identity_loss = torch.mean(identity_loss)
                else:
                    identity_loss = gamma * safe_l2(edit_latent, dim=[1], reduce="mean")
                
                total_loss += identity_loss
                        
            total_loss.backward()
            optimizer.step()

            batch_size = len(tokens)
            total_loss_meter.update(total_loss.item(), batch_size)
            listening_meter.update(listening_loss.item(), batch_size)
            
            if gamma > 0:
                identity_meter.update(identity_loss.item(), batch_size)

        result = dict()
        result['total_loss'] = total_loss_meter.avg
        result['listening_loss'] = listening_meter.avg
        result['identity_loss'] = identity_meter.avg
        return result

    @torch.no_grad()
    def evaluate(self, pretrained_listener, dataloader, criterion, gamma=0, adaptive_id_penalty="", device="cuda"):

        self.eval()
        pretrained_listener.eval()

        listening_meter = AverageMeter()
        identity_meter = AverageMeter()
        total_loss_meter = AverageMeter()

        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            distractor = batch['stimulus'][:, 0].to(device)
            target_labels = batch['label'].to(device)
            batch_size = len(tokens)
            assert (target_labels == 1).all()

            edit_latent, _ = self(tokens, distractor)
            transformed_distractor = distractor + edit_latent

            if self.self_contrast:
                context = torch.stack([distractor, transformed_distractor], dim=1)
            else:
                target = batch['stimulus'][:, 1].to(device)
                context = torch.stack([target, transformed_distractor], dim=1)

            logits = pretrained_listener(tokens, context)
            listening_loss = criterion(logits, target_labels).item()
            listening_meter.update(listening_loss, batch_size)
                        
            if gamma > 0:                
                if adaptive_id_penalty != "":
                    if adaptive_id_penalty == "distractor_listening_prob":                        
                        distractor_probs = torch.softmax(logits[:, 0])                            
                        identity_loss = gamma * distractor_probs * safe_l2(edit_latent, dim=[1], reduce=None)
                        identity_loss = torch.mean(identity_loss)
                else:
                    identity_loss = gamma * safe_l2(edit_latent, dim=[1], reduce="mean")                
            else:
                identity_loss = 0

            identity_meter.update(identity_loss, batch_size)            
            total_loss = listening_loss + identity_loss
            total_loss_meter.update(total_loss, batch_size)

        result = dict()
        result['total_loss'] = total_loss_meter.avg
        result['listening_loss'] = listening_meter.avg
        result['identity_loss'] = identity_meter.avg

        return result


@torch.no_grad()
def get_transformed_latent_code(direction_finder, data_loader, stimulus_index, scales=[1], device="cuda"):
    """  Extract transformation for a given latent code based on LatentDirectionFinder    
    """
    direction_finder.eval()
    
    all_z_codes = defaultdict(list)
    all_tokens = []
    all_edit_latents = []
    all_magnitudes = []

    for batch in data_loader:
        t = batch['tokens'].to(device)
        s = batch['stimulus'][:, stimulus_index].to(device)
        edit_latent, guessed_mag = direction_finder(t, s)
        
        for scale in scales:
            if scale == 0:  # no transformation, just return the input/starting latent code
                transformed = s
            else:
                transformed = s + scale * edit_latent

            all_z_codes[scale].append(transformed.cpu())
        all_edit_latents.append(edit_latent.cpu())
        all_magnitudes.append(guessed_mag.cpu())

        all_tokens.extend(t.tolist())

    all_edit_latents = torch.cat(all_edit_latents).numpy()
    all_magnitudes = torch.cat(all_magnitudes).numpy()

    for scale in scales:
        all_z_codes[scale] = torch.cat(all_z_codes[scale]).numpy()

    results = {'z_codes': all_z_codes,
               'tokens': all_tokens,
               'edit_latents': all_edit_latents,
               'magnitudes': all_magnitudes}

    return results