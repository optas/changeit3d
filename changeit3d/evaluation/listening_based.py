import torch
import torch.nn.functional as F
from changeit3d.utils.basics import iterate_in_chunks

@torch.no_grad()
def listening_fit_for_changeit3dnet(changeit3dnet, pretrained_listener, dataloader, stimulus_index=0, device='cuda'):
    changeit3dnet.eval()
    pretrained_listener.eval()

    all_probs = list()
    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        visual_stimulus = batch['stimulus'][:, stimulus_index].to(device)
        transformed = changeit3dnet.transform(tokens, visual_stimulus)        
        context = torch.stack([visual_stimulus, transformed], dim=1)
        logits = pretrained_listener(tokens, context)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)

    all_probs = torch.cat(all_probs).numpy()
    all_boosts = all_probs[:, 1] - all_probs[:, 0]
    avg_listening_boost = all_boosts.mean()
    avg_winners = (all_boosts > 0).mean()
    return all_probs, all_boosts, avg_winners, avg_listening_boost


@torch.no_grad()
def listening_fit_on_raw_pcs(source_pcs, transformed_pcs, tokens_encoded, pretrained_listener, batch_size=256, device='cuda'):
    
    pretrained_listener.eval()
    
    if not len(source_pcs) == len(transformed_pcs):
        raise ValueError()
    
    all_probs = list()        
    for batch_idx in iterate_in_chunks(range(len(source_pcs)), batch_size):
        tokens = torch.from_numpy(tokens_encoded[batch_idx]).to(device)        
        visual_stimulus = torch.from_numpy(source_pcs[batch_idx]).to(device)
        transformed = torch.from_numpy(transformed_pcs[batch_idx]).to(device)
        
        context = torch.stack([visual_stimulus, transformed], dim=1)
        logits = pretrained_listener(tokens, context)
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)

    all_probs = torch.cat(all_probs).numpy()
    all_boosts = all_probs[:, 1] - all_probs[:, 0]
    avg_listening_boost = all_boosts.mean()
    avg_winners = (all_boosts > 0).mean()
    return all_probs, all_boosts, avg_winners, avg_listening_boost
