import torch
from torch import nn

from ..utils.stats import AverageMeter
from ..losses.chamfer import chamfer_loss


class MonolithicChangeIt3D(nn.Module):
    ## altentive baseline to our final system
    def __init__(self,
                 visual_encoder,
                 language_encoder,                 
                 fuser,
                 decoder,
                 n_pc_points=None):
        
        
        super().__init__()
        self.visual_encoder = visual_encoder
        self.language_encoder = language_encoder
        self.fuser = fuser
        self.decoder = decoder            
        self.n_pc_points = n_pc_points
        
    def __call__(self, tokens, visual_stimulus, bcn_format=True):
        
        if bcn_format:
            visual_stimulus = visual_stimulus.transpose(2, 1).contiguous()
            
        lang_emb = self.language_encoder(tokens)
        vis_emb = self.visual_encoder(visual_stimulus)
                    
        joint_emb = torch.concat([vis_emb, lang_emb], axis=1)        
        fused_emb = self.fuser(joint_emb)        
        output = self.decoder(fused_emb)        
        
        if self.n_pc_points is not None:
            output = output.view([len(tokens), self.n_pc_points, 3])
            
        return output

    def single_epoch_train(self, trainloader, optimizer, device="cuda"):        
        loss_meter = AverageMeter()
        self.train()                                
        for batch in trainloader:
            tokens = batch['tokens'].to(device)
            distractor = batch['stimulus'][:, 0].to(device)
            target = batch['stimulus'][:, 1].to(device)                        
            assert (batch['label'] == 1).all()
            
            optimizer.zero_grad()            
            prediction = self(tokens, distractor)
                        
            # compute losses            
            loss = chamfer_loss(target, prediction).mean()
            loss.backward()
            optimizer.step()

            batch_size = len(tokens)
            loss_meter.update(loss.item(), batch_size)
            
        result = {'total_loss': loss_meter.avg}        
        return result
        
    @torch.no_grad()
    def evaluate(self, dataloader, device="cuda"):        
        loss_meter = AverageMeter()
        self.eval()    
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            distractor = batch['stimulus'][:, 0].to(device)
            target = batch['stimulus'][:, 1].to(device)                        
            assert (batch['label'] == 1).all()
                        
            prediction = self(tokens, distractor)
            loss = chamfer_loss(target, prediction).mean()
            batch_size = len(tokens)
            loss_meter.update(loss.item(), batch_size)
            
        result = {'total_loss': loss_meter.avg}
        return result
        
        
        

