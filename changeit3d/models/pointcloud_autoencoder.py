"""
PC-AE.
Originally created at 5/22/20, for Python 3.x
2022 Panos Achlioptas (optas.github.io)
"""

import torch
from torch import nn
from ..losses.chamfer import chamfer_loss
from ..utils.stats import AverageMeter


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, pointclouds, bcn_format=True):
        """
        :param pointclouds: B x N x 3
        :param bcn_format: the AE.encoder works with Batch x Color (xyz) x Number of points format
        """

        b_size, n_points, _ = pointclouds.shape

        if bcn_format:
            pointclouds = pointclouds.transpose(2, 1).contiguous()

        z = self.encoder(pointclouds)
        recon = self.decoder(z).view([b_size, n_points, 3])
        return recon

    @torch.no_grad()
    def embed(self, pointclouds, bcn_format=True):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :param bcn_format: the AE.encoder works with Batch x Color (xyz) x Number of points format
        :return: B x latent-dimension of AE
        """
        if bcn_format:
            pointclouds = pointclouds.transpose(2, 1).contiguous()
        return self.encoder(pointclouds)

    @torch.no_grad()
    def embed_dataset(self, loader, device='cuda'):
        latents = []
        self.eval()
        for batch in loader:
            b_pc = batch['pointcloud'].to(device)
            latent_b = self.embed(b_pc)
            latents.append(latent_b.cpu())
        latents = torch.cat(latents).numpy()
        return latents

    def train_for_one_epoch(self, loader, optimizer, device='cuda', loss_rule="chamfer"):
        """ Train the auto encoder for one epoch based on the Chamfer (or EMD) loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()
        for batch in loader:
            b_pc = batch['pointcloud'].to(device)
            recon = self(b_pc)

            # Backward to optimize according to Chamfer loss.
            optimizer.zero_grad()
            if loss_rule == "chamfer":
                loss = chamfer_loss(b_pc, recon).mean()
            elif loss_rule == "emd":
                raise NotImplementedError(" First install earth's mover distance loss")
                # loss = emd_loss(b_pc, recon, transpose=False).mean()
            else:
                raise NotImplementedError()

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(b_pc))
        return loss_meter.avg

    @torch.no_grad()
    def reconstruct(self, loader, device='cuda', loss_rule="chamfer"):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        """

        reconstructions = []
        losses_per_example = []
        loss_meter = AverageMeter()

        self.eval()
        for batch in loader:
            b_pc = batch['pointcloud'].to(device)
            recon = self(b_pc)
            if loss_rule == "chamfer":
                loss = chamfer_loss(b_pc, recon)
            elif loss_rule == "emd":
                raise NotImplementedError(" First install earth's mover distance loss")                
                # loss = emd_loss(b_pc, recon, transpose=False)
            else:
                raise NotImplementedError()
            losses_per_example.extend(loss.cpu())
            reconstructions.append(recon.cpu())
            loss_meter.update(loss.mean().item(), len(b_pc))

        reconstructions = torch.cat(reconstructions).numpy()
        losses_per_example = torch.stack(losses_per_example).numpy()
        return reconstructions, losses_per_example, loss_meter.avg