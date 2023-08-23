import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from ..utils.stats import AverageMeter


class PointcloudClassifier(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        # self.idx_to_class_name = idx_to_class_name

    def __call__(self, batch):
        res = dict()
        latents = self.encoder(batch['pointcloud'])
        res['class_logits'] = self.decoder(latents)
        return res


    def single_epoch_train(self, data_loader, criterion, optimizer, device, ignore_label=-1, channel_last=False):
        batch_keys = ['pointcloud', 'model_class']
        total_loss = AverageMeter()
        total_acc = AverageMeter()

        # Set the model in training mode
        self.train()
        np.random.seed()

        for batch in data_loader:
            if channel_last:
                batch['pointcloud'] = batch['pointcloud'].permute(0, 2, 1)

            for k in batch_keys:  # Move the batch to gpu
                if len(batch[k]):
                    batch[k] = batch[k].to(device)

            # Forward pass
            logits = self(batch)['class_logits']

            # Loss
            optimizer.zero_grad()
            loss = criterion(logits, target=batch['model_class'])
            loss.backward()
            optimizer.step()

            b_acc, effective_batch_size = self.prediction_stats(logits, batch['model_class'], ignore_label)

            total_acc.update(b_acc, effective_batch_size)
            total_loss.update(loss.item(), effective_batch_size)

        return total_loss.avg, total_acc.avg


    @torch.no_grad()
    def evaluate_on_dataset(self, data_loader, criterion, device, ignore_label=-1, channel_last=False):
        batch_keys = ['pointcloud', 'model_class']
        total_loss = AverageMeter()
        total_acc = AverageMeter()

        # Set the model in eval mode
        self.eval()

        for batch in data_loader:
            if channel_last:
                batch['pointcloud'] = batch['pointcloud'].permute(0, 2, 1)

            # Move the batch to gpu
            for k in batch_keys:
                if len(batch[k]) > 0:
                    batch[k] = batch[k].to(device)

            # Forward Pass
            logits = self(batch)['class_logits']
            b_acc, bsize = self.prediction_stats(logits, batch['model_class'], ignore_label=ignore_label)

            if criterion is not None:
                loss = criterion(logits, target=batch['model_class'])
            else:
                loss = torch.Tensor([-bsize])

            # Update the total loss, accuracy
            total_acc.update(b_acc, bsize)
            total_loss.update(loss.item(), bsize)

        return total_loss.avg, total_acc.avg

    @torch.no_grad()
    def get_predictions(self, data_loader, device, channel_last=False, evaluation=True):

        if evaluation:
            self.eval()

        all_logits = []
        for batch in data_loader:
            if channel_last:
                batch['pointcloud'] = batch['pointcloud'].permute(0, 2, 1)

            batch['pointcloud'] = batch['pointcloud'].to(device)
            logits = self(batch)['class_logits']
            all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits)
        all_probs = F.softmax(all_logits, dim=1)
        return all_logits.numpy(), all_probs.numpy()

    @staticmethod
    def prediction_stats(logits, gt_labels, ignore_label):
        # takes care of the ignore_label
        with torch.no_grad():
            valid_indices = gt_labels != ignore_label
            effective_batch_size = valid_indices.sum().item()

            predictions = logits.argmax(dim=-1)
            predictions = predictions[valid_indices]
            gt_labels = gt_labels[valid_indices]
            correct_guessed_mask = gt_labels == predictions
            b_acc = correct_guessed_mask.double().mean().item()
        return b_acc, effective_batch_size
