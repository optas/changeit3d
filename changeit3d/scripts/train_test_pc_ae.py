"""
Script to train/test a pointcloud AutoEncoder.


Originally created at 4/23/21, for Python 3.x
2022 Panos Achlioptas (https://optas.github.io)
"""

import torch
import tqdm
import warnings
import numpy as np
import os.path as osp
from torch import optim

from changeit3d.in_out.arguments import parse_train_test_pc_ae_arguments
from changeit3d.in_out.pointcloud import (prepare_vanilla_pointcloud_datasets,
                                          prepare_pointcloud_dataloaders,
                                          deterministic_data_loader)
from changeit3d.in_out.basics import pickle_data, save_state_dicts, load_state_dicts
from changeit3d.models.model_descriptions import describe_pc_ae

# Argument-handling.
args = parse_train_test_pc_ae_arguments(save_args=True)

# Prepare pointcloud data.
datasets, _ = prepare_vanilla_pointcloud_datasets(args)
data_loaders = prepare_pointcloud_dataloaders(datasets, args)

# Make an AE.
device = torch.device("cuda:" + str(args.gpu_id))
model = describe_pc_ae(args).to(device)

if args.load_pretrained_model:
    best_epoch = load_state_dicts(args.pretrained_model_file, model=model)
    print('Loading pretrained model @epoch', best_epoch)
    print('Losses for this model/epoch:')
    for split in ['train', 'val', 'test']:
        loss = model.reconstruct(data_loaders[split], device=device)[-1]
        print(split, loss)

# Train it.
if args.do_training:
    model_name = 'best_model.pt'
    save_new_model_file = osp.join(args.log_dir, model_name)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                              patience=args.lr_patience,
                                                              verbose=True, min_lr=5e-7)

    start_epoch = 1
    min_val_loss = np.Inf
    val_not_improved = 0

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + args.max_train_epochs)):
        np.random.seed()
        train_loss = model.train_for_one_epoch(data_loaders['train'], optimizer, device=device)
        val_loss = model.reconstruct(data_loaders['val'], device=device)[-1]
        lr_scheduler.step(val_loss)

        test_recons, _, test_loss = model.reconstruct(data_loaders['test'], device=device)
        print('{}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, train_loss, test_loss, val_loss), end=' ')

        if val_loss < min_val_loss:
            print('* validation loss improved *')
            min_val_loss = val_loss
            save_state_dicts(save_new_model_file, epoch=epoch, model=model, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
            val_not_improved = 0
        else:
            val_not_improved += 1
            if val_not_improved == args.train_patience:
                print(f'Validation loss did not improve for {val_not_improved} consecutive epochs. Training is '
                      f'stopped.')
                break
            print()

    # Load model with best per-validation loss.
    best_epoch = load_state_dicts(osp.join(args.log_dir, model_name), model=model)
    print('per-validation optimal epoch', best_epoch)
    print('losses at this epoch:', best_epoch)
    for split in ['train', 'val', 'test']:
        reconstructions, losses_per_example, loss = model.reconstruct(data_loaders[split], device=device)
        print(split, loss)


# Extracting latent codes of the above trained system.
if args.extract_latent_codes:
    uid_to_latent = dict()
    n_data_points = 0
    for split in data_loaders:
        loader = data_loaders[split]
        n_data_points += len(loader.dataset)

        if split == 'train':
            loader = deterministic_data_loader(loader,
                                               **{'batch_size': args.batch_size,
                                                  'worker_init_fn': lambda x: np.random.seed(seed=int(args.random_seed))
                                                  })

        latents = model.embed_dataset(loader, device=device)
        data_uids = loader.dataset.model_metadata['model_uid']

        if len(data_uids) != len(latents):
            raise ValueError('The pointcloud dataset/loader has to have the model_uid attribute set. '
                             'See: ')

        for k, v in zip(data_uids, latents):
            uid_to_latent[k] = v

    if n_data_points != len(uid_to_latent):
        warnings.warn("The uids in the underlying pointcloud dataset are -not- unique. This can lead to unexpected "
                      "behavior.")

    save_out_latent_file = osp.join(args.log_dir, 'latent_codes.pkl')
    pickle_data(save_out_latent_file, uid_to_latent)
