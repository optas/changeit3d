"""
Script to train a pointCloud-based shape classifier.

Originally created at 4/23/21, for Python 3.x
Panos Achlioptas (https://optas.github.io)
"""

import tqdm
import torch
import warnings
import pandas as pd
import os.path as osp
from torch import nn
from torch import optim

from changeit3d.in_out.basics import pickle_data, torch_save_model, save_state_dicts, load_state_dicts
from changeit3d.in_out.pointcloud import prepare_vanilla_pointcloud_datasets, prepare_pointcloud_dataloaders
from changeit3d.in_out.arguments import parse_train_test_pc_clf_arguments
from changeit3d.models.model_descriptions import describe_pc_clf

# Argument-handling.
args = parse_train_test_pc_clf_arguments(save_args=True)

##
# Prepare pointcloud data.
##
datasets, class_name_to_idx = prepare_vanilla_pointcloud_datasets(args)
dataloaders = prepare_pointcloud_dataloaders(datasets, args)

if args.do_training:
    out_f = osp.join(args.log_dir, 'class_name_to_idx.pkl')
    if osp.exists(out_f):
        warnings.warn('Existing class_name_to_idx.pkl will be overwritten')
    pickle_data(out_f, class_name_to_idx)


# Make a PC-CLF.
device = torch.device("cuda:" + str(args.gpu_id))
model = describe_pc_clf(args).to(device)

# Optionally, load pretrained model
if args.pretrained_model_file is not None:
    epoch_loaded = load_state_dicts(args.pretrained_model_file, model=model)
    print('Pretrained model is loaded at epoch', epoch_loaded)

    test_acc = model.evaluate_on_dataset(dataloaders['test'], criterion=None,
                                         device=device, channel_last=True)[-1]
    print('Its test accuracy is:', test_acc)
    
    

##
# Train the model.
##
if args.do_training:
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model_out_file = osp.join(args.log_dir, 'best_model.pt')

    start_epoch = 1
    max_val_acc = 0
    val_not_improved = 0

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + args.max_train_epochs)):

        train_loss, train_acc = model.single_epoch_train(dataloaders['train'], criterion, optimizer,
                                                         device=device, channel_last=True)

        val_loss, val_acc = model.evaluate_on_dataset(dataloaders['val'], criterion, device, channel_last=True)
        test_loss, test_acc = model.evaluate_on_dataset(dataloaders['test'], criterion, device, channel_last=True)

        print('{}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, train_loss, val_loss, test_loss))
        print('{}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, train_acc, val_acc, test_acc), end=' ')

        if val_acc > max_val_acc:
            print('* validation accuracy improved *')
            max_val_acc = val_acc
            save_state_dicts(model_out_file, epoch=epoch, model=model)
            val_not_improved = 0
        else:
            val_not_improved += 1
            if val_not_improved == args.train_patience:
                print(
                    f'Validation loss did not improve for {val_not_improved} consecutive epochs. Training is stopped.')
                break
            print()

    print('Training is Complete.')

    # Re-load the model with best per-validation loss.
    best_epoch = load_state_dicts(osp.join(model_out_file), model=model)
    print('Reload best per-validation model at epoch', best_epoch)

    print(f'accuracies at this ({best_epoch}) epoch:')
    for split in ['train', 'val', 'test']:
        accuracy = model.evaluate_on_dataset(dataloaders[split], criterion=None,
                                             device=device, ignore_label=-1,
                                             channel_last=True)[1]
        print(split, accuracy)

    # pickle that model for easy access
    torch_save_model(model, osp.join(args.log_dir, 'best_model.pkl'))

##
# Show some statistics
##
print('Per class test predictions.')
dloader = dataloaders['test']
dset = dloader.dataset

L, P = model.get_predictions(dloader, channel_last=True, device=device)
df = pd.DataFrame([(P.argmax(1) == dset.model_classes), dset.model_classes]).T
df.columns = ['guess_correct', 'shape_class_int']
idx_to_class_name = {v: k for k, v in class_name_to_idx.items()}
df = df.assign(class_name=df.shape_class_int.apply(lambda x: idx_to_class_name[x]))
print(df.groupby('class_name')['guess_correct'].mean().sort_values(ascending=False))