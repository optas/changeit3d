"""
Script to train the Monolithic baseline for Language-Assisted Deformations.
By Panos Achlioptas (https://optas.github.io)
"""

import torch
import numpy as np
import os.path as osp
import pandas as pd
from torch import optim
from functools import partial
from ast import literal_eval

from changeit3d.in_out.basics import (create_logger,                                      
                                      torch_save_model,
                                      save_state_dicts,
                                      load_state_dicts,
                                      pickle_data)

from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.in_out.arguments import parse_train_test_monolithic_changeit3d
from changeit3d.in_out.pointcloud import pc_loader_from_npz, uniform_subsample, center_in_unit_sphere
from changeit3d.language.vocabulary import Vocabulary
from changeit3d.models.model_descriptions import monolithic_alternative_to_changeit3d
from changeit3d.evaluation.all_metrics import run_all_metrics

    
# Argument-handling.
args = parse_train_test_monolithic_changeit3d()
logger = create_logger(args.log_dir)


##
# Prepare the input data
##
df = pd.read_csv(args.shape_talk_file)
df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
vocab = Vocabulary.load(args.vocab_file)


# constrain training in language of particular classes
if len(args.restrict_shape_class) > 0:
    mask = df.target_object_class.isin(set(args.restrict_shape_class))
    df = df[mask].copy()
    df.reset_index(inplace=True, drop=True)
    logger.info('Restricting to class(es) {}. Total utterances: {}'.format(args.restrict_shape_class, len(df)))


## ensure all relevant shapes are reachable
all_model_uids = set(df.source_uid.unique()).union(set(df.target_uid.unique()))
print('Annotated shapes:', len(all_model_uids))
assert all([osp.exists(osp.join(args.top_pc_dir, x + '.npz')) for x in all_model_uids]), "all pc-files need to exist"


##
# Prepare the data loaders
##

# make df compatible with LanguageContrastive Dataset
df = df.assign(target=df.target_uid)
df = df.assign(distractor_1=df.source_uid)

def to_stimulus_func(x, top_pc_dir, n_samples=2048, random_seed=None, scale_pc=False):
    pc = pc_loader_from_npz(osp.join(top_pc_dir, x + '.npz'))
    pc, _ = uniform_subsample(pc, n_samples, random_seed)
    if scale_pc:
        pc = center_in_unit_sphere(pc)
    return pc


dataloaders = dict()
for split in ['train', 'val', 'test']:
    ndf = df[df.changeit_split == split].copy()
    ndf.reset_index(inplace=True, drop=True)
    seed = None if split == 'train' else args.random_seed
    batch_size = args.batch_size if split == 'train' else 2 * args.batch_size
        
    loader = partial(to_stimulus_func, 
                     top_pc_dir=args.top_pc_dir, n_samples=args.n_pc_points, 
                     random_seed=seed, scale_pc=args.scale_in_u_sphere)
        
    dataset = LanguageContrastiveDataset(ndf,
                                         loader,
                                         n_distractors=1,
                                         shuffle_items=False) # important, target *always* last
    
    dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=split == 'train',
                                                     num_workers=args.num_workers,
                                                     worker_init_fn=lambda x: np.random.seed(seed))


device = torch.device("cuda:" + str(args.gpu_id))
model = monolithic_alternative_to_changeit3d(args.n_pc_points, vocab)
model = model.to(device)


##
# Optimization 
##
optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                          factor=0.5, patience=args.lr_patience,
                                                          verbose=True, min_lr=5e-7)


## Optionally, load pretrained model.
if args.pretrained_model is not None:
    best_epoch = load_state_dicts(args.pretrained_model, model=model, 
                                  optimizer=optimizer, lr_scheduler=lr_scheduler, map_location="cpu")
    logger.info(f'Loading pretrained model @epoch {best_epoch}')
    model = model.to(device)    
    start_epoch = best_epoch
else:
    start_epoch = 0


##
# Train it.
##

if args.train:
    epochs_val_not_improved = 0
    best_val_loss = np.Inf    
    checkpoint_file = osp.join(args.log_dir, 'best_model.pt')

    logger.info('Start training of the Monolithic alternative to ChangeIt3DNet.')
    for epoch in range(start_epoch + 1, start_epoch + args.max_train_epochs + 1):
        np.random.seed()

        # training
        train_losses = \
            model.single_epoch_train(dataloaders['train'],                                     
                                     optimizer,
                                     device=device)
            
        logger.info(f"@epoch-{epoch}")
        logger.info("train/val losses:")
        train_losses_str = " ".join(["{:15} {:.5f}".format(key, val) for key, val in train_losses.items()])
        logger.info(train_losses_str)

        # validation
        val_losses = \
            model.evaluate(dataloaders['val'],                                                      
                           device=device)
        val_losses_str = " ".join(["{:15} {:.5f}".format(key, val) for key, val in val_losses.items()])
        logger.info(val_losses_str)

        lr_scheduler.step(val_losses['total_loss'])
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            epochs_val_not_improved = 0
            save_state_dicts(checkpoint_file, epoch=epoch, model=model,
                             optimizer=optimizer, lr_scheduler=lr_scheduler)
        else:
            epochs_val_not_improved += 1
            logger.info("* validation loss did not improved *")

        if epochs_val_not_improved == args.train_patience:
            logger.warning(
                f'Validation loss did not improve for {epochs_val_not_improved} consecutive epochs. Training is stopped.')
            break

    logger.info('Training is done!')

    # Load best model per validation set
    best_epoch = load_state_dicts(checkpoint_file, model=model)
    logger.info(f'per-validation optimal epoch {best_epoch}')

    # save one more time the model, this time as a module directly working for inference
    checkpoint_pkl_file = osp.join(args.log_dir, 'best_model.pkl')
    torch_save_model(model, checkpoint_pkl_file)
    
    
if args.test:
    
    def prepare_dataset_for_testing(check_loader):
        ### Prepare a dataset that is fair w.r.t other ablated systems: i.e., they use *single* utterances to change the input; 
        ### not multi- (merged) ones;
         
        ndf = check_loader.dataset.df.copy()
        
        # Now *break down the **merged** sentences
        # (use the special token that was used to merge them (which in our case that is vocab.dia) )
        disjoint_utters = ndf.utterance_spelled.apply(lambda x: x.split(vocab.idx2word[vocab.dia])) 
        
        def pick_one_at_random(utters):
            np.random.seed(args.random_seed) 
            return np.random.choice(utters)

        single_utter = disjoint_utters.apply(pick_one_at_random)  # pick one of the many at random        
        single_utter = single_utter.apply(lambda x: x.strip())
        tokens = single_utter.apply(lambda x: x.split())
        max_len = tokens.apply(len).max()
        ndf['tokens_encoded'] = tokens.apply(lambda x: vocab.encode(x, max_len=max_len))
        ndf['utterance_spelled'] = single_utter
        ndf['tokens'] = tokens
        ndf = ndf.drop(columns={'utterance', 'tokens_len'})
        
        dataset = LanguageContrastiveDataset(ndf,
                                            loader,
                                            n_distractors=1,
                                            shuffle_items=False) # important, target *always* last

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                worker_init_fn=lambda x: np.random.seed(args.random_seed))
        return test_loader
        

    all_reconstructions = []
    all_input_pcs = []
    all_tokens_used = []            
    # check_loader = dataloaders['test']
    check_loader = prepare_dataset_for_testing(dataloaders['test'])
    
    model.eval()
    with torch.no_grad():        
        for batch in check_loader:
            tokens = batch['tokens']
            source = batch['stimulus'][:, 0]
            output = model(tokens.to(device), source.to(device))
            all_reconstructions.append(output.cpu())
            all_input_pcs.append(source.cpu())
            all_tokens_used.append(output.cpu())            
        all_reconstructions = torch.cat(all_reconstructions).numpy()
        all_input_pcs = torch.cat(all_input_pcs).numpy()
        all_tokens_used = torch.cat(all_tokens_used).numpy()
        
    if all_reconstructions.shape[-2] != args.n_sample_points:
        all_reconstructions = np.array([uniform_subsample(s, args.n_sample_points, args.random_seed)[0] for s in all_reconstructions])
    
    if all_input_pcs.shape[-2] != args.n_sample_points:
        all_input_pcs = np.array([uniform_subsample(s, args.n_sample_points, args.random_seed)[0] for s in all_input_pcs])
        
    sentences = check_loader.dataset.df.utterance_spelled.values    
    gt_classes = check_loader.dataset.df.source_object_class

    results_on_metrics = run_all_metrics(all_reconstructions, all_input_pcs, gt_classes, sentences, vocab, args, logger)    
    pickle_data(osp.join(args.log_dir, 'evaluation_metric_results.pkl'), results_on_metrics)
    
        
    
    
        