"""
Script to train a Neural Listener based on given raw pointcloud representations of 3D shapes.
2022 Panos Achlioptas (https://optas.github.io)
"""

import torch
import numpy as np
import pandas as pd
import os.path as osp
from functools import partial
from torch import nn
from torch import optim
from ast import literal_eval

from changeit3d.in_out.basics import (create_logger,
                                      pickle_data,
                                      torch_save_model,
                                      save_state_dicts,
                                      load_state_dicts)

from changeit3d.in_out.arguments import parse_train_test_raw_listener_arguments
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.language.vocabulary import Vocabulary
from changeit3d.models.listening_oriented import ablation_raw_pointnet, ablation_raw_dgcnn
from changeit3d.models.listening_oriented import single_epoch_train, evaluate_listener
from changeit3d.in_out.pointcloud import pc_loader_from_npz, uniform_subsample

# Argument-handling.
args = parse_train_test_raw_listener_arguments()
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
assert all([osp.exists(osp.join(args.top_raw_pc_dir, x + '.npz')) for x in all_model_uids]), "all pc-files need to exist"


##
# Prepare the data loaders
##

# make df compatible with LanguageContrastive Dataset
df = df.assign(target=df.target_uid)
df = df.assign(distractor_1=df.source_uid)

def to_stimulus_func(x, top_raw_pc_dir, n_samples=4096, random_seed=None):
    pc = pc_loader_from_npz(osp.join(top_raw_pc_dir, x + '.npz'))
    pc, _ = uniform_subsample(pc, n_samples, random_seed)
    return pc

dataloaders = dict()
for split in ['train', 'val', 'test']:
    ndf = df[df.listening_split == split].copy()
    ndf.reset_index(inplace=True, drop=True)
    seed = None if split == 'train' else args.random_seed
    batch_size = args.batch_size if split == 'train' else 2 * args.batch_size
    shuffle_items = split == 'train'
    
    loader = partial(to_stimulus_func, top_raw_pc_dir=args.top_raw_pc_dir, 
                     n_samples=args.n_pc_points, random_seed=seed)
        
    dataset = LanguageContrastiveDataset(ndf,
                                         loader,
                                         n_distractors=1,
                                         shuffle_items=shuffle_items)
    
    dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=shuffle_items,
                                                     num_workers=args.num_workers,
                                                     worker_init_fn=lambda x: np.random.seed(seed))

##
# Build Listening Model
##

if args.listening_model == 'pointnet-ablation1':
    model = ablation_raw_pointnet(vocab)
elif args.listening_model == "dgcnn-ablation1":
    model = ablation_raw_dgcnn(vocab)    
else:
    raise NotImplementedError()

print('Listening Architecture:')
print(model)

device = torch.device("cuda:" + str(args.gpu_id))
model = model.to(device)

##
# Optimization
##
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=args.lr_patience,
                                                          verbose=True, min_lr=5e-7)


if args.pretrained_model_file is not None:
    best_epoch = load_state_dicts(args.pretrained_model_file, model=model, 
                                  optimizer=optimizer, lr_scheduler=lr_scheduler, map_location="cpu")
    logger.info(f'Loading pretrained listener @epoch {best_epoch}')
    model = model.to(device)
    test_acc = evaluate_listener(model, dataloaders['test'], device=device, return_logits=True)['accuracy']
    logger.info(f'Test accuracy at that loaded epoch is : {test_acc}')
    start_epoch = best_epoch  
else:
    start_epoch = 0
    
##
# Training.
##
if args.do_training:
    epochs_val_not_improved = best_test_accuracy = best_val_accuracy = 0
    
    checkpoint_file = osp.join(args.log_dir, 'best_model.pt')
    logger.info('Start training of the listener.')

    for epoch in range(start_epoch + 1, start_epoch + args.max_train_epochs + 1):
        np.random.seed()
        train_acc = single_epoch_train(model,
                                       dataloaders['train'],
                                       criterion, optimizer, device=device)['accuracy']

        logger.info(f"@epoch-{epoch} train {train_acc:.3f}")

        for split in ['val', 'test']:
            epoch_accuracy = evaluate_listener(model, dataloaders[split], device=device)['accuracy']

            if split == 'val':
                lr_scheduler.step(epoch_accuracy)

                if epoch_accuracy > best_val_accuracy:
                    epochs_val_not_improved = 0
                    best_val_accuracy = epoch_accuracy
                    save_state_dicts(checkpoint_file, epoch=epoch, model=model,
                                     optimizer=optimizer, lr_scheduler=lr_scheduler)
                else:
                    epochs_val_not_improved += 1

            logger.info("{} {:.3f}".format(split, epoch_accuracy))

            if split == 'test' and epochs_val_not_improved == 0:
                best_test_accuracy = epoch_accuracy

        if epochs_val_not_improved == 0:
            logger.info("* validation accuracy improved *")

        logger.info("\nbest test accuracy {:.3f}".format(best_test_accuracy))

        if epochs_val_not_improved == args.train_patience:
            logger.warning(
                f'Validation loss did not improve for {epochs_val_not_improved} consecutive epochs. Training is stopped.')
            break

    # Load newly trained model with best per-validation loss.
    logger.info('Training is done!')
    best_epoch = load_state_dicts(checkpoint_file, model=model)
    logger.info(f'per-validation optimal epoch {best_epoch}')
    test_acc = evaluate_listener(model, dataloaders['test'], device=device, return_logits=True)['accuracy']
    logger.info(f'(verifying) test accuracy at that epoch is : {test_acc}')

    # save one more time the model, this time as a module directly working for inference
    checkpoint_pkl_file = osp.join(args.log_dir, 'best_model.pkl')
    torch_save_model(model, checkpoint_pkl_file)

##
# Testing
##

logger.info('Running detailed inference')

train_df = dataloaders['train'].dataset.df
training_examples = set(train_df.target.unique())
training_examples = training_examples.union(set(train_df.distractor_1.unique()))

evaluation_results = dict()  # store those in the end.

for split in ['test', 'val']:
    evaluation_results[split] = dict()
    logger.info(f'Split: {split}')

    res = evaluate_listener(model, dataloaders[split], device=device, return_logits=True)
    evaluation_results[split]['accuracy'] = res['accuracy']
    logger.info(f"accuracy {res['accuracy']}")

    probabilities = torch.softmax(torch.Tensor(res['logits']), dim=1)
    guess_correct = torch.argmax(probabilities, 1) == 1
    assert abs(guess_correct.double().mean() - res['accuracy']) < 10e-5

    augmented_df = dataloaders[split].dataset.df.copy(deep=True)
    augmented_df = augmented_df.assign(guessed_correct=guess_correct.tolist())
    augmented_df = augmented_df.assign(guessed_probs=probabilities.tolist())
    evaluation_results[split]['augmented_df_with_predictions'] = augmented_df.copy(deep=True)

    # ShapeTalk vs. ShapeGlot analysis
    sg_mask = augmented_df.assignmentid == 'shapeglot'
    if sg_mask.sum() > 0:
        acc_on_not_sg = augmented_df[~sg_mask]['guessed_correct'].mean()
        evaluation_results[split]['accuracy_excluding_SG_examples'] = acc_on_not_sg
        logger.info(f'Accuracy on examples not from ShapeGlot {acc_on_not_sg}\n')

    # Saliency analysis
    logger.info('Accuracies per sentence saliency [-1 - 4]. Note "-1" designates ShapeGlot')
    accuracy_per_saliency = augmented_df.groupby('saliency')['guessed_correct'].mean()
    evaluation_results[split]['accuracy_per_saliency'] = accuracy_per_saliency
    logger.info(accuracy_per_saliency)
    logger.info('\n')


    # Hard vs. easy context analysis
    missing_context_mask = (augmented_df['hard_context'].isna()) & ~(augmented_df.assignmentid == 'shapeglot')
    n_missing = missing_context_mask.sum()
    if n_missing > 0:
        logger.info(f'Warning: {n_missing} non-shapeglot stimuli do not have context hardness information')

    for tag in ['using_all_data', 'excluding_sg_examples']:
        if tag == 'using_all_data':
            temp = augmented_df[~augmented_df.hard_context.isna()]
        elif tag == 'excluding_sg_examples':
            temp = augmented_df[(~augmented_df.hard_context.isna()) & (augmented_df.assignmentid != 'shapeglot')]
        else:
            assert False

        accuracy_on_hard = temp[temp.hard_context]['guessed_correct'].mean()
        evaluation_results[split][f'accuracy_on_hard_{tag}'] = accuracy_on_hard
        logger.info(f"Average Performance in hard pairs for/when {tag}: {accuracy_on_hard}")

        accuracy_on_easy = temp[~temp.hard_context]['guessed_correct'].mean()
        evaluation_results[split][f'accuracy_on_easy_{tag}'] = accuracy_on_easy
        logger.info(f"Average Performance in easy pairs for/when {tag}: {accuracy_on_easy}")


    ## per shape-class, per source, or target dataset (ShapeNet, vs. ModelNet vs. PartNet) analysis
    for key in ['target_object_class', 'source_dataset', 'target_dataset']:
        acc_per_key = augmented_df.groupby(key)['guessed_correct'].mean()
        evaluation_results[split][f'accuracy_analyzed_per_{key}'] = acc_per_key
        logger.info(f'Accuracy by grouping on: {key}')
        logger.info(acc_per_key.sort_values().to_string() + '\n')


    ## Final and more "Esoteric" analysis regarding the effect of seeing the distractor during the listening training.
    if split != 'train':
        targets = set(augmented_df.target.unique())
        distractor_in_training = augmented_df.distractor_1.isin(training_examples)
        distractor_fraction_in_train = f"{distractor_in_training.sum()} / {len(distractor_in_training)}"
        evaluation_results[split]['distractor_fraction_in_train'] = distractor_fraction_in_train
        logger.info(
            f'Examples containing distractors that where seen as targets or '
            f'distractors during training {distractor_fraction_in_train}')

        ac1 = augmented_df[distractor_in_training].guessed_correct.mean()
        evaluation_results[split]['accuracy_when_distractor_in_train'] = ac1
        logger.info(f'Accuracy when the distractor was seen in training {ac1}')
        ac2 = augmented_df[~distractor_in_training].guessed_correct.mean()
        evaluation_results[split]['accuracy_when_distractor_not_in_train'] = ac2
        logger.info(f'Accuracy when the distractor was *not* seen in training {ac2}')
    logger.info('\n')


if hasattr(args, 'save_analysis_results') and args.save_analysis_results:
    pickle_data(osp.join(args.log_dir, 'analysis_of_trained_listener.pkl'), evaluation_results)