"""
Routines that concern I/O operations directly relevant to the training/testing of a -ChangeIt3DNet- architecture.

Originally created sometime around 2021, for Python 3.x
Around 2022 Panos Achlioptas (https://optas.github.io)
"""

import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from functools import partial
from .basics import unpickle_data
from .datasets.shape_glot import add_sg_to_snt
from .language_contrastive_dataset import LanguageContrastiveDataset
from ..language.vocabulary import Vocabulary
from ..models.listening_oriented import evaluate_listener


def load_pickled_shape_latent_codes(args):
    shape_to_latent_code = next(unpickle_data(args.latent_codes_file))
    shape_latent_dim = len(list(shape_to_latent_code.values())[0])
    return shape_to_latent_code, shape_latent_dim


def prepare_input_data(args, logger=None):
    def _print(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    shape_to_latent_code, shape_latent_dim = load_pickled_shape_latent_codes(args)
    msg = 'Latent codes with dimension {} are loaded.'.format(shape_latent_dim)
    _print(msg)

    df = pd.read_csv(args.shape_talk_file)
    
    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if hasattr(args, "add_shape_glot") and args.add_shape_glot:
        raise NotImplementedError('Not in public version')
        df = add_sg_to_snt(df, vocab, args.split_file)

    # make df compatible with LanguageContrastive Dataset
    df = df.assign(target=df.target_uid)
    df = df.assign(distractor_1=df.source_uid)

    # constrain training in language of particular classes
    if len(args.restrict_shape_class) > 0:
        mask = df.target_object_class.isin(set(args.restrict_shape_class))
        df = df[mask].copy()
        df.reset_index(inplace=True, drop=True)

        msg = 'Restricting to class(es) {}. Total utterances: {}'.format(args.restrict_shape_class, len(df))
        _print(msg)

    # Last remove training/val stimuli for which the source (distractor) 
    # shape wins the comparison against the ground-truth target i.e., the listener is wrong here.
    if args.clean_train_val_data:
        device = torch.device("cuda:" + str(args.gpu_id))
        pretrained_listener = torch.load(args.pretrained_listener_file).to(device)
        
        def to_stimulus_func(x):
            return shape_to_latent_code[x]

        dataset = LanguageContrastiveDataset(df, to_stimulus_func, n_distractors=1, shuffle_items=False)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        listening_res = evaluate_listener(pretrained_listener, dataloader, device=device, return_logits=True)
                        
        # print its test-accuracy.
        listening_acc = ((listening_res['logits'].argmax(1) == 1) & (df.listening_split == 'test')).sum()
        listening_acc /= (df.listening_split == 'test').sum()
        _print(f"Pretrained Listener has a test accuracy of: {listening_acc}")

        # drop train/val confusing examples
        drop_mask = (listening_res['logits'].argmax(1) != 1) & (df.changeit_split.isin(['train', 'val']))
        _print(f'Dropping {sum(drop_mask)} examples from train/val because the provided listener does not correctly predict the target for them.')
        df = df[~drop_mask].copy()
        df.reset_index(inplace=True, drop=True)

    return df, shape_to_latent_code, shape_latent_dim, vocab


def shape_uid_to_stimulus(uid, shape_to_latent_code=None):
    if shape_to_latent_code is not None:
        return shape_to_latent_code[uid]


def prepare_input_data_loaders(df, shape_to_latent_code, args):
    to_stimulus_func = partial(shape_uid_to_stimulus, shape_to_latent_code=shape_to_latent_code)

    data_loaders = dict()
    for split in ['train', 'val', 'test']:
        ndf = df[df.listening_split == split].copy()
        ndf.reset_index(inplace=True, drop=True)

        seed = None if split == 'train' else args.random_seed
        batch_size = args.batch_size if split == 'train' else 2 * args.batch_size

        dataset = LanguageContrastiveDataset(ndf,
                                             to_stimulus_func,
                                             n_distractors=1,
                                             shuffle_items=False)  # important, target *always* last

        data_loaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=batch_size,
                                                          shuffle=split == 'train',
                                                          num_workers=args.num_workers,
                                                          worker_init_fn=lambda x: np.random.seed(seed))
    return data_loaders


def dataloader_for_expression(expression, vocab, shape_uids, to_stimulus_func, batch_size=None, num_workers=10):
    """
    :param expression:  list of tokens, describing a single expression
    :param vocab:
    :param shape_uids: pandas dataframe listing uids of shapes
    :param to_stimulus_func:
    :param batch_size:
    :param num_workers:
    :return:
    """
    df = shape_uids.copy()
    df = df.assign(tokens_encoded=[vocab.encode(expression)] * len(df))
    dataset = LanguageContrastiveDataset(df, to_stimulus_func, n_distractors=0)

    if batch_size is None:
        batch_size = len(df)

    res = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    return res


def shape_with_expression_dataloader_convenient(dloader, expressions, vocab, batch_size=None,
                                                shape_class=None, num_workers=10, verbose=False):
    assert type(expressions[0]) is list
    max_len = max([len(exp) for exp in expressions])

    if shape_class is not None:
        if type(shape_class) == str:
            mask_class = dloader.dataset.df.target_object_class == shape_class
        else:
            assert type(shape_class) == list
            mask_class = dloader.dataset.df.target_object_class.isin(shape_class)
    else:
        n_examples = len(dloader.dataset.df)
        mask_class = pd.Series([True] * n_examples)

    shape_uids = pd.DataFrame(dloader.dataset.df.target[mask_class].unique())  # keep each target once
    shape_uids = shape_uids.assign(key=1)
    shape_uids.columns = ['target', 'key']
    shape_uids = shape_uids.assign(target_object_class=shape_uids.target.apply(lambda x: x.split('/')[0])) # convention: class/dataset/name

    tokens = []
    for exp in expressions:
        tokens.append(vocab.encode(exp, max_len=max_len))
    tokens = pd.DataFrame([tokens]).T
    tokens.columns = ['tokens_encoded']
    tokens = tokens.assign(key=1)

    result = pd.merge(shape_uids, tokens, on='key').drop("key", 1)

    to_stimulus_func = dloader.dataset.to_stimulus_func
    dataset = LanguageContrastiveDataset(result, to_stimulus_func, n_distractors=0)
    assert len(dataset) == len(shape_uids) * len(expressions)
    if batch_size is None:
        batch_size = len(dataset)

    res = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers)
    if verbose:
        print(f"max-expression-len: {max_len} "
              f"len-shapes: {len(shape_uids)}, "
              f"len-expressions: {len(expressions)}, "
              f"len-dataset: {len(dataset)}")
    return res

