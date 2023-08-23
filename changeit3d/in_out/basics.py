"""
Basic functions that deal primarily with I/O operations such counting files under a directory, pickling data etc.
    Also, includes some basic functionality oriented with I/O for pytorch (save/load models) etc.

Originally created at 9/18/20, for Python 3.x
2020 Panos Achlioptas (https://optas.github.io)
"""

import os
import re
import sys
import json
import torch
import pprint
import logging
import warnings
import os.path as osp
from six.moves import cPickle
from argparse import ArgumentParser


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def create_dir(dir_path):
    """Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def files_in_subdirs(top_dir, search_pattern):
    join = os.path.join
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


def immediate_subdirectories(top_dir, full_path=True):
    dir_names = [name for name in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, name))]
    if full_path:
        dir_names = [osp.join(top_dir, name) for name in dir_names]
    return dir_names


def splitall(path):
    """Split a filepath to its constituents pieces
    :param path: (string)
    :return: list
    Examples:
        splitall('a/b/c') -> ['a', 'b', 'c']
        splitall('/a/b/c/')  -> ['/', 'a', 'b', 'c', '']

    NOTE: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = osp.split(path)
        if parts[0] == path:   # Sentinel for absolute paths.
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # Sentinel for relative paths.
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def read_saved_args(config_file, override_or_add_args=None, verbose=False):
    """
    :param config_file: json file containing arguments
    :param override_args: dict e.g., {'gpu': '0'} will set the resulting arg.gpu to be 0
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_or_add_args is not None:
        for key, val in override_or_add_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args

##
# Pytorch oriented basic I/O routines:
##

def torch_save_model(model, path):
    """ Wrap torch.save to catch standard warning of not finding the nested implementations.
    :param model:
    :param path:

    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.save(model, path)


def torch_load_model(checkpoint_file, map_location=None):
    """ Wrap torch.load to catch standard warning of not finding the nested implementations.
    :param checkpoint_file:
    :param map_location:
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(checkpoint_file, map_location=map_location)
    return model


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """ Save torch items with a state_dict"""
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """ Load torch items from saved state_dictionaries"""
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
