"""
Argument handling.
Originally created in 2020, for Python 3.x
2022 Panos Achlioptas (optas.github.io)
"""
import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from termcolor import colored
from .basics import create_dir


def str2bool(v):
    """ boolean values for argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def positive_int(value):
    """
    Make sure the passed value to argparse is convertible to a positive integer or raise a Type Error.
    Args:
        value: the value to be checked
    Returns: the value converted to an integer
    Example:
        parser = argparse.ArgumentParser(...)
        parser.add_argument('foo', type=positive_int)
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def _finish_parsing_args(parser, notebook_options, save_args=False):
    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()  # Read from command line.

    if args.experiment_tag is not None:
        args.log_dir = osp.join(args.log_dir, args.experiment_tag)

    if args.use_timestamp:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        args.log_dir = osp.join(args.log_dir, timestamp)

    if not osp.exists(args.log_dir):
        create_dir(args.log_dir)

    # pprint them
    print(colored('\n\nInput arguments:\n\n', 'red'))

    args_string = pprint.pformat(vars(args))
    print(args_string)

    if save_args:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args


def parse_train_test_pc_ae_arguments(notebook_options=None, save_args=True):
    """ Default/Main arguments for training or evaluating a PC based deep AE.
    :param notebook_options: (optional) list with arguments passed as strings. This can be handy e.g., if you are calling
        the function inside a jupyter notebook. Else, the arguments will be read by the command line.
    :return: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='train/test a pointcloud-based AE.')

    # Non-optional arguments
    parser.add_argument('-log_dir', type=str, required=True, help='where to save training-progress, model, etc.')
    parser.add_argument('-data_dir', type=str, required=True, help='top directory containing pointcloud data')
    parser.add_argument('-split_file', type=str, required=True, help='csv file indicating the split (train/test/val)'
                                                                     'for each pointcloud datum')

    # Model parameters
    parser.add_argument('--n_pc_points', type=int, default=2**13, help='points per shape')
    parser.add_argument('--encoder_net', type=str, default='pointnet', help='encoding architecture')
    parser.add_argument('--decoder_net', type=str, default='mlp')
    parser.add_argument('--encoder_conv_layers', type=int, nargs='+', default=[32, 64, 64, 128, 256])
    parser.add_argument('--decoder_fc_neurons', type=int, nargs='+', default=[256, 256, 512])

    # Training parameters
    parser.add_argument('--do_training', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=positive_int, default=350)
    parser.add_argument('--loss_function', type=str, default='chamfer', choices=['chamfer', 'emd'])
    parser.add_argument('--train_patience', type=int, default=14, help='maximum consecutive epochs where the '
                                                                       'validation loss does not improve '
                                                                       'before we stop training.')
    parser.add_argument('--lr_patience', type=int, default=10, help='maximum waiting of epochs where the validation '
                                                                    'reconstruction e.g., Chamfer loss does not '
                                                                    'improve before we reduce the learning-rate.')
    parser.add_argument('--save_each_epoch', type=str2bool, default=False, help='Save the model at each epoch, '
                                                                                'else will only save the one that '
                                                                                'achieved the minimal per-validation '
                                                                                'split loss.')
    parser.add_argument('--deterministic_point_cloud_sampling', type=str2bool, default=False,
                        help="During training, for any given shape always use the same pointcloud (if True), "
                             " or allow some stochastic variations based on pointcloud sub-sampling. "
                             "Note. for test/val data we use deterministic sub-sampling")

    # Data related parameters
    parser.add_argument('--batch_size', type=positive_int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--scale_in_u_sphere', type=str2bool, default=False, help="feed input pointcloud that are first put into unit-sphere ")

    # Misc
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--use_timestamp', default=True, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Testing
    parser.add_argument('--load_pretrained_model', type=str2bool, default=False)
    parser.add_argument('--pretrained_model_file', type=str)
    parser.add_argument('--extract_latent_codes', type=str2bool, default=True)

    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args


def parse_train_test_pc_clf_arguments(notebook_options=None, save_args=True):
    """ Default/Main arguments for training or evaluating a PC based deep classification network.
    :param notebook_options: (optional) list with arguments passed as strings. This can be handy e.g., if you are calling
        the function inside a jupyter notebook. Else, the arguments will be read by the command line.
    :return: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='train/test a pointcloud-based classifier.')

    # Non-optional arguments
    parser.add_argument('-data_dir', type=str, required=True, help='top directory containing point-cloud data')
    parser.add_argument('-split_file', type=str, required=True, help='csv file indicating the split (train/test/val)'
                                                                     'for each pointcloud datum')

    # Model parameters
    parser.add_argument('--n_pc_points', type=int, default=2048, help='points per shape')
    parser.add_argument('--encoder_net', type=str, default='pointnet', help='encoding architecture')
    parser.add_argument('--decoder_net', type=str, default='mlp')
    parser.add_argument('--encoder_conv_layers', type=int, nargs='*')
    parser.add_argument('--decoder_fc_neurons', type=int, nargs='*')

    # Training parameters
    parser.add_argument('--do_training', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=int, default=100)
    parser.add_argument('--train_patience', type=int, default=10, help='maximum consecutive epochs where the '
                                                                       'validation loss does not improve '
                                                                       'before we stop training.')
    parser.add_argument('--lr_patience', type=int, default=6, help='maximum waiting of epochs where the validation '
                                                                    'reconstruction e.g., Chamfer loss does not '
                                                                    'improve before we reduce the learning-rate.')
    parser.add_argument('--save_each_epoch', type=str2bool, default=False, help='Save the model at each epoch, '
                                                                                'else will only save the one that '
                                                                                'achieved the minimal per-validation '
                                                                                'split loss.')
    parser.add_argument('--deterministic_point_cloud_sampling', type=str2bool, default=False,
                        help="During training, for any given shape always use the same pointcloud (if True), "
                             " or allow some stochastic variations based on pointcloud sub-sampling. "
                             "Note. for test/val data we use deterministic sub-sampling")

    # Data related parameters
    parser.add_argument('--batch_size', type=positive_int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--n_classes', type=positive_int, help="if not provided will be inferred by the training "
                                                               "classes of ""the split_df")
    parser.add_argument('--scale_in_u_sphere', type=str2bool, default=True, help="feed input pointcloud that are first put into unit-sphere ")
    

    # Misc
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save training-progress, model, etc.')
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--use_timestamp', default=True, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Testing
    parser.add_argument('--pretrained_model_file', type=str)

    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args


def parse_train_test_latent_listener_arguments(notebook_options=None, save_args=True):
    """ Default/Main arguments for training or evaluating a neural (latent-based) listener.
    :param notebook_options: (optional) list with arguments passed as strings. This can be handy e.g., if you are calling
        the function inside a jupyter notebook. Else, the arguments will be read by the command line.
    :return: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='train/test latent neural listener')

    # Non-optional arguments
    parser.add_argument('-shape_talk_file', type=str, required=True, help='referential language data')
    parser.add_argument('-vocab_file', type=str, required=True, help='vocabulary file')
    parser.add_argument('-latent_codes_file', type=str, required=True, help='shape_uid_to_latent_code dictionary')

    # Dataset oriented
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--add_shape_glot', type=str2bool, default=False)

    # Listening-model parameters
    parser.add_argument('--listening_model', type=str, default='ablation_model_one', help="ablation_model_one is transformer-based"
                                                                                          "ablation_model_two is lstm-based")

    # Training parameters
    parser.add_argument('--do_training', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_patience', type=int, default=15)
    parser.add_argument('--lr_patience', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # Testing only/explicit parameters
    parser.add_argument('--save_analysis_results', type=str2bool, default=True)

    # Misc
    parser.add_argument('--pretrained_model_file', type=str, help="if provided, the underlying pre-trained listener "
                                                                  "will be loaded before new training, etc. happens")
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save checkpoints, etc.')
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--use_timestamp', default=True, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args


def parse_train_test_raw_listener_arguments(notebook_options=None, save_args=True):
    """ Default/Main arguments for training or evaluating a neural raw (pointcloud-based) listener.
    :param notebook_options: (optional) list with arguments passed as strings. This can be handy e.g., if you are calling
        the function inside a jupyter notebook. Else, the arguments will be read by the command line.
    :return: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='train/test raw pointcloud neural listener')

    # Non-optional arguments
    parser.add_argument('-shape_talk_file', type=str, required=True, help='referential language data')
    parser.add_argument('-vocab_file', type=str, required=True, help='vocabulary file')
    parser.add_argument('-top_raw_pc_dir', type=str, required=True, help='directory where ShapeTalk pointclouds are located')

    # Dataset oriented
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--n_pc_points', type=int, default=2048, help="number of extracted points per input shape")    

    # Listening-model parameters
    parser.add_argument('--listening_model', type=str, default='dgcnn-ablation1', 
                        choices=["pointnet-ablation1", "dgcnn-ablation1"])

    # Training parameters
    parser.add_argument('--do_training', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_patience', type=int, default=15)
    parser.add_argument('--lr_patience', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # Testing only/explicit parameters
    parser.add_argument('--save_analysis_results', type=str2bool, default=True)

    # Misc
    parser.add_argument('--pretrained_model_file', type=str, help="if provided, the underlying pre-trained listener "
                                                                  "will be loaded before new training, etc. happens")
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save checkpoints, etc.')
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--use_timestamp', default=True, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args


def parse_train_changeit3d_arguments(notebook_options=None, save_args=True):
    """ Default/Main arguments for training or evaluating a shape editor (ChangeIt3D) guided by language.
    :param notebook_options: (optional) list with arguments passed as strings. This can be handy e.g., if you are calling
        the function inside a jupyter notebook. Else, the arguments will be read by the command line.
    :return: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description='train/test language-assisted shape editor (changeit3d)')

    # Non-optional arguments
    parser.add_argument('-shape_talk_file', type=str, required=True, help='referential language data')
    parser.add_argument('-vocab_file', type=str, required=True, help='vocabulary file')
    parser.add_argument('-latent_codes_file', type=str, required=True, help='shape_uid_to_latent_code dictionary')
    parser.add_argument('-pretrained_listener_file', type=str, required=True)

    # Dataset oriented
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--add_shape_glot', type=str2bool, default=False)
    parser.add_argument('--clean_train_val_data', type=str2bool, default=True, help="use the guiding listener to drop examples it misclassifies")
    
    # Training parameters
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=positive_int, default=150)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_patience', type=int, default=10)
    parser.add_argument('--lr_patience', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0)    

    # Specialized for shape editors:
    parser.add_argument('--identity_penalty', type=float, default=0)
    parser.add_argument('--shape_editor_variant', type=str, default='decoupling_mag_direction', 
                        choices=['decoupling_mag_direction', 'coupled'], 
                        help="to decouple the computation of the magnitude of the edit from the direction, or not")
    parser.add_argument('--self_contrast', type=str2bool, default=True)
    parser.add_argument('--adaptive_id_penalty', type=str, default='')

    # Misc
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save checkpoints, etc.')
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--use_timestamp', default=True, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args


def parse_evaluate_changeit3d_arguments(notebook_options=None, save_args=True):

    parser = argparse.ArgumentParser(description='Evaluation of 3D lang-assisted shape editor')
    
    
    parser.add_argument('-shape_talk_file', type=str, required=True, help='referential language data')
    parser.add_argument('-vocab_file', type=str, required=True, help='vocabulary file')
    parser.add_argument('-latent_codes_file', type=str, required=True, help='shape_uid_to_latent_code dictionary')    
    parser.add_argument('-pretrained_changeit3d', type=str, required=True, help='string pointing to saved file')    
    parser.add_argument('-top_pc_dir', type=str, required=True, help='top dir location of gt pointclouds')
    
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=['chair', 'table', 'lamp'])        
    parser.add_argument('--pretrained_shape_classifier', type=str, help='if given, will be used to measure '
                                                                        'the Class-Preservation (CP) score.')    
    parser.add_argument('--compute_fpd', default=True, type=str2bool, help='if shape classifier is given and this is True, it will also compute Frechet PointCloud based Distance')
    parser.add_argument('--shape_part_classifiers_top_dir', type=str, help='if given, pretrained classifiers located here will be loaded and will be used to measure '
                                                                           'localized-GD (l-GD) score under Chamfer loss')
    parser.add_argument('--pretrained_oracle_listener', type=str, help='if given, will be used to measure '
                                                                       'the Linguistic-Association Boost (LAB) score')

    parser.add_argument('--shape_generator_type', type=str, default="pcae", choices=["pcae", "sgf", "imnet"])
    parser.add_argument('--pretrained_shape_generator', type=str, required=False, help='you must pass it when using pcae')

    parser.add_argument('--n_sample_points', type=positive_int, default=2048, help="extracted pointcloud points per shape used for evaluation")

    parser.add_argument('--sub_sample_dataset', type=positive_int)

    parser.add_argument('--gpu_id', type=int, default=0)
    
    parser.add_argument('--save_reconstructions', default=False, type=str2bool, help="save or not the output transformed shapes")

    parser.add_argument('--use_timestamp', default=False, type=str2bool, help='use launch time for logging')

    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging purposes')

    parser.add_argument('--random_seed', type=int, default=2022)

    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save checkpoints, etc.')
    
    parser.add_argument('--clean_train_val_data', type=str2bool, default=False)
    
    parser.add_argument('--batch_size', type=int, default=1024)
    
    parser.add_argument('--num_workers', type=int, default=10)
    
    parser.add_argument('--evaluate_retrieval_version', type=str2bool, default=False, help='execute a nearest-neighbor retrieval '
                                                                                           'in the latent space instead of decoding '
                                                                                           'the transformed shape (see paper for details).')
    
    args = _finish_parsing_args(parser, notebook_options, save_args)
    
    return args




def parse_train_test_monolithic_changeit3d(notebook_options=None, save_args=True):

    parser = argparse.ArgumentParser(description='Train/test Monolithic alternative to ChangeIt3DNet')
    
    # Non-optional
    parser.add_argument('-shape_talk_file', type=str, required=True, help='referential language data')
    parser.add_argument('-vocab_file', type=str, required=True, help='vocabulary file')
    parser.add_argument('-top_pc_dir', type=str, required=True, help='top dir location of gt pointclouds')
        
    # Training parameters
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--max_train_epochs', type=positive_int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_patience', type=int, default=5)
    parser.add_argument('--lr_patience', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0)    
    
    # Data related
    parser.add_argument('--n_pc_points', type=positive_int, default=2048, help="pointcloud points per shape for shape reconstructions (output of monolithic system)")    
    parser.add_argument('--restrict_shape_class', type=str, nargs='*', default=[])
    parser.add_argument('--scale_in_u_sphere', type=str2bool, default=False, help="feed input pointcloud that are first put into unit-sphere ")
    
    # Test related (i.e., evaluate the model)        
    parser.add_argument('--test', type=str2bool, default=False)
    parser.add_argument('--pretrained_model', type=str, required=False, help='string pointing to saved file to be loaded first')
    parser.add_argument('--n_sample_points', type=positive_int, default=2048, help="extracted pointcloud points per shape for evaluation purposes")    
    parser.add_argument('--pretrained_shape_classifier', type=str, help='if given, will be used to measure '
                                                                        'the Class-Preservation (CP) score.')    
    parser.add_argument('--compute_fpd', default=True, type=str2bool, help='if shape classifier is given and this is True, it will also compute Frechet PointCloud based Distance')    
    parser.add_argument('--shape_part_classifiers_top_dir', type=str, help='if given, pretrained classifiers located here will be loaded and will be used to measure '
                                                                            'localized-GD (l-GD) score under Chamfer loss')
    parser.add_argument('--pretrained_oracle_listener', type=str, help='if given, will be used to measure the Linguistic-Association Boost'
                                                                        '(LAB) score.')
        
          
    # Misc
    parser.add_argument('--log_dir', type=str, default='./logs', help='where to save checkpoints, etc.')
    parser.add_argument('--random_seed', type=int, default=2022)
    parser.add_argument('--use_timestamp', default=False, type=str2bool, help='use launch time for logging')
    parser.add_argument('--experiment_tag', type=str, help='will be used to for logging')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = _finish_parsing_args(parser, notebook_options, save_args)
    return args