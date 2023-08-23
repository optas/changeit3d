"""
Script to evaluate a Language-Assisted 3D Shape Edit/Deformation System (ChangeIt3D)

Notice the main code for the metric-evaluation is at the function ```run_all_metrics'''.
"""


import torch
import numpy as np
import pandas as pd
import os.path as osp
from functools import partial

from changeit3d.in_out.changeit3d_net import prepare_input_data
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.in_out.pointcloud import pc_loader_from_npz, uniform_subsample
from changeit3d.in_out.basics import pickle_data
from changeit3d.in_out.basics import create_logger
from changeit3d.in_out.arguments import parse_evaluate_changeit3d_arguments

from changeit3d.utils.distances import k_euclidean_neighbors
from changeit3d.utils.basics import parallel_apply

from changeit3d.models.model_descriptions import load_pretrained_changeit3d_net
from changeit3d.models.model_descriptions import load_pretrained_pc_ae

from changeit3d.evaluation.auxiliary import pc_ae_transform_point_clouds, sgf_transform_point_clouds
from changeit3d.evaluation.all_metrics import run_all_metrics

from changeit3d.external_tools.sgf.loader import initialize_and_load_sgf


args = parse_evaluate_changeit3d_arguments()
logger = create_logger(args.log_dir)


##
# Prepare the input data
##
df, shape_to_latent_code, shape_latent_dim, vocab = prepare_input_data(args, logger)

##
# Prepare the data loaders
##
def to_stimulus_func(x):
    return shape_to_latent_code[x]

split = 'test'
ndf = df[df.changeit_split == split].copy()
ndf.reset_index(inplace=True, drop=True)

if args.sub_sample_dataset is not None:
    np.random.seed(args.random_seed)
    ndf = ndf.sample(args.sub_sample_dataset)
    ndf.reset_index(inplace=True, drop=True)
    
dataset = LanguageContrastiveDataset(ndf,
                                     to_stimulus_func,
                                     n_distractors=1,
                                     shuffle_items=False)  # important, source (distractor) now is first

dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         worker_init_fn=lambda _ : np.random.seed(args.random_seed)) 


##
# Loading pretrained ChangetIt3DNet.
##
logger.info('Loading pretrained ChangetIt3DNet (C3DNet)')
c3d_net, best_epoch, c3d_args = load_pretrained_changeit3d_net(args.pretrained_changeit3d, shape_latent_dim, vocab)
device = torch.device("cuda:" + str(args.gpu_id))
c3d_net = c3d_net.to(device)
logger.info(f'The model is variant `{c3d_args.shape_editor_variant}` trained with {c3d_args.identity_penalty} identity penalty and Self-Contrast={c3d_args.self_contrast}.')
logger.info(f'Loaded at epoch {best_epoch}.')


##
# Loading pretrained Shape Generator (AutoEncoder).
##
if args.shape_generator_type == 'pcae':
    pc_ae, pc_ae_args = load_pretrained_pc_ae(args.pretrained_shape_generator)
    pc_ae = pc_ae.to(device)
    pc_ae = pc_ae.eval()
elif args.shape_generator_type == 'sgf':    
    sgf_ae = initialize_and_load_sgf()
    sgf_trainer = sgf_ae.trainer
    sgf_trainer.cfg.inference.num_points = args.n_sample_points
elif args.shape_generator_type == 'imnet':
    raise NotImplementedError()
else:
    assert False


check_loader = dataloader
gt_classes = check_loader.dataset.df.source_object_class

# Decode edits
if args.shape_generator_type == 'pcae':
    transformation_results = pc_ae_transform_point_clouds(pc_ae,
                                                          c3d_net,
                                                          check_loader,
                                                          stimulus_index=0,
                                                          scales=[0, 1],  # use "0" to get also the simple reconstruction of the decoder (no edit)
                                                          device=device)
        
elif args.shape_generator_type == 'sgf':        
    transformation_results = sgf_transform_point_clouds(sgf_trainer, 
                                                        c3d_net,
                                                        check_loader,
                                                        stimulus_index=0,                                                            
                                                        scales=[1],      # add "0" to get also the simple reconstruction of the decoder (no edit)
                                                        n_points_per_shape=args.n_sample_points, 
                                                        batch_size=512,   # you might want to change this depending on the size of your GPU 
                                                        max_recon_batches=None)    
elif args.shape_generator_type == 'imnet':
        raise NotImplementedError()        
else:
        assert False
        
transformed_shapes = transformation_results['recons'][1]
language_used = [vocab.decode_print(s) for s in transformation_results['tokens']]
gt_pc_files = check_loader.dataset.df.source_uid.apply(lambda x: osp.join(args.top_pc_dir, x + ".npz")).tolist()

if args.save_reconstructions:
    outputs = dict()
    outputs['transformed_shapes'] = transformed_shapes
    outputs['language_used'] = language_used
    outputs['gt_input_pc_files'] = gt_pc_files
    pickle_data(osp.join(args.log_dir, 'evaluation_outputs.pkl'), outputs)
    
## Sample point-clouds to desired granularity for evaluation            
if transformed_shapes.shape[-2] != args.n_sample_points:
    transformed_shapes = np.array([uniform_subsample(s, args.n_sample_points, args.random_seed)[0] for s in transformed_shapes])
    
## Load ground-truth input point-clouds
pc_loader =  partial(pc_loader_from_npz, n_samples=args.n_sample_points, random_seed=args.random_seed)
gt_pcs = parallel_apply(gt_pc_files, pc_loader, n_processes=20) # or, gt_pcs = [pc_loader(m) for m in gt_pc_files]
gt_pcs = np.array(gt_pcs)

sentences = ndf.utterance_spelled.values
gt_classes = gt_classes.values
results_on_metrics = run_all_metrics(transformed_shapes, gt_pcs, gt_classes, sentences, vocab, args, logger)

# Save (pickle) results 
pickle_data(osp.join(args.log_dir, 'evaluation_metric_results.pkl'), results_on_metrics)


if args.evaluate_retrieval_version:
    ## Nearest neighbor (retrieval) baseline    
    df_temp = pd.read_csv(args.shape_talk_file)
    all_train_shapes = df_temp[df_temp.changeit_split == 'train']['source_uid'].unique()
    print('number of all training shapes', len(all_train_shapes))

    train_latens = torch.from_numpy(np.array([shape_to_latent_code[s] for s in all_train_shapes]))
    transformed_latents = torch.from_numpy(transformation_results['z_codes'][1])

    _, n_ids = k_euclidean_neighbors(1, transformed_latents, train_latens)
    retrieved_shapes_uids = all_train_shapes[n_ids.squeeze().tolist()]

    retrieved_files = [osp.join(args.top_pc_dir, x + ".npz") for x in retrieved_shapes_uids]
    pc_loader =  partial(pc_loader_from_npz, n_samples=args.n_sample_points, random_seed=args.random_seed)
    retrieved_pcs = np.array(parallel_apply(retrieved_files, pc_loader, n_processes=20))

    results_on_retrieval_version = run_all_metrics(retrieved_pcs, gt_pcs, gt_classes, sentences, vocab, args, logger)
    
    # Save (pickle) results 
    pickle_data(osp.join(args.log_dir, 'evaluation_metric_results_for_retrieval.pkl'), results_on_retrieval_version)
