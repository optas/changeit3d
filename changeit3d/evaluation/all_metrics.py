'''
Functions implementing metric evaluations for ChangeIt3d (C3D).

Notice. If you want to use your own pre-trained shape_classifier, or shape_part_classifier, be sure to adapt the input/output
to/of the provided ChangeIt3D model so the 3D data (e.g., PointClouds) are consistently scaled/aligned across those neural-networks.

To this end, see the input `network_input_transformations' functions used below by the ``run_all_metrics'' function.
        
circa 2022, Panos Achlioptas (https://optas.github.io)
'''

import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch.utils.data import DataLoader
from collections import OrderedDict
from functools import partial

from changeit3d.in_out.basics import unpickle_data
from changeit3d.in_out.pointcloud import (PointcloudDataset, 
                                          swap_axes_of_pointcloud,
                                          center_in_unit_sphere)
from changeit3d.evaluation.generic_metrics import (chamfer_dists,
                                                   chamfer_dists_on_masked_pointclouds,
                                                   difference_in_probability,
                                                   get_clf_probabilities)
from changeit3d.evaluation.listening_based import listening_fit_on_raw_pcs
from changeit3d.evaluation.semantic_part_based import mark_part_reference_in_sentences, masks_of_referred_parts_for_pcs
from changeit3d.evaluation.fpd import calculate_fpd
from changeit3d.models.shape_part_segmentors import shape_net_parts_segmentor_inference, load_shape_net_parts_segmentor


network_transformations = dict()

## this is the transformation we use in our pretrained pc-based-classifier
## YOU MUST Change this accordingly if you use a different classifier
network_transformations['shape_classifier'] = partial(center_in_unit_sphere, in_place=False)

def pc_transform_for_part_predictor(pc):
    pc = swap_axes_of_pointcloud(pc, [0, 2, 1])
    pc = center_in_unit_sphere(pc)
    return pc        

## this is the transformation we use in our pretrained part-predictor-classifier
## YOU MUST Change this accordingly if you use a different part-predictor
network_transformations['part_predictor'] = pc_transform_for_part_predictor


@torch.no_grad()
def run_all_metrics(transformed_shapes, gt_pcs, gt_classes, sentences, vocab, args, logger, 
                    network_input_transformations=network_transformations):
    """
    transformed_shapes:  (np.array): N_shapes, N_points_per_shape, 3
    gt_pcs: (np.array): N_shapes, N_points_per_shape, 3
    
    gt_classes: (1-D iterable) with names of classes per shape

    sentences: (list of strings) strings used to transform gt_pcs into the transformed_shapes, they are assumed to 
                                 be space tokenizable already (i.e., by a simple .split() operation)
    """
    
    
    results_on_metrics = dict()
    
    device = torch.device("cuda:" + str(args.gpu_id))
    
    #Prepare input:
    gt_classes = pd.Series(gt_classes, name='shape_class')  #convert to pandas for ease of use via .groupby    
    tokens = pd.Series([sent.split() for sent in sentences])
    max_len = tokens.apply(len).max()
    tokens_encoded = np.stack(tokens.apply(lambda x: vocab.encode(x, max_len=max_len))) # encode to ints and put them in a N-shape array
    
    
    #############################################
    # Test-1: Holistic (regular) Chamfer distance
    #############################################
    scale_chamfer_by = 1000
    batch_size_for_cd = 2048
    for cmp_with in [gt_pcs]:
        holistic_cd_mu, holistic_cds = chamfer_dists(cmp_with, transformed_shapes,
                                                    bsize=min(len(transformed_shapes), batch_size_for_cd), 
                                                    device=device)
        torch.cuda.empty_cache()
        score = round(holistic_cd_mu * scale_chamfer_by, 3)
        score_per_class = (pd.concat([gt_classes, pd.DataFrame(holistic_cds.tolist())], axis=1).groupby('shape_class').mean()*scale_chamfer_by).round(3)
        score_per_class = score_per_class.reset_index().rename(columns={0: 'holistic-chamfer'})
        
        logger.info(f"Chamfer Distance (all pairs), Average: {score}")    
        logger.info(f"Chamfer Distance (all pairs), Average, per class:")
        logger.info(score_per_class)
        
        results_on_metrics['Chamfer_all_pairs_average'] = score
        results_on_metrics['Chmafer_all_pairs_per_class'] = score_per_class


    #############################################
    # Test-2: LAB
    #############################################    
    # Loading (optionally) a separately trained ORACLE neural listener, to be used for LAB measurements.
    oracle_listener = None
    if args.pretrained_oracle_listener:
        oracle_listener = torch.load(args.pretrained_oracle_listener).eval().to(device)
        _, all_boosts, avg_wins, avg_boost = listening_fit_on_raw_pcs(gt_pcs, transformed_shapes, tokens_encoded, oracle_listener, device=device)
        
        torch.cuda.empty_cache()
        logger.info(f"LAB Average:{avg_boost}")
        logger.info(f"LAB-related-metric: Times edit is favored by listener against the original input, Average:{avg_wins}")
        results_on_metrics['LAB_avg'] = avg_boost
        results_on_metrics['LAB_wins'] = avg_wins

        score_per_class = (pd.concat([gt_classes, pd.DataFrame(all_boosts.tolist())], axis=1).groupby('shape_class').mean())
        score_per_class = score_per_class.reset_index().rename(columns={0:'LAB'})
        logger.info(f"LAB (all pairs), Average, per class:")
        logger.info(score_per_class)

        results_on_metrics['LAB_per_class'] = score_per_class

        
    #############################################
    # Test-3: Class-Distortion
    #############################################   
        
    # Loading (optionally) a shape-clf to measure the Class-Distortion (CD) score
    shape_clf = None
    if args.pretrained_shape_classifier is not None:
        shape_clf = torch.load(args.pretrained_shape_classifier).to(device)
        clf_idx_file = osp.join(osp.dirname(args.pretrained_shape_classifier), 'class_name_to_idx.pkl')
        clf_name_to_idx = next(unpickle_data(clf_idx_file))
        logger.info(f'A classifier trained to recognize {len(clf_name_to_idx)} shape classes was loaded.')        

    if shape_clf is not None:    
        collected_probs = dict()  # get the classification probabilities for the shapes * pre and post* the edit
        for shapes, tag in zip([gt_pcs, transformed_shapes], ['gt', 'transformed']):    
                        
            shapes_normalized = np.array([network_input_transformations['shape_classifier'](s) for s in shapes])
            collected_probs[tag] = \
                get_clf_probabilities(shape_clf,
                                    shapes_normalized,
                                    clf_feed_key='pointcloud',
                                    clf_res_key='class_logits',
                                    channel_last=True,
                                    bsize=500,
                                    device=device)    
                
        gt_class_labels = gt_classes.apply(lambda x: clf_name_to_idx.get(x, None))
        
        if gt_class_labels.isna().sum() > 0:
            raise ValueError('The classifier was not trained on some of the object classes you are trying to use it for evaluation.')
        gt_class_labels = torch.Tensor(gt_class_labels.tolist()).long()
            
        scores_per_class = dict()
        total_avg_class_distortion = 0
        for u in gt_classes.unique():        
            idx_per_class = (gt_classes[gt_classes == u].index).tolist()
            
            diff = difference_in_probability(collected_probs['gt'][idx_per_class],
                                            collected_probs['transformed'][idx_per_class],
                                            gt_class_labels[idx_per_class])
            
            scores_per_class[u] = diff
            logger.info(f'\n (Average) Class Distortion for {u}: {scores_per_class[u]}')                
            total_avg_class_distortion += diff * len(idx_per_class)
        
        total_avg_class_distortion /= len(gt_classes)                
        logger.info(f'\n (Average) Class Distortion (all classes): {total_avg_class_distortion}')
                
        # print more stats
        for pcs in ['gt', 'transformed']:
            pred = collected_probs[pcs].argmax(1)
            guessed_correct = pred == gt_class_labels
            logger.info(f"\nThe classifier guesses the classes of the ** {pcs} pointclouds ** with accuracy {guessed_correct.double().mean()}")
            per_class_guessing = pd.concat([gt_classes, pd.Series(guessed_correct, name='guessed_correct')], axis=1)
            logger.info(per_class_guessing.groupby('shape_class')['guessed_correct'].mean().to_markdown())
            logger.info('\n')        
            
        results_on_metrics['Class-Distortion (all classes, average)']  = total_avg_class_distortion
        results_on_metrics['Class-Distortion (per class average)'] = pd.DataFrame(scores_per_class, index=['CD']).transpose()
        torch.cuda.empty_cache()
        
    #############################################
    # Test-5: Frechet Pointcloud Distance
    #############################################
    
    if args.pretrained_shape_classifier and args.compute_fpd:
        fpd_scores_per_class = dict()    
        total_avg_fpd = 0        
        for u in gt_classes.unique():        
            idx_per_class = (gt_classes[gt_classes == u].index).tolist()
            input_gt = torch.from_numpy(np.array([network_input_transformations['shape_classifier'](s) for s in gt_pcs[idx_per_class]]))
            input_trans = torch.from_numpy(np.array([network_input_transformations['shape_classifier'](s) for s in transformed_shapes[idx_per_class]]))
                        
            fpd_scores_per_class[u] = calculate_fpd(input_gt, input_trans, pretrained_model_file=args.pretrained_shape_classifier, batch_size=500)
            logger.info(f"Class = {u}, FPD-score = {round(fpd_scores_per_class[u], 3)}")
            
            total_avg_fpd += fpd_scores_per_class[u] * len(idx_per_class)        
                    
        total_avg_fpd /= len(gt_classes)
        logger.info(f'Average across all classes={round(total_avg_fpd, 3)}\n')
        
        results_on_metrics['FPD (all classes, average)']  = total_avg_fpd
        results_on_metrics['FPD (per class average)'] = pd.DataFrame(fpd_scores_per_class, index=['FPD']).transpose()
        torch.cuda.empty_cache()

    #############################################
    # Test-6: Part-Based (Localized) Metrics
    #############################################              

    if args.shape_part_classifiers_top_dir:
        
        lgd_scores = dict()
        lgd_stats = dict()
        lgd_scores['without_parts_per_class'] = dict()
        lgd_scores['GD_on_pairs_with_parts_per_class'] = dict()
        lgd_scores['with_parts_per_class'] = dict()
        lgd_scores['with_parts_normalized_per_class'] = dict()
                                    
        for shape_class in gt_classes.unique():
            # Loading a part-clf to measure the localized-Geometric-Difference (l-GD) score
            file_location = osp.join(args.shape_part_classifiers_top_dir, f'best_seg_model_{shape_class}.pth')
            evaluating_part_predictor = load_shape_net_parts_segmentor(file_location, shape_class)
            evaluating_part_predictor = evaluating_part_predictor.to(device)
                
            idx_per_class = (gt_classes[gt_classes == shape_class].index).tolist()
            input_gt = gt_pcs[idx_per_class]
            input_trans = transformed_shapes[idx_per_class]
            
            gt_loader = DataLoader(PointcloudDataset(input_gt, pc_transform=network_input_transformations['part_predictor']), batch_size=128, num_workers=10)        
            trans_loader = DataLoader(PointcloudDataset(input_trans, pc_transform=network_input_transformations['part_predictor']), batch_size=128, num_workers=10)
            
            # step-1 predict the parts of -all- shapes coupled with lang
            pred_parts_gt = shape_net_parts_segmentor_inference(evaluating_part_predictor, gt_loader, device=device)
            pred_parts_trans = shape_net_parts_segmentor_inference(evaluating_part_predictor, trans_loader, device=device)
            
            
            ## Collect data about the predicted/referred part before computing LGD

            # step-2 find the actual references that do part-specific language to focus            
            _, ref_parts_idx = mark_part_reference_in_sentences(tokens[idx_per_class], gt_classes[idx_per_class])
            rows_to_part_idx = OrderedDict()
            no_part_referred = []
            for i, v in enumerate(ref_parts_idx):
                if len(v) > 0:  # i.e., there is a part to be removed for this example
                    rows_to_part_idx[i] = v
                else:
                    no_part_referred.append(i)  # no part referred (so should be ignored from all localized-GD computations)

            # compute masks covering the points that belong to a referred part
            mask_gt = masks_of_referred_parts_for_pcs(input_gt, pred_parts_gt, rows_to_part_idx)
            mask_tr = masks_of_referred_parts_for_pcs(input_trans, pred_parts_trans, rows_to_part_idx)
            assert len(mask_gt) == len(mask_tr) & len(input_gt) == len(input_trans)
        
            # find shapes for which all points belong to the part(s), so the PC will be empty post its removal
            all_points_g = np.where((mask_gt == False).sum(1) == mask_gt.shape[1])[0]
            all_points_t = np.where((mask_tr == False).sum(1) == mask_tr.shape[1])[0]
            all_points = list(set(all_points_g).union(all_points_t))        
                    
            # Also find cases where the part-predictor did not find any point belonging to the referred parts in BOTH or EITHER source/target
            referred_part_not_predicted_at_all = []
            referred_part_not_predicted_in_one_of_them = []
            for i, p in enumerate(ref_parts_idx):
                pred_has_it = set(p) <= set(np.unique(pred_parts_gt[i]))
                gt_has_it = set(p) <= set(np.unique(pred_parts_trans[i]))
                
                if not pred_has_it and not gt_has_it:
                    referred_part_not_predicted_at_all.append(i)
                if not pred_has_it or not gt_has_it:                
                    referred_part_not_predicted_in_one_of_them.append(i)
            
            ## OK - NOW:
            # 
            # A. - let's compute localized measurements where **we exclude** the referred part
            
            # for this we should ignore CD on the following:
            ignore_excluding = no_part_referred                    # obvious
            ignore_excluding += all_points                         # if all points are assigned to referred parts the entire shape will be reduced to the empty set
            ignore_excluding += referred_part_not_predicted_at_all # the predictor did not find the part(s) in both of them, so there is nothing to exclude        
            lgd_stats[f"{shape_class}_without_n_ignore"] = len(set(ignore_excluding))
            
            ## let's put 1's on the masks of the above shape pairs to not have chamfer distance computation crash (we will ignore these results anyway post computation)
            mask_gt_excluding = mask_gt.copy()
            mask_tr_excluding = mask_tr.copy()
            mask_gt_excluding[ignore_excluding] = np.ones_like(mask_gt_excluding[ignore_excluding])
            mask_tr_excluding[ignore_excluding] = np.ones_like(mask_tr_excluding[ignore_excluding])
                                    
            # compute the cd on all points but the parts
            _, masked_cds = chamfer_dists_on_masked_pointclouds(input_gt, input_trans, mask_gt_excluding, mask_tr_excluding, bsize=2048, device=device)
            torch.cuda.empty_cache()
            masked_cd_mu = np.mean([mc for c, mc in enumerate(masked_cds) if c not in set(ignore_excluding)])
            score = round(masked_cd_mu * scale_chamfer_by, 3)
            
            lgd_scores['without_parts_per_class'][shape_class] = score
            logger.info(f'(l)-GD excluding part(s) ({shape_class}): {score}')
            
                    
            # All the following are Optional computations not really used in the paper.
            # B. - let's compute localized measurements where **we keep only** the referred parts
            
            # First complement the raw masks i.e., here 1's correspond to points of parts only
            mask_gt_including = np.array((1 - mask_gt), bool)
            mask_tr_including = np.array((1 - mask_tr), bool)
                    
            #Here we have to ignore a slightly different set of shape pairs
            ignore_including = no_part_referred                            # obvious
            ignore_including += referred_part_not_predicted_in_one_of_them # the predictor did not find the part(s) in either of them, so it does not make sense to compare                
            lgd_stats[f"{shape_class}_with_n_ignore"] = len(set(ignore_including))
                    
            mask_gt_including[ignore_including] = np.ones_like(mask_gt_including[ignore_including])    # change the masks, so you can compute Chamfer-Dist (but ignore result in the end)
            mask_tr_including[ignore_including] = np.ones_like(mask_tr_including[ignore_including])
                        
            _, masked_cds = chamfer_dists_on_masked_pointclouds(input_gt, input_trans, mask_gt_including, mask_tr_including, bsize=2048, device=device)
            torch.cuda.empty_cache()
            masked_cd_mu = np.mean([mc for c, mc in enumerate(masked_cds) if c not in set(ignore_including)])
            score = round(masked_cd_mu * scale_chamfer_by, 3)
                            
            lgd_scores['with_parts_per_class'][shape_class] = score        
            logger.info(f'(l)-GD on the part(s) ({shape_class}): {score}')
                
            # compute (again) the cd ignoring part masks altogether
            _, unmasked_cds = chamfer_dists(input_gt, input_trans, bsize=2048, device=device)
            score = np.mean([hc for c, hc in enumerate(unmasked_cds) if c not in set(ignore_including)])
            score = round(score * scale_chamfer_by, 3)
            logger.info(f'GD on entire shapes when both have predicted referred parts  ({shape_class}): {score}')                
            lgd_scores['GD_on_pairs_with_parts_per_class'][shape_class] = score
                    
            norm_score = np.mean([hc for c, hc in enumerate(unmasked_cds/ masked_cds) if c not in set(ignore_including)])        
            norm_score = round(norm_score, 3)
            lgd_scores['with_parts_normalized_per_class'][shape_class] = norm_score
            logger.info(f'(l)-GD on the part(s) normalized ({shape_class}): {norm_score}')
            
        results_on_metrics['lgd_per_shape_class'] = lgd_scores
        logger.info('\n---------')
        
        lgd_averages = dict()
        for metric in lgd_scores.keys():
            score = n_total = 0
            
            for shape_class in lgd_scores[metric]:                
                idx_per_class = (gt_classes[gt_classes == shape_class].index).tolist()
                n_pairs_total = len(idx_per_class)
                
                if 'without' in metric:
                    active_pairs_for_metric = n_pairs_total - lgd_stats[f"{shape_class}_without_n_ignore"]            
                else:
                    active_pairs_for_metric = n_pairs_total - lgd_stats[f"{shape_class}_with_n_ignore"]            
                    
                score += active_pairs_for_metric * lgd_scores[metric][shape_class]
                n_total +=active_pairs_for_metric
                            
            average = round(score / n_total, 3)
            name = metric.replace('per_class', 'average')
            logger.info(f"{name}: {average}")
            lgd_averages[name] = average    
            
        results_on_metrics['lgd_averages'] = lgd_averages
   
    return results_on_metrics

