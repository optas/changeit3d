"""

Note the xxxx_transform_point_clouds methods below, work like this:
    A. Use a latent-based ```direction-finder``` to find an "updated" latent code GIVEN:
            i) a starting shape (latent),
            ii) referential language that denotes the desired edit, and,
            iii) a transformation/edit magnitude that reflects how far your edit can go inside the latent space.
    B. Use the decoder of a ```{pc_ae, sgf, imnet}``` to decode i.e., reconstruct the updated latents.

"""

import torch
from collections import defaultdict

from ..utils.basics import iterate_in_chunks
from ..models.changeit3d_net import get_transformed_latent_code


@torch.no_grad()
def pc_ae_transform_point_clouds(pc_ae, direction_finder, data_loader, stimulus_index, scales=[1], device="cuda"):
    """

    Args:
        pc_ae:
        direction_finder:
        data_loader:
        stimulus_index: stimulus location in data_loader (e.g., is it the first or the second shape)
        scales: use them to multiply the edit_latent before you add it to the original latent, this way you can *manually* boost or attenuate the edit's effect
        device:

    Returns:
        Let's assume that:
            1) the input language/shape-dataset concerns the transformation for N shape-language pairs.
            2) the latent space is L-dimensional

        then,

         A dictionary carrying the following items:
            'z_codes' -> dict, carrying the updated *final* latent codes. The keys are the input magnitudes. Each value is
                an N x L numpy array.
            'recons' -> dict, the N decoded/reconstructed point-clouds. The keys are input magnitudes. Each value is
                an N x PC-points x 3 numpy array.
            'tokens' -> list of lists, the N sentences used to create the transformations
            'edit_latents' -> N x L numpy array, the latents corresponding to the edits (before adding them to each input)
            'magnitudes' -> N x 1 numpy array, carrying the magnitudes the editing network guessed for each input.
    """

    results = get_transformed_latent_code(direction_finder, data_loader, stimulus_index, scales=scales, device=device)

    pc_ae.eval()
    all_recons = defaultdict(list)

    for key, val in results['z_codes'].items():
        recons = pc_ae.decoder(torch.from_numpy(val).to(device))
        recons = recons.view([len(recons), -1, 3]).cpu()
        all_recons[key].append(recons)

    for key in all_recons:
        all_recons[key] = torch.cat(all_recons[key]).numpy()

    results['recons'] = all_recons
    return results


@torch.no_grad()
def sgf_transform_point_clouds(sgf_trainer, direction_finder, data_loader, stimulus_index, scales=[1],
                               n_points_per_shape=2048, device="cuda", batch_size=64, max_recon_batches=None):
    """ See pc_ae_transform_point_clouds
    """

    results = get_transformed_latent_code(direction_finder, data_loader, stimulus_index, scales=scales,
                                          device=device)

    sgf_trainer.encoder.eval()
    sgf_trainer.score_net.eval()
    all_recons = defaultdict(list)
    
    for magnitude, all_z_of_magnitude in results['z_codes'].items():
        recons_of_magnitude = []
        cnt = 0
        for x in iterate_in_chunks(range(len(all_z_of_magnitude)), batch_size):
            z = torch.from_numpy(all_z_of_magnitude[x]).to(device)
            recons, _ = sgf_trainer.langevin_dynamics(z, num_points=n_points_per_shape)
            recons_of_magnitude.append(recons.cpu())
            cnt += 1
            if max_recon_batches is not None and cnt >= max_recon_batches:
                break
        all_recons[magnitude] = torch.cat(recons_of_magnitude).numpy()

    results['recons'] = all_recons
    return results



@torch.no_grad()
def imnet_transform_point_clouds(IMNET, direction_finder, data_loader, stimulus_index, scales=[1], device="cuda"):
    """ See pc_ae_transform_point_clouds 
    """
    
    results = get_transformed_latent_code(direction_finder, data_loader, stimulus_index, scales=scales,
                                          device=device)
    print('RECONSTRUCTION IS NOT YET IMPLEMENTED!')
    return results

    IMNET.eval()
    all_recons = defaultdict(list)
    for key, val in results['z_codes'].items():
        recons = IMNET(torch.from_numpy(val).to(device))
        recons = recons.view([len(recons), -1, 3]).cpu()
        all_recons[key].append(recons)

    for key in all_recons:
        all_recons[key] = torch.cat(all_recons[key]).numpy()

    results['recons'] = all_recons
    return results


