import sys
import dill
    

def initialize_and_load_sgf(sgf_code_dir=None, sgf_ae_ui_pkl=None):
    """loading a pretrained SGF-AE in the changeit3d repository/environment
        
        See the README.md for how to install SGF in changeit3d repo ( https://github.com/optas/ShapeGF )

    Args:
        sgf_code_dir (string): where you installed the ShapeGF repository
        sgf_ae_ui_pkl (string): the SGF-latent-interface-pub.pkl file you generated per the instructions of the above repo
            (upon running there: `python latents_interface.py`)
        device (string, optional): cuda device to run.
        
    Returns:
        the pretrained SGF-AE
    """    
    
    if sgf_code_dir is None:
       sgf_code_dir = '/home/panos/Git_Repos/ShapeGF'
    
    sys.path.append(sgf_code_dir)
    
    if sgf_ae_ui_pkl is None:
        sgf_ae_ui_pkl = '/home/panos/Git_Repos/ShapeGF/SGF-latent-interface-pub.pkl'
    
    try:
        with open(sgf_ae_ui_pkl, "rb") as f:
            ae = dill.load(f)
    except FileNotFoundError:
        print('You have to install first the Shape GF (adapted) repository. Look instructions: https://github.com/optas/ShapeGF')
    except Exception:
        raise
        
    # if device is not None:        
    #     ae.trainer.encoder = ae.trainer.encoder.to(device)
    #     ae.trainer.score_net = ae.trainer.score_net.to(device)
    #     ae.trainer.device = device
    return ae