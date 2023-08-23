import sys
import dill
    
def initialize_and_load_imnet(imnet_code_dir=None, imnet_ae_ui_pkl=None, device=None):
    """loading a pretrained ImNet-AE in the changeit3d repository/environment
        
        See the README.md for how to install ImNet in changeit3d repo ( https://github.com/ianhuang0630/IM-Net-ShapeTalkPretraining )

    Args:
        imnet_code_dir (string): where you installed the IM-Net-ShapeTalkPretraining repository
        imnet_ae_ui_pkl (string): the IMNET-latent-interface-ld3de-pub.pkl file you generated per the instructions of the above repo
            (upon running there: `./get_pickled_AE_interface.sh`)
        device (string): cuda device to run.
        
    Returns:
        the pretrained ImNet-AE
    """    
    
    if imnet_code_dir is None:
       imnet_code_dir = '/home/panos/Git_Repos/IM-Net-ShapeTalkPretraining'
    
    if imnet_ae_ui_pkl is None:
        imnet_ae_ui_pkl = '/home/panos/Git_Repos/IM-Net-ShapeTalkPretraining/IMNET-latent-interface-ld3de-pub.pkl'
        
    sys.path.append(imnet_code_dir)
                
    with open(imnet_ae_ui_pkl, "rb") as f:
        ae = dill.load(f)
    
    if device is not None:
        ae.im_ae.device = device
        ae.im_ae.im_network = ae.im_ae.im_network.to(device)
    return ae