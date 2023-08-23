
## Basic interface to load a pretrained ImNet with ShapeTalk

Once you have followed the steps in [Im-Net-ShapeTalkPretraining repo](https://github.com/ianhuang0630/IM-Net-ShapeTalkPretraining) to install it to your system and you have downloaded the pretrained weights of ImNet with Shapetalk, you can load the underlying UI in your current environment (e.g., in changeIt3D) like this:

```python
from changeit3d.external_tools.imnet import loader

imnet_ae = loader.initialize_and_load_imnet(imnet_code_dir=<location_you_installed_above_repo>, imnet_ae_ui_pkl=<location_you_generated_pkl_file_per_above_repo>)
imnet_ae.eval_z(<latent-vectors>) # to decode latent vectors like those corresponding to an ImNet pretrained with ShapeTalk 
```

(You can use **changeit3d.scripts.download_pretrained_nets.py** to downloaded pre-extracted shape latents for a ShapeTalk pretrained ImNet, among other pretrained/pre-extracted shape latent representations).
